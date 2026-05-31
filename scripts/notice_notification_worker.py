#!/usr/bin/env python3
"""Send iMessage notifications for newly crawled Supabase notices."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from realtime import (
    AsyncRealtimeClient,
    RealtimePostgresChangesListenEvent,
    RealtimeSubscribeStates,
)
from supabase import create_client

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

LOG_PATH = ROOT / "scripts" / "notice_notification_worker.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
    ],
)
logger = logging.getLogger("notice_notification_worker")


def get_supabase_client():
    url = os.getenv("SUPABASE_URL")
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if os.getenv("SUPABASE_DB_URL") and not service_key:
        return None
    key = service_key or os.getenv("SUPABASE_KEY")
    if not url or not key:
        if os.getenv("SUPABASE_DB_URL"):
            return None
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY or SUPABASE_KEY are required.")
    return create_client(url, key)


def get_realtime_config() -> tuple[str, str] | None:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
    if not url or not key or url.startswith("your_") or key.startswith("your_"):
        return None
    return f"{url.rstrip('/')}/realtime/v1", key


def ensure_realtime_publication() -> None:
    """Best-effort: make public.notices visible to Supabase Realtime."""
    db_url = os.getenv("SUPABASE_DB_URL")
    if not db_url:
        return
    try:
        import psycopg

        with psycopg.connect(db_url, connect_timeout=10) as conn:
            with conn.cursor() as cur:
                cur.execute("alter publication supabase_realtime add table public.notices")
            conn.commit()
    except Exception as exc:
        text = str(exc).lower()
        if "already member" not in text and "duplicate_object" not in text:
            logger.info("Realtime publication 확인 생략/실패: %s", exc)


def load_recent_notices(client, lookback_minutes: int, limit: int) -> list[dict[str, Any]]:
    since = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
    if client is None:
        import psycopg

        with psycopg.connect(os.environ["SUPABASE_DB_URL"], connect_timeout=10) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select id, notice_id, title, url, posted_at, category,
                           category_type, job_types, body, notice_score, embedding, crawled_at
                    from public.notices
                    where crawled_at >= %s
                    order by crawled_at desc
                    limit %s
                    """,
                    (since, limit),
                )
                columns = [desc.name for desc in cur.description]
                return [dict(zip(columns, row)) for row in cur.fetchall()]

    res = (
        client.table("notices")
        .select("id,notice_id,title,url,posted_at,category,category_type,job_types,body,notice_score,embedding,crawled_at")
        .gte("crawled_at", since.isoformat())
        .order("crawled_at", desc=True)
        .limit(limit)
        .execute()
    )
    return res.data or []


def load_users(client) -> list[dict[str, Any]]:
    if client is None:
        import psycopg

        with psycopg.connect(os.environ["SUPABASE_DB_URL"], connect_timeout=10) as conn:
            with conn.cursor() as cur:
                cur.execute("select user_id, name, phone, interests, track from public.users")
                columns = [desc.name for desc in cur.description]
                return [
                    user
                    for user in (dict(zip(columns, row)) for row in cur.fetchall())
                    if user.get("phone")
                ]

    res = (
        client.table("users")
        .select("user_id,name,phone,interests,track")
        .execute()
    )
    return [user for user in (res.data or []) if user.get("phone")]


def load_max_notice_score(client) -> float:
    if client is None:
        import psycopg

        with psycopg.connect(os.environ["SUPABASE_DB_URL"], connect_timeout=10) as conn:
            with conn.cursor() as cur:
                cur.execute("select max(coalesce(notice_score, 0)) from public.notices")
                value = (cur.fetchone() or [0])[0]
                return float(value or 1.0)

    res = (
        client.table("notices")
        .select("notice_score")
        .order("notice_score", desc=True)
        .limit(1)
        .execute()
    )
    data = res.data or []
    if not data:
        return 1.0
    return float(data[0].get("notice_score") or 1.0)


def parse_json_array(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return [value] if value else []
    return []


def parse_embedding(value: Any) -> list[float] | None:
    parsed = parse_json_array(value)
    if not parsed:
        return None
    try:
        return [float(item) for item in parsed]
    except (TypeError, ValueError):
        return None


def build_notification_targets(
    notices: list[dict[str, Any]],
    users: list[dict[str, Any]],
    max_notice_score: float,
    threshold: float,
) -> list[dict[str, Any]]:
    from recommend import (
        CATEGORY_WEIGHT,
        FILTER_CATEGORIES,
        JOB_TYPE_WEIGHT,
        MODEL_WEIGHT,
        OLD_CATEGORIES,
        PENALTY_CATEGORIES,
        SCORE_WEIGHT,
        get_job_score,
        load_two_tower_model,
    )

    sbert, model, device = load_two_tower_model()
    max_notice_score = max(max_notice_score, 1.0)
    targets: list[dict[str, Any]] = []
    user_vec_cache: dict[str, Any] = {}

    for user in users:
        if not user or not user.get("phone"):
            continue

        interests = parse_json_array(user.get("interests"))
        rec_interests = [interest for interest in interests if interest not in FILTER_CATEGORIES]
        if not rec_interests:
            continue

        user_id = user.get("user_id") or user.get("id") or user.get("phone")
        user_text = f"{user.get('college', '')} {user.get('track', '')} {user.get('grade', '')} 관심사: {', '.join(rec_interests)}"
        if user_id not in user_vec_cache:
            import torch

            user_emb = sbert.encode([user_text], convert_to_numpy=True)
            user_tensor = torch.tensor(user_emb, dtype=torch.float).to(device)
            with torch.no_grad():
                user_vec_cache[user_id] = model.forward_user(user_tensor).cpu().numpy()

        user_vec = user_vec_cache[user_id]

        for notice in notices:
            category = notice.get("category", "")
            if category in FILTER_CATEGORIES or category in OLD_CATEGORIES:
                continue
            if float(notice.get("notice_score") or 0.0) <= 0.01:
                continue

            item_emb = parse_embedding(notice.get("embedding"))
            if item_emb is None:
                logger.info("공지 embedding 없음, 알림 점수 계산 제외: notice=%s", notice.get("id"))
                continue

            import numpy as np

            sim_score = float(np.dot(np.array(item_emb, dtype=np.float32), user_vec.T).flatten()[0])
            sim_norm = (sim_score + 1.0) / 2.0
            n_score = float(notice.get("notice_score") or 0.0) / max_notice_score

            if category in rec_interests:
                cat_score = 1.0
            elif category in PENALTY_CATEGORIES:
                cat_score = -1.0
            else:
                cat_score = 0.0

            job_score = get_job_score(user.get("track", ""), notice)
            final_score = (
                MODEL_WEIGHT * sim_norm
                + CATEGORY_WEIGHT * cat_score
                + JOB_TYPE_WEIGHT * job_score
                + SCORE_WEIGHT * n_score
            )
            if final_score < threshold:
                continue

            category_type = parse_json_array(notice.get("category_type") or notice.get("job_types"))
            targets.append({
                "notice_db_id": notice.get("id"),
                "notice_id": notice.get("notice_id", ""),
                "title": notice.get("title", ""),
                "category": category,
                "category_type": category_type,
                "url": notice.get("url", ""),
                "user_id": user_id,
                "user_name": user.get("name", ""),
                "phone": user.get("phone"),
                "track": user.get("track", ""),
                "interests": interests,
                "final_score": final_score,
                "sim_score": sim_norm,
                "cat_score": cat_score,
                "job_score": job_score,
                "n_score": n_score,
            })

    return targets


def delivery_exists(client, notice_db_id: int, user_id: str, channel: str) -> bool:
    if client is None:
        import psycopg

        with psycopg.connect(os.environ["SUPABASE_DB_URL"], connect_timeout=10) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select 1
                    from public.notification_deliveries
                    where notice_db_id = %s and user_id = %s and channel = %s and status = 'sent'
                    limit 1
                    """,
                    (notice_db_id, user_id, channel),
                )
                return cur.fetchone() is not None

    res = (
        client.table("notification_deliveries")
        .select("id,status")
        .eq("notice_db_id", notice_db_id)
        .eq("user_id", user_id)
        .eq("channel", channel)
        .eq("status", "sent")
        .limit(1)
        .execute()
    )
    return bool(res.data)


def record_delivery(
    client,
    target: dict[str, Any],
    channel: str,
    status: str,
    error: str | None = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    if client is None:
        import psycopg

        with psycopg.connect(os.environ["SUPABASE_DB_URL"], connect_timeout=10) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into public.notification_deliveries (
                        notice_db_id, notice_id, user_id, channel, status,
                        error, sent_at, updated_at
                    )
                    values (%s, %s, %s, %s, %s, %s, %s, %s)
                    on conflict (notice_db_id, user_id, channel) do update set
                        status = excluded.status,
                        error = excluded.error,
                        sent_at = excluded.sent_at,
                        updated_at = excluded.updated_at
                    """,
                    (
                        target["notice_db_id"],
                        target.get("notice_id"),
                        target["user_id"],
                        channel,
                        status,
                        error,
                        now if status == "sent" else None,
                        now,
                    ),
                )
            conn.commit()
        return

    row = {
        "notice_db_id": target["notice_db_id"],
        "notice_id": target.get("notice_id"),
        "user_id": target["user_id"],
        "channel": channel,
        "status": status,
        "error": error,
        "sent_at": now if status == "sent" else None,
        "updated_at": now,
    }
    client.table("notification_deliveries").upsert(
        row,
        on_conflict="notice_db_id,user_id,channel",
    ).execute()


def format_message(target: dict[str, Any]) -> str:
    track = target.get("track") or "관심 트랙"
    name = target.get("user_name") or "사용자"
    title = target.get("title", "").strip()
    category = target.get("category") or "공지"
    url = target.get("url")
    message = f"[상상파인더] {track}의 {name}님이 관심 있으신 {category} 분야의 새 공지가 올라왔어요!\n\n{title}"
    return f"{message}\n{url}" if url else message


def normalize_phone_for_messages(phone: str) -> str:
    """Format Korean mobile numbers for Messages, preferring E.164."""
    digits = re.sub(r"\D", "", phone or "")
    if not digits:
        return phone
    if digits.startswith("010") and len(digits) == 11:
        return f"+82{digits[1:]}"
    if digits.startswith("82"):
        return f"+{digits}"
    if phone.strip().startswith("+"):
        return phone.strip()
    return f"+{digits}" if len(digits) >= 10 else phone


def escape_applescript(value: str) -> str:
    return (
        (value or "")
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )


def has_recent_imessage_success(recipient: str) -> bool:
    """Use the local Messages DB to avoid iMessage for handles that only fail."""
    db_path = Path.home() / "Library" / "Messages" / "chat.db"
    if not db_path.exists():
        return False
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                select count(*)
                from message m
                join handle h on m.handle_id = h.rowid
                where h.id = ?
                  and m.service in ('iMessage', 'iMessageLite')
                  and coalesce(m.error, 0) = 0
                """,
                (recipient,),
            )
            return (cur.fetchone() or [0])[0] > 0
    except Exception:
        return False


def verify_recent_outgoing_message(recipient: str, message: str) -> None:
    """Raise if Messages recorded an immediate send error for this message."""
    db_path = Path.home() / "Library" / "Messages" / "chat.db"
    if not db_path.exists():
        return
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                select coalesce(m.error, 0)
                from message m
                join handle h on m.handle_id = h.rowid
                where h.id = ?
                  and m.text = ?
                order by m.date desc
                limit 1
                """,
                (recipient, message),
            )
            row = cur.fetchone()
            if row and row[0] != 0:
                raise RuntimeError(f"Messages recorded delivery error {row[0]} for {recipient}")
    except RuntimeError:
        raise
    except Exception:
        return


def send_messages_service(recipient: str, message: str, service_type: str) -> None:
    """Submit one message through a specific Messages account type."""
    safe_recipient = escape_applescript(recipient)
    safe_service_type = "iMessage" if service_type == "iMessage" else "SMS"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as tmp:
        tmp.write(message)
        message_path = tmp.name
    script = f"""
tell application "Messages"
    set messageText to (read (POSIX file "{escape_applescript(message_path)}") as «class utf8»)
    set targetService to first account whose service type = {safe_service_type} and enabled is true
    set targetBuddy to participant "{safe_recipient}" of targetService
    send messageText to targetBuddy
    return "success:{safe_service_type}:{safe_recipient}"
end tell
"""
    try:
        result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
        if result.returncode != 0:
            error = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(error or f"osascript exited with {result.returncode}")
        time.sleep(2)
        verify_recent_outgoing_message(recipient, message)
    finally:
        try:
            os.unlink(message_path)
        except OSError:
            pass


def send_imessage(phone: str, message: str) -> str:
    """Try iMessage first, falling back to SMS on any detected failure."""
    recipient = normalize_phone_for_messages(phone)

    try:
        send_messages_service(recipient, message, "iMessage")
        return "iMessage"
    except Exception as imessage_error:
        logger.warning(
            "iMessage 발송 실패, SMS fallback 시도: recipient=%s error=%s",
            recipient,
            imessage_error,
        )
        send_messages_service(recipient, message, "SMS")
        return "SMS"


def process_notices_for_notifications(
    args: argparse.Namespace,
    client,
    notices: list[dict[str, Any]],
    source: str,
) -> int:
    if not notices:
        logger.info("%s: 신규 후보 공지 없음", source)
        return 0

    users = load_users(client)
    if not users:
        logger.info("%s: 알림 대상 users 없음", source)
        return 0

    max_notice_score = load_max_notice_score(client)
    targets = build_notification_targets(
        notices,
        users=users,
        max_notice_score=max_notice_score,
        threshold=args.score_threshold,
    )
    logger.info(
        "%s: 공지 %d건, 유저 %d명, 추천 점수 %.2f 이상 대상 %d건",
        source,
        len(notices),
        len(users),
        args.score_threshold,
        len(targets),
    )

    sent = 0
    for target in targets:
        notice_db_id = target.get("notice_db_id")
        user_id = target.get("user_id")
        phone = target.get("phone")
        if not notice_db_id or not user_id or not phone:
            continue
        if delivery_exists(client, notice_db_id, user_id, args.channel):
            continue

        message = format_message(target)
        if args.dry_run:
            logger.info("[DRY-RUN] %s -> %s", phone, message.replace("\n", " / "))
            continue

        try:
            route = send_imessage(phone, message)
            record_delivery(client, target, args.channel, "sent")
            sent += 1
            logger.info(
                "발송 완료: notice=%s user=%s phone=%s route=%s final_score=%.3f",
                notice_db_id,
                user_id,
                phone,
                route,
                target.get("final_score", 0.0),
            )
        except Exception as exc:
            record_delivery(client, target, args.channel, "failed", str(exc)[:1000])
            logger.exception("발송 실패: notice=%s user=%s phone=%s", notice_db_id, user_id, phone)

        time.sleep(args.send_delay)

    logger.info("이번 실행 발송 완료: %d건", sent)
    return sent


def run_once(args: argparse.Namespace) -> int:
    client = get_supabase_client()
    notices = load_recent_notices(client, args.lookback_minutes, args.limit)
    return process_notices_for_notifications(args, client, notices, "polling")


async def run_polling_loop(args: argparse.Namespace, lock: asyncio.Lock) -> None:
    while True:
        await asyncio.sleep(args.interval)
        try:
            async with lock:
                await asyncio.to_thread(run_once, args)
        except Exception:
            logger.exception("polling 보정 실행 실패")


async def run_realtime_daemon(args: argparse.Namespace) -> None:
    lock = asyncio.Lock()

    logger.info("데몬 시작: Realtime 즉시 발송 + %d초마다 polling 보정", args.interval)
    async with lock:
        await asyncio.to_thread(run_once, args)

    polling_task = asyncio.create_task(run_polling_loop(args, lock))
    realtime_config = get_realtime_config()
    if not realtime_config or args.polling_only:
        logger.info("Realtime 비활성화: polling 보정만 실행합니다.")
        await polling_task
        return

    ensure_realtime_publication()
    realtime_url, realtime_key = realtime_config

    while True:
        rt = AsyncRealtimeClient(realtime_url, token=realtime_key, auto_reconnect=True)
        try:
            channel = rt.channel("sangsangfinder-notice-inserts")

            def on_insert(payload: dict[str, Any]) -> None:
                record = (payload.get("data") or {}).get("record") or payload.get("record")
                if not record:
                    logger.warning("Realtime INSERT payload에 record가 없습니다: %s", payload)
                    return

                async def handle_insert() -> None:
                    async with lock:
                        client = get_supabase_client()
                        await asyncio.to_thread(
                            process_notices_for_notifications,
                            args,
                            client,
                            [record],
                            "realtime",
                        )

                asyncio.create_task(handle_insert())

            def on_subscribe(status, error) -> None:
                if status == RealtimeSubscribeStates.SUBSCRIBED:
                    logger.info("Supabase Realtime 구독 시작: public.notices INSERT")
                elif error:
                    logger.warning("Supabase Realtime 구독 상태=%s error=%s", status, error)
                else:
                    logger.info("Supabase Realtime 구독 상태=%s", status)

            channel.on_postgres_changes(
                RealtimePostgresChangesListenEvent.Insert,
                on_insert,
                table="notices",
                schema="public",
            )
            await channel.subscribe(on_subscribe)
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            await rt.close()
            polling_task.cancel()
            raise
        except Exception:
            logger.exception("Realtime 루프 실패, 10초 후 재연결")
            try:
                await rt.close()
            except Exception:
                pass
            await asyncio.sleep(10)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Supabase 신규 공지를 users 관심사와 매칭해 iMessage로 발송합니다.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--daemon", action="store_true", help="Realtime 구독 + polling 보정 반복 실행")
    parser.add_argument("--interval", type=int, default=900, help="polling 보정 실행 간격(초)")
    parser.add_argument("--lookback-minutes", type=int, default=1440, help="신규 후보 공지 조회 범위")
    parser.add_argument("--limit", type=int, default=100, help="한 번에 조회할 최대 공지 수")
    parser.add_argument("--send-delay", type=float, default=0.5, help="발송 사이 대기 시간(초)")
    parser.add_argument("--channel", default="imessage", help="notification_deliveries 채널명")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="알림 발송 Two-Tower 최종점수 기준")
    parser.add_argument("--dry-run", action="store_true", help="실제 발송/이력 저장 없이 대상만 로그 출력")
    parser.add_argument("--polling-only", action="store_true", help="Realtime 없이 polling 보정만 실행")
    args = parser.parse_args()

    if args.daemon:
        asyncio.run(run_realtime_daemon(args))
        return

    try:
        run_once(args)
    except Exception:
        logger.exception("알림 워커 실행 실패")


if __name__ == "__main__":
    main()
