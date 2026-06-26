#!/usr/bin/env python3
"""Send iMessage notifications for new or updated Supabase notices."""

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


def ensure_notification_delivery_schema() -> None:
    """Best-effort: allow one delivery per notice/user/channel/category."""
    db_url = os.getenv("SUPABASE_DB_URL")
    if not db_url:
        return
    try:
        import psycopg

        with psycopg.connect(db_url, connect_timeout=10) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    alter table public.notification_deliveries
                        add column if not exists category text not null default '';
                    alter table public.notification_deliveries
                        drop constraint if exists notification_deliveries_notice_db_id_user_id_channel_key;
                    create unique index if not exists idx_notification_deliveries_notice_user_channel_category
                        on public.notification_deliveries (notice_db_id, user_id, channel, category);
                    """
                )
            conn.commit()
    except Exception as exc:
        logger.info("notification_deliveries 카테고리 스키마 확인 생략/실패: %s", exc)


def load_recent_notices(client, lookback_minutes: int, limit: int) -> list[dict[str, Any]]:
    since = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
    if client is None:
        import psycopg

        with psycopg.connect(os.environ["SUPABASE_DB_URL"], connect_timeout=10) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select id, notice_id, title, url, posted_at, category,
                           category_type, job_types, body, notice_score, embedding, crawled_at, updated_at
                    from public.notices
                    where crawled_at >= %s
                       or updated_at >= %s
                    order by greatest(crawled_at, updated_at) desc
                    limit %s
                    """,
                    (since, since, limit),
                )
                columns = [desc.name for desc in cur.description]
                return [dict(zip(columns, row)) for row in cur.fetchall()]

    res = (
        client.table("notices")
        .select("id,notice_id,title,url,posted_at,category,category_type,job_types,body,notice_score,embedding,crawled_at,updated_at")
        .or_(f"crawled_at.gte.{since.isoformat()},updated_at.gte.{since.isoformat()}")
        .order("updated_at", desc=True)
        .limit(limit)
        .execute()
    )
    return res.data or []


def load_users(client) -> list[dict[str, Any]]:
    if client is None:
        import psycopg

        for query in (
            "select user_id, name, phone, interests, track, college, grade from public.users",
            "select user_id, name, phone, interests, track from public.users",
        ):
            try:
                with psycopg.connect(os.environ["SUPABASE_DB_URL"], connect_timeout=10) as conn:
                    with conn.cursor() as cur:
                        cur.execute(query)
                        columns = [desc.name for desc in cur.description]
                        return [
                            user
                            for user in (dict(zip(columns, row)) for row in cur.fetchall())
                            if user.get("phone")
                        ]
            except Exception as exc:
                if "college" not in query:
                    raise
                logger.info("users college/grade 컬럼 조회 실패, 기본 컬럼으로 재시도: %s", exc)

    for select_columns in (
        "user_id,name,phone,interests,track,college,grade",
        "user_id,name,phone,interests,track",
    ):
        try:
            res = client.table("users").select(select_columns).execute()
            return [user for user in (res.data or []) if user.get("phone")]
        except Exception as exc:
            if "college" not in select_columns:
                raise
            logger.info("users college/grade 컬럼 조회 실패, 기본 컬럼으로 재시도: %s", exc)

    return []


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


def get_notification_targets(notices: list[dict], users: list[dict] | None = None) -> list[dict]:
    """
    신규 공지 리스트를 받아 Two-Tower 점수 기반 표준편차로 알림 대상자 반환.
    """
    import numpy as np
    import torch

    from recommend import get_supabase, load_two_tower_model

    if users is None:
        try:
            res = get_supabase().table("users").select(
                "user_id,name,phone,interests,track,college,grade"
            ).execute()
            users = res.data or []
        except Exception as e:
            logger.info("users 테이블 로드 오류: %s", e)
            return []

    if not users or not notices:
        return []

    try:
        sbert, model, device = load_two_tower_model()
    except Exception as e:
        logger.info("모델 로드 오류: %s", e)
        return []

    targets = []

    for notice in notices:
        category = notice.get("category", "")
        category_type = parse_json_array(notice.get("category_type") or notice.get("job_types"))

        notice_emb = parse_embedding(notice.get("embedding"))
        if notice_emb is None:
            logger.info("공지 embedding 없음, 알림 점수 계산 제외: notice=%s", notice.get("id"))
            continue
        item_vec = np.array(notice_emb, dtype=np.float32)

        user_scores = []
        for user in users:
            interests = parse_json_array(user.get("interests"))

            if category not in interests:
                user_scores.append((user, 0.0))
                continue

            college = user.get("college", "")
            track = user.get("track", "")
            grade = user.get("grade", "")

            user_text = f"{college} {track} {grade} 관심사: {', '.join(interests)}"
            user_emb = sbert.encode([user_text], convert_to_numpy=True)
            user_tensor = torch.tensor(user_emb, dtype=torch.float).to(device)
            with torch.no_grad():
                user_vec = model.forward_user(user_tensor).cpu().numpy()[0]

            sim_score = float(np.dot(item_vec, user_vec))
            sim_norm = (sim_score + 1.0) / 2.0
            user_scores.append((user, sim_norm))

        valid_scores = [(user, score) for user, score in user_scores if score > 0]
        if not valid_scores:
            continue

        scores_arr = np.array([score for _, score in valid_scores])
        mean = scores_arr.mean()
        std = scores_arr.std()
        threshold = mean + 0.5 * std

        logger.info(
            "[알림] %s | 평균:%.3f 표준편차:%.3f 컷:%.3f",
            notice.get("title", "")[:30],
            mean,
            std,
            threshold,
        )

        for user, score in valid_scores:
            if score < threshold:
                continue
            if not user.get("phone"):
                continue

            interests = parse_json_array(user.get("interests"))
            targets.append({
                "notice_db_id": notice.get("id"),
                "notice_id": notice.get("notice_id", ""),
                "title": notice.get("title", ""),
                "category": category,
                "category_type": category_type,
                "url": notice.get("url", ""),
                "user_id": user.get("user_id") or user.get("phone"),
                "user_name": user.get("name", ""),
                "phone": user.get("phone"),
                "track": user.get("track", ""),
                "interests": interests,
                "score": round(float(score), 4),
            })

    return targets


def delivery_exists(client, notice_db_id: int, user_id: str, channel: str, category: str) -> bool:
    if client is None:
        import psycopg

        with psycopg.connect(os.environ["SUPABASE_DB_URL"], connect_timeout=10) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select 1
                    from public.notification_deliveries
                    where notice_db_id = %s
                      and user_id = %s
                      and channel = %s
                      and category = %s
                      and status = 'sent'
                    limit 1
                    """,
                    (notice_db_id, user_id, channel, category or ""),
                )
                return cur.fetchone() is not None

    res = (
        client.table("notification_deliveries")
        .select("id,status")
        .eq("notice_db_id", notice_db_id)
        .eq("user_id", user_id)
        .eq("channel", channel)
        .eq("category", category or "")
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
                        notice_db_id, notice_id, user_id, channel, category, status,
                        error, sent_at, updated_at
                    )
                    values (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    on conflict (notice_db_id, user_id, channel, category) do update set
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
                        target.get("category") or "",
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
        "category": target.get("category") or "",
        "status": status,
        "error": error,
        "sent_at": now if status == "sent" else None,
        "updated_at": now,
    }
    client.table("notification_deliveries").upsert(
        row,
        on_conflict="notice_db_id,user_id,channel,category",
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

    targets = get_notification_targets(notices, users=users)
    logger.info(
        "%s: 공지 %d건, 유저 %d명, 표준편차 컷 대상 %d건",
        source,
        len(notices),
        len(users),
        len(targets),
    )

    sent = 0
    for target in targets:
        notice_db_id = target.get("notice_db_id")
        user_id = target.get("user_id")
        phone = target.get("phone")
        if not notice_db_id or not user_id or not phone:
            continue
        if delivery_exists(client, notice_db_id, user_id, args.channel, target.get("category") or ""):
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
                "발송 완료: notice=%s user=%s phone=%s route=%s score=%.3f",
                notice_db_id,
                user_id,
                phone,
                route,
                target.get("score", 0.0),
            )
        except Exception as exc:
            record_delivery(client, target, args.channel, "failed", str(exc)[:1000])
            logger.exception("발송 실패: notice=%s user=%s phone=%s", notice_db_id, user_id, phone)

        time.sleep(args.send_delay)

    logger.info("이번 실행 발송 완료: %d건", sent)
    return sent


def run_once(args: argparse.Namespace) -> int:
    client = get_supabase_client()
    ensure_notification_delivery_schema()
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

    logger.info("데몬 시작: Realtime INSERT/UPDATE 즉시 발송 + %d초마다 polling 보정", args.interval)
    async with lock:
        await asyncio.to_thread(run_once, args)

    polling_task = asyncio.create_task(run_polling_loop(args, lock))
    realtime_config = get_realtime_config()
    if not realtime_config or args.polling_only:
        logger.info("Realtime 비활성화: polling 보정만 실행합니다.")
        await polling_task
        return

    ensure_realtime_publication()
    ensure_notification_delivery_schema()
    realtime_url, realtime_key = realtime_config

    while True:
        rt = AsyncRealtimeClient(realtime_url, token=realtime_key, auto_reconnect=True)
        try:
            channel = rt.channel("sangsangfinder-notice-changes")

            def on_notice_change(payload: dict[str, Any]) -> None:
                record = (payload.get("data") or {}).get("record") or payload.get("record")
                event_type = (
                    (payload.get("data") or {}).get("type")
                    or payload.get("eventType")
                    or payload.get("type")
                    or "CHANGE"
                )
                if not record:
                    logger.warning("Realtime %s payload에 record가 없습니다: %s", event_type, payload)
                    return

                async def handle_notice_change() -> None:
                    async with lock:
                        client = get_supabase_client()
                        await asyncio.to_thread(
                            process_notices_for_notifications,
                            args,
                            client,
                            [record],
                            f"realtime {event_type}",
                        )

                asyncio.create_task(handle_notice_change())

            def on_subscribe(status, error) -> None:
                if status == RealtimeSubscribeStates.SUBSCRIBED:
                    logger.info("Supabase Realtime 구독 시작: public.notices INSERT/UPDATE")
                elif error:
                    logger.warning("Supabase Realtime 구독 상태=%s error=%s", status, error)
                else:
                    logger.info("Supabase Realtime 구독 상태=%s", status)

            channel.on_postgres_changes(
                RealtimePostgresChangesListenEvent.Insert,
                on_notice_change,
                table="notices",
                schema="public",
            )
            channel.on_postgres_changes(
                RealtimePostgresChangesListenEvent.Update,
                on_notice_change,
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
    parser.add_argument("--score-threshold", type=float, default=0.5, help="deprecated: 표준편차 컷 방식에서는 사용하지 않음")
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
