#!/usr/bin/env python3
"""Re-crawl notice bodies with OCR and replace Supabase body values."""

from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crawling.crawler import get_post_content  # noqa: E402
from crawling.supabase_store import SOURCE, _connect  # noqa: E402

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _clean_body(body: str) -> str:
    return _CONTROL_RE.sub("", body)


def _load_rows(
    start_date: date,
    end_date: date,
    hansung_only: bool,
    ids: list[int] | None,
) -> list[dict[str, Any]]:
    filters = [
        "source = %s",
        "posted_at >= %s",
        "posted_at <= %s",
    ]
    params: list[Any] = [SOURCE, start_date, end_date]

    if hansung_only:
        filters.append("url like %s")
        params.append("https://www.hansung.ac.kr/%")

    if ids:
        filters.append("id = any(%s)")
        params.append(ids)

    sql = f"""
        select id, notice_id, title, url, posted_at, length(coalesce(body, '')) as old_body_len
        from notices
        where {' and '.join(filters)}
        order by posted_at asc nulls last, id asc
    """
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    return [
        {
            "id": row[0],
            "notice_id": row[1],
            "title": row[2],
            "url": row[3],
            "posted_at": row[4],
            "old_body_len": row[5],
        }
        for row in rows
    ]


def _update_body(notice_id: int, body: str) -> None:
    body = _clean_body(body)
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                update notices
                set body = %s,
                    raw = jsonb_set(coalesce(raw, '{}'::jsonb), '{body}', to_jsonb(%s::text), true),
                    updated_at = now()
                where id = %s
                """,
                (body, body, notice_id),
            )
        conn.commit()


def backfill(
    start_date: date,
    end_date: date,
    *,
    hansung_only: bool,
    dry_run: bool,
    delay: float,
    limit: int | None,
    ids: list[int] | None,
) -> int:
    rows = _load_rows(start_date, end_date, hansung_only, ids)
    if limit is not None:
        rows = rows[:limit]

    print(
        f"Target notices: {len(rows)} "
        f"({start_date.isoformat()}..{end_date.isoformat()}, hansung_only={hansung_only})",
        flush=True,
    )
    if dry_run:
        for row in rows[:20]:
            print(
                f"DRY {row['id']}\t{row['posted_at']}\t{row['notice_id']}\t"
                f"old_len={row['old_body_len']}\t{row['title'][:80]}",
                flush=True,
            )
        if len(rows) > 20:
            print(f"... {len(rows) - 20} more", flush=True)
        return 0

    updated = 0
    skipped_empty = 0
    failed = 0

    for index, row in enumerate(rows, start=1):
        title = row["title"] or ""
        print(
            f"[{index}/{len(rows)}] id={row['id']} notice_id={row['notice_id']} "
            f"date={row['posted_at']} old_len={row['old_body_len']} {title[:70]}",
            flush=True,
        )
        try:
            new_body = get_post_content(row["url"])
            new_body = _clean_body(new_body or "")
            new_len = len(new_body or "")
            if not new_body:
                skipped_empty += 1
                print("  skip: extracted empty body", flush=True)
            else:
                _update_body(row["id"], new_body)
                updated += 1
                has_ocr = "[이미지 OCR]" in new_body or "[PDF OCR]" in new_body or "[첨부PDF]" in new_body
                print(f"  updated: new_len={new_len} ocr_marker={has_ocr}", flush=True)
        except Exception as exc:
            failed += 1
            print(f"  failed: {type(exc).__name__}: {exc}", flush=True)

        if delay > 0:
            time.sleep(delay)

    print(
        f"Done. updated={updated}, skipped_empty={skipped_empty}, failed={failed}, total={len(rows)}",
        flush=True,
    )
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill Supabase notice bodies by re-running crawler OCR.",
    )
    parser.add_argument("--start-date", required=True, type=_parse_date)
    parser.add_argument("--end-date", required=True, type=_parse_date)
    parser.add_argument("--include-external", action="store_true", help="Include non-hansung URLs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ids", nargs="+", type=int, help="Only process specific notices.id values")
    args = parser.parse_args()

    if args.end_date < args.start_date:
        raise ValueError("--end-date must be on or after --start-date")

    backfill(
        args.start_date,
        args.end_date,
        hansung_only=not args.include_external,
        dry_run=args.dry_run,
        delay=args.delay,
        limit=args.limit,
        ids=args.ids,
    )


if __name__ == "__main__":
    main()
