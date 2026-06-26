#!/usr/bin/env python3
"""Recompute Supabase notice_score and pgvector embeddings for notices."""

from __future__ import annotations

import argparse
import sys
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

from crawling.notice_processor import calc_embedding, calc_notice_score, load_model  # noqa: E402
from crawling.supabase_store import SOURCE, _connect  # noqa: E402


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{float(value):.9g}" for value in values) + "]"


def _load_rows(start_date: date, end_date: date, ids: list[int] | None) -> list[dict[str, Any]]:
    filters = [
        "source = %s",
        "posted_at >= %s",
        "posted_at <= %s",
        "url like %s",
    ]
    params: list[Any] = [SOURCE, start_date, end_date, "https://www.hansung.ac.kr/%"]
    if ids:
        filters.append("id = any(%s)")
        params.append(ids)

    sql = f"""
        select id, notice_id, title, url, posted_at, posted_date_text, category, body, views
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
            "title": row[2] or "",
            "url": row[3] or "",
            "posted_at": row[4],
            "posted_date_text": row[5],
            "category": row[6] or "",
            "body": row[7] or "",
            "views": row[8] or 0,
        }
        for row in rows
    ]


def _update_embedding(notice_id: int, notice_score: float, embedding: list[float]) -> None:
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                update notices
                set notice_score = %s,
                    embedding = %s::vector,
                    updated_at = now()
                where id = %s
                """,
                (notice_score, _vector_literal(embedding), notice_id),
            )
        conn.commit()


def reindex(start_date: date, end_date: date, ids: list[int] | None, dry_run: bool) -> int:
    rows = _load_rows(start_date, end_date, ids)
    print(f"Target notices: {len(rows)} ({start_date.isoformat()}..{end_date.isoformat()})", flush=True)
    if dry_run:
        for row in rows[:20]:
            print(
                f"DRY {row['id']}\t{row['posted_at']}\t{row['notice_id']}\t"
                f"body_len={len(row['body'])}\t{row['title'][:80]}",
                flush=True,
            )
        if len(rows) > 20:
            print(f"... {len(rows) - 20} more", flush=True)
        return 0

    sbert, model, device = load_model()
    updated = 0
    for index, row in enumerate(rows, start=1):
        notice = {
            "title": row["title"],
            "url": row["url"],
            "posted_at": row["posted_at"].isoformat() if row["posted_at"] else None,
            "date": row["posted_date_text"] or (row["posted_at"].isoformat() if row["posted_at"] else ""),
            "category": row["category"],
            "body": row["body"],
            "views": row["views"],
        }
        notice_score = calc_notice_score(notice)
        embedding = calc_embedding(notice, sbert, model, device, notice_score)
        _update_embedding(row["id"], notice_score, embedding)
        updated += 1
        print(
            f"[{index}/{len(rows)}] updated id={row['id']} notice_id={row['notice_id']} "
            f"score={notice_score} dims={len(embedding)} body_len={len(row['body'])}",
            flush=True,
        )
    print(f"Done. updated={updated}", flush=True)
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute Supabase notice embeddings.")
    parser.add_argument("--start-date", required=True, type=_parse_date)
    parser.add_argument("--end-date", required=True, type=_parse_date)
    parser.add_argument("--ids", nargs="+", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.end_date < args.start_date:
        raise ValueError("--end-date must be on or after --start-date")
    reindex(args.start_date, args.end_date, args.ids, args.dry_run)


if __name__ == "__main__":
    main()
