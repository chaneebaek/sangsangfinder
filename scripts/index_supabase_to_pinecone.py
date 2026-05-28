"""
Backfill Supabase notices into Pinecone embeddings.

Run from the project root:
    VECTOR_DB=pinecone python scripts/index_supabase_to_pinecone.py
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("VECTOR_DB", "pinecone")

from crawling.supabase_store import SOURCE, _connect  # noqa: E402
from api.core.models import _notice_doc_id, get_vector_collection, index_notices  # noqa: E402


def _load_supabase_notices(
    year: str | None = None,
    limit: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict[str, Any]]:
    params: list[Any] = [SOURCE]
    filters = ["source = %s"]

    if year:
        filters.append(
            """
            (
                extract(year from posted_at)::text = %s
                or posted_date_text like %s
            )
            """
        )
        params.extend([year, f"{year}%"])

    if start_date:
        filters.append("posted_at >= %s")
        params.append(datetime.strptime(start_date, "%Y-%m-%d").date())
    if end_date:
        filters.append("posted_at <= %s")
        params.append(datetime.strptime(end_date, "%Y-%m-%d").date())

    sql = f"""
        select title, url, posted_at, posted_date_text, category, body, views
        from notices
        where {' and '.join(filters)}
        order by posted_at desc nulls last, id desc
    """
    if limit:
        sql += " limit %s"
        params.append(limit)

    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    notices = []
    for title, url, posted_at, posted_date_text, category, body, views in rows:
        if not url:
            continue
        date_text = posted_date_text or (posted_at.isoformat() if posted_at else "")
        notices.append(
            {
                "title": title or "",
                "url": url,
                "date": date_text,
                "category": category,
                "body": body or "",
                "views": views,
            }
        )
    return notices


def _filter_missing_from_vector_db(notices: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chunk0_ids = [f"{_notice_doc_id(item['url'])}_0" for item in notices]
    existing_ids = set(get_vector_collection().get(ids=chunk0_ids)["ids"]) if chunk0_ids else set()
    return [
        item
        for item, chunk0_id in zip(notices, chunk0_ids)
        if chunk0_id not in existing_ids
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index Supabase notices that are missing from Pinecone."
    )
    parser.add_argument("--year", help="Only index notices from this year, e.g. 2026.")
    parser.add_argument("--start-date", help="Only index notices on or after this date (YYYY-MM-DD).")
    parser.add_argument("--end-date", help="Only index notices on or before this date (YYYY-MM-DD).")
    parser.add_argument("--limit", type=int, help="Limit the number of Supabase rows to scan.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Index all loaded notices through the normal manifest/content-hash path.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and re-index all loaded notices even when the manifest hash is unchanged.",
    )
    parser.add_argument("--notice-batch-size", type=int, default=20)
    parser.add_argument("--embed-batch-size", type=int, default=16)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    notices = _load_supabase_notices(
        year=args.year,
        limit=args.limit,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    if not notices:
        raise SystemExit("No Supabase notices found.")

    target_notices = notices if args.all else _filter_missing_from_vector_db(notices)
    logging.info(
        "Supabase notices loaded: %d, Pinecone missing: %d",
        len(notices),
        len(target_notices),
    )

    if not target_notices:
        print("No missing Pinecone vectors found.")
        return

    indexed = index_notices(
        target_notices,
        force=args.force,
        notice_batch_size=args.notice_batch_size,
        embed_batch_size=args.embed_batch_size,
    )
    print(f"Indexed {indexed} Supabase notices into Pinecone.")


if __name__ == "__main__":
    main()
