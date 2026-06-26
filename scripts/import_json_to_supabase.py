#!/usr/bin/env python3
"""Import existing notice JSON files into Supabase/Postgres."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crawling.supabase_store import ensure_schema, upsert_notices  # noqa: E402


def load_notices(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")

    notices = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{index} is not a JSON object")
        if not item.get("url") or not item.get("title"):
            raise ValueError(f"{path}:{index} must include title and url")
        notices.append(item)
    return notices


def import_file(path: Path, batch_size: int) -> int:
    notices = load_notices(path)
    total = 0

    for start in range(0, len(notices), batch_size):
        batch = notices[start:start + batch_size]
        total += upsert_notices(batch)
        print(f"{path}: {min(start + batch_size, len(notices))}/{len(notices)} upserted")

    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import notice JSON files into the Supabase notices table.",
    )
    parser.add_argument("paths", nargs="+", type=Path, help="JSON files to import")
    parser.add_argument("--batch-size", type=int, default=100, help="Rows per DB batch")
    parser.add_argument("--skip-schema", action="store_true", help="Do not create/check DB schema")
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    if not args.skip_schema:
        ensure_schema()

    total = 0
    for path in args.paths:
        total += import_file(path, args.batch_size)

    print(f"Done. Upserted {total} rows from {len(args.paths)} file(s).")


if __name__ == "__main__":
    main()
