"""
Index notice embeddings into the configured vector DB.

Run from the project root:
    python scripts/index_notices.py
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from api.core.models import index_notices, load_notices_cache  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Index Sangsangfinder notices into the configured vector DB.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed all notices even when the manifest hash is unchanged.",
    )
    parser.add_argument(
        "--notice-batch-size",
        type=int,
        default=20,
        help="Number of notices to encode, save, and checkpoint at a time.",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=16,
        help="Number of chunks to pass through the embedding model at a time.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    notices = load_notices_cache()
    if not notices:
        raise SystemExit("No notices found. Check NOTICES_CACHE_PATH.")

    indexed = index_notices(
        notices,
        force=args.force,
        sync_deletions=True,
        notice_batch_size=args.notice_batch_size,
        embed_batch_size=args.embed_batch_size,
    )
    print(f"Indexed {indexed} notices.")


if __name__ == "__main__":
    main()
