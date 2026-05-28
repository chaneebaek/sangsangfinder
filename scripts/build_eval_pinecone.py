"""
Build or refresh a Pinecone namespace for retrieval evaluation.

Default behavior:
  - index qa_dataset_generation/data/test_notices_2025.json

For full 2025/2026 evaluation, pass --full. Use --force to re-embed notices even
when their manifest entries look unchanged.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path


ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CORPORA = [ROOT / "qa_dataset_generation" / "data" / "test_notices_2025.json"]
FULL_EVAL_CORPORA = [
    ROOT / "qa_dataset_generation" / "data" / "test_notices_2025.json",
    ROOT / "data" / "data_2026.json",
]
DEFAULT_EVAL_NAMESPACE = "eval"
DEFAULT_EVAL_STATE = ROOT / "pinecone_index" / "eval"


def load_notices(paths: list[Path]) -> list[dict]:
    seen_urls = set()
    notices = []
    for path in paths:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            url = item.get("url")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            notices.append(
                {
                    "title": item.get("title", ""),
                    "url": url,
                    "date": item.get("date", ""),
                    "body": item.get("body", ""),
                    "category": item.get("category", ""),
                }
            )
    return notices


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Pinecone evaluation namespace.")
    parser.add_argument(
        "--corpus",
        type=Path,
        action="append",
        default=None,
        help="Corpus JSON file. Can be passed multiple times.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Index both the 2025 QA corpus and current 2026 notices.",
    )
    parser.add_argument(
        "--namespace",
        default=DEFAULT_EVAL_NAMESPACE,
        help="Pinecone namespace to index.",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=DEFAULT_EVAL_STATE,
        help="Local manifest/cache directory for the evaluation namespace.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed notices even when manifest entries look unchanged.",
    )
    parser.add_argument(
        "--sync-deletions",
        action="store_true",
        help="Delete vectors for notices no longer present in the selected corpora.",
    )
    parser.add_argument(
        "--pooling",
        choices=["cls", "mean"],
        default="cls",
        help="Pooling mode used by SimCSEEmbedder.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Embedding model name/path. Defaults to api.core.config.BASE_MODEL_EMBED.",
    )
    parser.add_argument(
        "--backend",
        choices=["simcse", "sentence-transformers"],
        default="simcse",
        help="Embedding backend.",
    )
    parser.add_argument("--notice-batch-size", type=int, default=20)
    parser.add_argument("--embed-batch-size", type=int, default=16)
    args = parser.parse_args()

    default_corpora = FULL_EVAL_CORPORA if args.full else DEFAULT_CORPORA
    corpora = [path.resolve() for path in (args.corpus or default_corpora)]
    state_dir = args.state_dir.resolve()

    os.environ["VECTOR_DB"] = "pinecone"
    os.environ["PINECONE_NAMESPACE"] = args.namespace
    os.environ["VECTOR_STATE_PATH"] = str(state_dir)
    os.environ["PINECONE_CACHE_PATH"] = str(state_dir / "pinecone_chunks.json")
    os.environ["SIMCSE_POOLING"] = args.pooling
    os.environ["EMBEDDER_BACKEND"] = args.backend
    if args.model:
        os.environ["BASE_MODEL_EMBED"] = args.model

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from api.core.models import get_vector_collection, index_notices

    notices = load_notices(corpora)
    if not notices:
        raise SystemExit("No notices loaded.")

    print("=" * 72)
    print("Building Pinecone evaluation namespace")
    print("=" * 72)
    print(f"Namespace : {args.namespace}")
    print(f"State dir : {state_dir}")
    print(f"Backend   : {args.backend}")
    print(f"Model     : {args.model or '(config default)'}")
    print(f"Pooling   : {args.pooling}")
    print("Corpora:")
    for corpus in corpora:
        print(f"  - {corpus}")
    print(f"Notices   : {len(notices)} unique URLs")

    indexed = index_notices(
        notices,
        force=args.force,
        sync_deletions=args.sync_deletions,
        notice_batch_size=args.notice_batch_size,
        embed_batch_size=args.embed_batch_size,
    )
    print(f"Indexed notices : {indexed}")
    print(f"Pinecone vectors: {get_vector_collection().count():,}")


if __name__ == "__main__":
    main()
