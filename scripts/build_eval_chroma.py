"""
Build an evaluation-only ChromaDB for production-like retrieval tests.

Default behavior:
  - copy the current production chroma_db, which already contains 2026 notices
  - add qa_dataset_generation/data/test_notices_2025.json

For pooling ablations, pass --pooling cls or --pooling mean. Use --no-seed
when the output DB must be fully re-embedded with that pooling mode.

The resulting DB is separate from the production DB by default:
  - chroma_db_eval_2025_2026
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT = ROOT / "chroma_db_eval_2025_2026"
DEFAULT_CORPORA = [ROOT / "qa_dataset_generation" / "data" / "test_notices_2025.json"]
FULL_EVAL_CORPORA = [
    ROOT / "qa_dataset_generation" / "data" / "test_notices_2025.json",
    ROOT / "data" / "data_2026.json",
]
PRODUCTION_CHROMA = ROOT / "chroma_db"


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
    parser = argparse.ArgumentParser(description="Build an evaluation-only ChromaDB.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output ChromaDB directory.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        action="append",
        default=None,
        help="Corpus JSON file. Can be passed multiple times.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the output DB before indexing.",
    )
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="Do not copy production chroma_db first; build only from --corpus files.",
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

    output = args.output.resolve()
    default_corpora = FULL_EVAL_CORPORA if args.no_seed else DEFAULT_CORPORA
    corpora = [p.resolve() for p in (args.corpus or default_corpora)]

    if output == (ROOT / "chroma_db").resolve():
        raise SystemExit("Refusing to overwrite the production chroma_db path.")
    if args.reset and output.exists():
        shutil.rmtree(output)
    if not args.no_seed and not output.exists():
        if not PRODUCTION_CHROMA.exists():
            raise SystemExit(f"Production chroma_db not found: {PRODUCTION_CHROMA}")
        shutil.copytree(PRODUCTION_CHROMA, output)

    os.environ["CHROMA_DB_PATH"] = str(output)
    os.environ["VECTOR_DB"] = "chroma"
    os.environ["SIMCSE_POOLING"] = args.pooling
    os.environ["EMBEDDER_BACKEND"] = args.backend
    if args.model:
        os.environ["BASE_MODEL_EMBED"] = args.model

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
    logging.getLogger("chromadb.telemetry.product").setLevel(logging.CRITICAL)
    logging.getLogger("chromadb.telemetry.posthog").setLevel(logging.CRITICAL)

    from api.core.models import get_chroma, index_notices

    notices = load_notices(corpora)
    if not notices:
        raise SystemExit("No notices loaded.")

    print("=" * 72)
    print("Building evaluation ChromaDB")
    print("=" * 72)
    print(f"Output : {output}")
    print(f"Seeded : {'no' if args.no_seed else PRODUCTION_CHROMA}")
    print(f"Backend: {args.backend}")
    print(f"Model  : {args.model or '(config default)'}")
    print(f"Pooling: {args.pooling}")
    print("Corpora:")
    for corpus in corpora:
        print(f"  - {corpus}")
    print(f"Notices: {len(notices)} unique URLs")

    indexed = index_notices(
        notices,
        force=True,
        sync_deletions=args.no_seed,
        notice_batch_size=args.notice_batch_size,
        embed_batch_size=args.embed_batch_size,
    )
    print(f"Indexed notices: {indexed}")
    print(f"Chroma chunks  : {get_chroma().count():,}")


if __name__ == "__main__":
    main()
