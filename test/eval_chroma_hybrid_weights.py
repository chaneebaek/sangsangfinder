"""
Evaluate production-like ChromaDB hybrid search across keyword/vector weights.

This script intentionally uses api.services.search_service.hybrid_search instead
of rebuilding dense scores in memory, so the evaluation follows the same Chroma
candidate retrieval, score fusion, URL de-duplication, and metadata path used by
the app.

Metrics:
  - Recall@K
  - MRR
  - NDCG@K

Ground truth:
  - qa_test_2025.jsonl points to notices in test_notices_2025.json.
  - Matching is done by notice URL, not title, because Chroma stores chunked
    documents and titles can be duplicated or normalized differently.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry.product").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry.posthog").setLevel(logging.CRITICAL)

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["VECTOR_DB"] = "chroma"

from api.core.models import get_chroma  # noqa: E402
from api.services.search_service import hybrid_search  # noqa: E402


QA_DATA_DIR = ROOT / "qa_dataset_generation" / "data"
CORPUS_PATH = QA_DATA_DIR / "test_notices_2025.json"
QA_PATH = QA_DATA_DIR / "qa_test_2025.jsonl"


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def recall_at_k(ranked_urls: list[str], gt_url: str, k: int) -> float:
    return 1.0 if gt_url in ranked_urls[:k] else 0.0


def mrr_score(ranked_urls: list[str], gt_url: str) -> float:
    for idx, url in enumerate(ranked_urls):
        if url == gt_url:
            return 1.0 / (idx + 1)
    return 0.0


def ndcg_at_k(ranked_urls: list[str], gt_url: str, k: int) -> float:
    for idx, url in enumerate(ranked_urls[:k]):
        if url == gt_url:
            return 1.0 / math.log2(idx + 2)
    return 0.0


def compute_scores(rows: list[dict], k: int) -> dict[str, float | int]:
    n = len(rows)
    if n == 0:
        return {f"Recall@{k}": 0.0, "MRR": 0.0, f"NDCG@{k}": 0.0, "n": 0}

    recall = sum(recall_at_k(r["ranked_urls"], r["gt_url"], k) for r in rows) / n
    mrr = sum(mrr_score(r["ranked_urls"], r["gt_url"]) for r in rows) / n
    ndcg = sum(ndcg_at_k(r["ranked_urls"], r["gt_url"], k) for r in rows) / n
    return {
        f"Recall@{k}": round(recall, 4),
        "MRR": round(mrr, 4),
        f"NDCG@{k}": round(ndcg, 4),
        "n": n,
    }


def format_ratio(alpha: float) -> str:
    keyword = 1.0 - alpha
    vector = alpha
    return f"{keyword:.1f}:{vector:.1f}"


def alpha_values(step: float, include_extremes: bool) -> list[float]:
    start = 0 if include_extremes else 1
    end = int(round(1.0 / step))
    stop = end + 1 if include_extremes else end
    return [round(i * step, 10) for i in range(start, stop)]


def build_eval_examples(corpus: list[dict], qa_list: list[dict]) -> tuple[list[dict], int]:
    title_to_notice = {notice["title"]: notice for notice in corpus}
    examples = []
    skipped = 0

    for qa in qa_list:
        notice = None
        notice_id = qa.get("notice_id")
        if isinstance(notice_id, int) and 0 <= notice_id < len(corpus):
            candidate = corpus[notice_id]
            if candidate.get("title") == qa.get("notice_title"):
                notice = candidate

        if notice is None:
            notice = title_to_notice.get(qa.get("notice_title"))

        if notice is None or not notice.get("url"):
            skipped += 1
            continue

        examples.append(
            {
                "question": qa["question"],
                "type": qa.get("type", "unknown"),
                "notice_title": qa.get("notice_title", notice.get("title", "")),
                "gt_url": notice["url"],
                "gt_date": notice.get("date", ""),
            }
        )

    return examples, skipped


def check_ground_truth_coverage(gt_urls: Iterable[str]) -> tuple[int, int]:
    collection = get_chroma()
    unique_urls = sorted(set(gt_urls))
    present = 0
    for url in unique_urls:
        found = collection.get(where={"url": url}, limit=1)
        if found.get("ids"):
            present += 1
    return present, len(unique_urls)


def evaluate_alpha(examples: list[dict], alpha: float, k: int) -> list[dict]:
    rows = []
    for ex in examples:
        results = hybrid_search(ex["question"], top_k=k, alpha=alpha)
        ranked_urls = [r["url"] for r in results]
        top = results[0] if results else {}
        rows.append(
            {
                **ex,
                "ranked_urls": ranked_urls,
                "top_title": top.get("title", ""),
                "top_url": top.get("url", ""),
                "top_date": top.get("date", ""),
                "top_score": top.get("score"),
                "hit": ex["gt_url"] in ranked_urls[:k],
            }
        )
    return rows


def print_score_table(results_by_alpha: list[tuple[float, dict]], k: int) -> None:
    print("\n" + "=" * 78)
    print(f"Overall metrics by weight (K={k})")
    print("=" * 78)
    print(
        f"{'alpha(vector)':>13} {'keyword:vector':>16} "
        f"{f'Recall@{k}':>10} {'MRR':>10} {f'NDCG@{k}':>10} {'n':>6}"
    )
    print("-" * 78)
    for alpha, scores in results_by_alpha:
        print(
            f"{alpha:>13.1f} {format_ratio(alpha):>16} "
            f"{scores[f'Recall@{k}']:>10.4f} {scores['MRR']:>10.4f} "
            f"{scores[f'NDCG@{k}']:>10.4f} {scores['n']:>6}"
        )


def print_type_breakdown(rows: list[dict], alpha: float, k: int) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["type"]].append(row)

    print("\n" + "=" * 78)
    print(f"Type breakdown for best alpha={alpha:.1f} (keyword:vector={format_ratio(alpha)})")
    print("=" * 78)
    print(f"{'type':<18} {f'Recall@{k}':>10} {'MRR':>10} {f'NDCG@{k}':>10} {'n':>6}")
    print("-" * 78)
    for qtype, group_rows in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        scores = compute_scores(group_rows, k)
        print(
            f"{qtype:<18} {scores[f'Recall@{k}']:>10.4f} {scores['MRR']:>10.4f} "
            f"{scores[f'NDCG@{k}']:>10.4f} {scores['n']:>6}"
        )


def print_miss_samples(rows: list[dict], limit: int) -> None:
    misses = [row for row in rows if not row["hit"]]
    if not misses:
        print("\nNo misses for the selected alpha.")
        return

    top_years = Counter((row.get("top_date") or "unknown")[:4] for row in misses)
    print("\n" + "=" * 78)
    print(f"Miss samples ({min(limit, len(misses))}/{len(misses)})")
    print("=" * 78)
    print("Top-1 year distribution among misses:", dict(sorted(top_years.items())))
    print("-" * 78)
    for idx, row in enumerate(misses[:limit], 1):
        print(f"{idx}. [{row['type']}] {row['question']}")
        print(f"   GT : {row['gt_date']} | {row['notice_title']}")
        print(f"   Top: {row.get('top_date', '')} | {row.get('top_title', '')} | score={row.get('top_score')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="Top-k for retrieval metrics.")
    parser.add_argument("--step", type=float, default=0.1, help="Alpha grid step.")
    parser.add_argument(
        "--no-extremes",
        action="store_true",
        help="Evaluate only 0.1..0.9 instead of including 0.0 and 1.0.",
    )
    parser.add_argument("--miss-limit", type=int, default=15, help="Number of miss samples to print.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N examples.")
    parser.add_argument(
        "--allow-missing-gt",
        action="store_true",
        help="Continue even if some ground-truth URLs are missing from ChromaDB.",
    )
    args = parser.parse_args()

    corpus = load_json(CORPUS_PATH)
    qa_list = load_jsonl(QA_PATH)
    examples, skipped = build_eval_examples(corpus, qa_list)
    if args.limit is not None:
        examples = examples[: args.limit]

    print("=" * 78)
    print("Production-like Chroma hybrid weight evaluation")
    print("=" * 78)
    print(f"QA file       : {QA_PATH}")
    print(f"Corpus file   : {CORPUS_PATH}")
    print(f"QA examples   : {len(examples)} (skipped={skipped})")
    print(f"Unique GT URLs : {len(set(ex['gt_url'] for ex in examples))}")
    print(f"Chroma chunks  : {get_chroma().count():,}")

    present, total = check_ground_truth_coverage(ex["gt_url"] for ex in examples)
    print(f"GT URL coverage in Chroma: {present}/{total}")
    if present < total:
        print("[WARN] Some ground-truth URLs are not present in ChromaDB; metrics will be deflated.")
        if not args.allow_missing_gt:
            print("Abort: re-index the target corpus into ChromaDB or pass --allow-missing-gt to continue anyway.")
            return

    all_results = []
    rows_by_alpha = {}
    for alpha in alpha_values(args.step, include_extremes=not args.no_extremes):
        print(f"\nEvaluating alpha={alpha:.1f} (keyword:vector={format_ratio(alpha)})...", flush=True)
        rows = evaluate_alpha(examples, alpha, args.k)
        scores = compute_scores(rows, args.k)
        all_results.append((alpha, scores))
        rows_by_alpha[alpha] = rows

    print_score_table(all_results, args.k)

    best_alpha, best_scores = max(
        all_results,
        key=lambda item: (item[1][f"NDCG@{args.k}"], item[1]["MRR"], item[1][f"Recall@{args.k}"]),
    )
    print("\nBest by NDCG/MRR/Recall tie-break:")
    print(
        f"  alpha(vector)={best_alpha:.1f}, keyword:vector={format_ratio(best_alpha)}, "
        f"Recall@{args.k}={best_scores[f'Recall@{args.k}']:.4f}, "
        f"MRR={best_scores['MRR']:.4f}, NDCG@{args.k}={best_scores[f'NDCG@{args.k}']:.4f}"
    )

    best_rows = rows_by_alpha[best_alpha]
    print_type_breakdown(best_rows, best_alpha, args.k)
    print_miss_samples(best_rows, args.miss_limit)


if __name__ == "__main__":
    main()
