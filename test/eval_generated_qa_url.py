#!/usr/bin/env python3
"""Evaluate generated QA JSON using URL ground truth against Pinecone search."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

os.environ["VECTOR_DB"] = "pinecone"

from api.core.models import get_vector_collection  # noqa: E402
from api.services.search_service import hybrid_search  # noqa: E402


DEFAULT_QA_PATH = ROOT / "data" / "qa_supabase_2026_smoke.json"


def load_qa_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("qas"), list):
        rows = payload["qas"]
    else:
        raise ValueError("QA file must be a JSON list or an object with a 'qas' list.")

    valid = []
    skipped = []
    for idx, row in enumerate(rows):
        question = str(row.get("question") or "").strip()
        gt_url = str(row.get("gt_url") or row.get("notice_url") or "").strip()
        if not question or not gt_url:
            skipped.append((idx, "missing question or gt_url"))
            continue
        valid.append(
            {
                **row,
                "question": question,
                "gt_url": gt_url,
                "type": row.get("type") or "unknown",
            }
        )

    if skipped:
        print(f"[WARN] skipped invalid rows: {len(skipped)}")
        for idx, reason in skipped[:5]:
            print(f"  - row {idx}: {reason}")
    return valid


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


def compute_scores(rows: list[dict[str, Any]], k: int) -> dict[str, float | int]:
    n = len(rows)
    if n == 0:
        return {f"Recall@{k}": 0.0, "MRR": 0.0, f"NDCG@{k}": 0.0, "n": 0}
    return {
        f"Recall@{k}": round(sum(recall_at_k(r["ranked_urls"], r["gt_url"], k) for r in rows) / n, 4),
        "MRR": round(sum(mrr_score(r["ranked_urls"], r["gt_url"]) for r in rows) / n, 4),
        f"NDCG@{k}": round(sum(ndcg_at_k(r["ranked_urls"], r["gt_url"], k) for r in rows) / n, 4),
        "n": n,
    }


def check_ground_truth_coverage(gt_urls: Iterable[str]) -> tuple[int, int]:
    collection = get_vector_collection()
    unique_urls = sorted(set(gt_urls))
    present = 0
    missing = []
    for url in unique_urls:
        found = collection.get(where={"url": url}, limit=1)
        if found.get("ids"):
            present += 1
        else:
            missing.append(url)
    if missing:
        print(f"[WARN] missing GT URLs in Pinecone: {len(missing)}")
        for url in missing[:5]:
            print(f"  - {url}")
    return present, len(unique_urls)


def alpha_values(step: float) -> list[float]:
    if step <= 0 or step > 1:
        raise ValueError("--step must be in (0, 1].")
    count = int(round(1.0 / step))
    return [round(i * step, 10) for i in range(count + 1)]


def format_ratio(alpha: float) -> str:
    return f"{1 - alpha:.1f}:{alpha:.1f}"


def evaluate(rows: list[dict[str, Any]], alpha: float, k: int) -> list[dict[str, Any]]:
    evaluated = []
    for row in rows:
        results = hybrid_search(row["question"], top_k=k, alpha=alpha)
        ranked_urls = [result["url"] for result in results if result.get("url")]
        top = results[0] if results else {}
        evaluated.append(
            {
                **row,
                "ranked_urls": ranked_urls,
                "top_url": top.get("url", ""),
                "top_title": top.get("title", ""),
                "hit": row["gt_url"] in ranked_urls[:k],
            }
        )
    return evaluated


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated QA JSON by URL ground truth.")
    parser.add_argument("--qa", type=Path, default=DEFAULT_QA_PATH)
    parser.add_argument("--type", default=None, help="Optional QA type filter.")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--allow-missing-gt", action="store_true")
    args = parser.parse_args()

    rows = load_qa_rows(args.qa)
    if args.type:
        rows = [row for row in rows if row.get("type") == args.type]
    if not rows:
        raise SystemExit("No valid QA rows to evaluate.")

    present, total = check_ground_truth_coverage(row["gt_url"] for row in rows)
    print(f"QA file       : {args.qa}")
    print(f"QA rows       : {len(rows)}")
    print(f"Unique GT URLs : {total}")
    print(f"GT URL coverage in Pinecone: {present}/{total}")
    if present < total and not args.allow_missing_gt:
        raise SystemExit("Abort: pass --allow-missing-gt to evaluate with missing GT URLs.")

    print(f"{'alpha(vector)':>13} {'keyword:vector':>16} {f'Recall@{args.k}':>10} {'MRR':>10} {f'NDCG@{args.k}':>10} {'n':>6}")
    results_by_alpha = []
    rows_by_alpha = {}
    for alpha in alpha_values(args.step):
        evaluated = evaluate(rows, alpha, args.k)
        scores = compute_scores(evaluated, args.k)
        results_by_alpha.append((alpha, scores))
        rows_by_alpha[alpha] = evaluated
        print(
            f"{alpha:>13.1f} {format_ratio(alpha):>16} "
            f"{scores[f'Recall@{args.k}']:>10.4f} {scores['MRR']:>10.4f} "
            f"{scores[f'NDCG@{args.k}']:>10.4f} {scores['n']:>6}"
        )

    best_alpha, best_scores = max(
        results_by_alpha,
        key=lambda item: (item[1][f"NDCG@{args.k}"], item[1]["MRR"], item[1][f"Recall@{args.k}"]),
    )
    print(
        f"BEST alpha={best_alpha:.1f} keyword:vector={format_ratio(best_alpha)} "
        f"Recall@{args.k}={best_scores[f'Recall@{args.k}']:.4f} "
        f"MRR={best_scores['MRR']:.4f} NDCG@{args.k}={best_scores[f'NDCG@{args.k}']:.4f}"
    )

    misses = [row for row in rows_by_alpha[best_alpha] if not row["hit"]]
    if misses:
        print(f"Misses at best alpha: {len(misses)}")
        for row in misses[:5]:
            print(f"  - [{row.get('type')}] {row['question']} | GT={row['notice_title']} | Top={row['top_title']}")


if __name__ == "__main__":
    main()
