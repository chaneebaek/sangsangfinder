"""
Evaluate reranker stages on a fixed hybrid candidate pool.

Compares the same Hybrid top-N candidates under four ranking strategies:
  - Hybrid
  - Hybrid + Feature
  - Hybrid + Cross-Encoder
  - Hybrid + Cross-Encoder + Feature

Metrics:
  - Recall@K
  - MRR
  - NDCG@K

Ground truth:
  - qa_test_2025.jsonl points to notices in test_notices_2025.json.
  - Matching is done by notice URL because Pinecone stores chunked documents.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("VECTOR_DB", "pinecone")

from api.core.models import get_vector_collection  # noqa: E402
from api.services.feature_reranker import rerank_notices  # noqa: E402
from api.services.search_service import hybrid_search  # noqa: E402


QA_DATA_DIR = ROOT / "qa_dataset_generation" / "data"
CORPUS_PATH = QA_DATA_DIR / "test_notices_2025.json"
QA_PATH = QA_DATA_DIR / "qa_test_2025.jsonl"
DEFAULT_CE_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_MIN_CE_FREE_GB = 3.0


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def compute_scores(rows: list[dict[str, Any]], k: int) -> dict[str, float | int]:
    n = len(rows)
    if n == 0:
        return {f"Recall@{k}": 0.0, "MRR": 0.0, f"NDCG@{k}": 0.0, "n": 0}

    recall = sum(recall_at_k(row["ranked_urls"], row["gt_url"], k) for row in rows) / n
    mrr = sum(mrr_score(row["ranked_urls"], row["gt_url"]) for row in rows) / n
    ndcg = sum(ndcg_at_k(row["ranked_urls"], row["gt_url"], k) for row in rows) / n
    return {
        f"Recall@{k}": round(recall, 4),
        "MRR": round(mrr, 4),
        f"NDCG@{k}": round(ndcg, 4),
        "n": n,
    }


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        return [1.0 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def build_eval_examples(corpus: list[dict[str, Any]], qa_list: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
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
    collection = get_vector_collection()
    unique_urls = sorted(set(gt_urls))
    present = 0
    for url in unique_urls:
        found = collection.get(where={"url": url}, limit=1)
        if found.get("ids"):
            present += 1
    return present, len(unique_urls)


def ce_text(notice: dict[str, Any], max_chars: int) -> str:
    parts = [
        f"제목: {notice.get('title', '')}",
        f"분류: {notice.get('category', '')}",
        f"날짜: {notice.get('date') or notice.get('posted_at') or ''}",
        str(notice.get("content") or notice.get("body") or ""),
    ]
    return "\n".join(part for part in parts if part.strip())[:max_chars]


def bytes_to_gib(size: int) -> float:
    return size / (1024**3)


def find_hf_cache_dir() -> Path:
    try:
        from huggingface_hub import constants

        return Path(constants.HF_HUB_CACHE)
    except Exception:
        return Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def is_model_cached(model_name: str) -> bool:
    try:
        from huggingface_hub import snapshot_download

        snapshot_dir = Path(snapshot_download(model_name, local_files_only=True))
    except Exception:
        return False

    weight_patterns = (
        "model.safetensors",
        "model-*.safetensors",
        "pytorch_model.bin",
        "pytorch_model-*.bin",
        "tf_model.h5",
        "model.ckpt*",
        "flax_model.msgpack",
    )
    return any(path.is_file() and path.stat().st_size > 1024 * 1024 for pattern in weight_patterns for path in snapshot_dir.glob(pattern))


def load_cross_encoder(model_name: str, min_free_gb: float):
    from sentence_transformers import CrossEncoder

    local_only = os.getenv("TRANSFORMERS_OFFLINE") == "1" or os.getenv("HF_HUB_OFFLINE") == "1"
    if not local_only and not is_model_cached(model_name):
        hf_cache_dir = find_hf_cache_dir()
        hf_cache_dir.mkdir(parents=True, exist_ok=True)
        free_gb = bytes_to_gib(shutil.disk_usage(hf_cache_dir).free)
        if free_gb < min_free_gb:
            raise RuntimeError(
                f"Not enough free disk space to download '{model_name}'. "
                f"Hugging Face cache '{hf_cache_dir}' has {free_gb:.2f} GiB free; "
                f"need at least {min_free_gb:.2f} GiB. Free disk space, choose a smaller cached "
                "model, or pass --skip-ce."
            )

    return CrossEncoder(model_name, max_length=512, trust_remote_code=True, local_files_only=local_only)


def rank_with_cross_encoder(
    ce_model,
    query: str,
    candidates: list[dict[str, Any]],
    batch_size: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    pairs = [(query, ce_text(candidate, max_chars)) for candidate in candidates]
    raw_scores = [float(score) for score in ce_model.predict(pairs, batch_size=batch_size, show_progress_bar=False)]
    norm_scores = minmax(raw_scores)

    reranked = []
    for original_rank, (candidate, raw_score, norm_score) in enumerate(
        zip(candidates, raw_scores, norm_scores),
        start=1,
    ):
        item = dict(candidate)
        item["hybrid_score"] = candidate.get("score", 0.0)
        item["score"] = round(norm_score, 6)
        item["cross_encoder"] = {
            "model_score": round(raw_score, 6),
            "normalized_score": round(norm_score, 6),
            "original_rank": original_rank,
        }
        reranked.append(item)

    reranked.sort(key=lambda item: (item["score"], -item["cross_encoder"]["original_rank"]), reverse=True)
    for rank, item in enumerate(reranked, start=1):
        item["cross_encoder"]["rerank_rank"] = rank
    return reranked


def urls(rows: list[dict[str, Any]]) -> list[str]:
    return [row["url"] for row in rows if row.get("url")]


def evaluate_examples(
    examples: list[dict[str, Any]],
    *,
    alpha: float,
    candidate_k: int,
    k: int,
    profile: dict[str, Any],
    ce_model,
    ce_batch_size: int,
    ce_max_chars: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, float]]:
    rows_by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    latency: dict[str, float] = defaultdict(float)

    for idx, ex in enumerate(examples, start=1):
        query = ex["question"]
        print(f"[{idx}/{len(examples)}] {query}", flush=True)

        started = time.perf_counter()
        candidates = hybrid_search(
            query,
            top_k=candidate_k,
            candidate_k=candidate_k,
            alpha=alpha,
            feature_rerank=False,
        )
        latency["Hybrid"] += time.perf_counter() - started

        started = time.perf_counter()
        feature_ranked = rerank_notices(candidates, profile=profile, top_k=candidate_k)
        latency["Hybrid+Feature"] += time.perf_counter() - started

        system_rankings = {
            "Hybrid": candidates,
            "Hybrid+Feature": feature_ranked,
        }

        if ce_model is not None:
            started = time.perf_counter()
            ce_ranked = rank_with_cross_encoder(ce_model, query, candidates, ce_batch_size, ce_max_chars)
            latency["Hybrid+CE"] += time.perf_counter() - started

            started = time.perf_counter()
            ce_feature_ranked = rerank_notices(ce_ranked, profile=profile, top_k=candidate_k)
            latency["Hybrid+CE+Feature"] += time.perf_counter() - started

            system_rankings["Hybrid+CE"] = ce_ranked
            system_rankings["Hybrid+CE+Feature"] = ce_feature_ranked

        for system_name, ranked in system_rankings.items():
            top = ranked[0] if ranked else {}
            rows_by_system[system_name].append(
                {
                    **ex,
                    "ranked_urls": urls(ranked),
                    "top_title": top.get("title", ""),
                    "top_url": top.get("url", ""),
                    "top_date": top.get("date", ""),
                    "top_score": top.get("score"),
                    "candidate_hit": ex["gt_url"] in urls(candidates),
                    "hit": ex["gt_url"] in urls(ranked)[:k],
                }
            )

    return rows_by_system, latency


def print_score_table(rows_by_system: dict[str, list[dict[str, Any]]], latency: dict[str, float], k: int) -> None:
    print("\n" + "=" * 92)
    print(f"Fixed-candidate reranker comparison (K={k})")
    print("=" * 92)
    print(f"{'system':<24} {f'Recall@{k}':>10} {'MRR':>10} {f'NDCG@{k}':>10} {'n':>6} {'sec/query':>10}")
    print("-" * 92)
    for system_name in ["Hybrid", "Hybrid+Feature", "Hybrid+CE", "Hybrid+CE+Feature"]:
        rows = rows_by_system.get(system_name)
        if not rows:
            continue
        scores = compute_scores(rows, k)
        sec_per_query = latency.get(system_name, 0.0) / max(1, len(rows))
        print(
            f"{system_name:<24} {scores[f'Recall@{k}']:>10.4f} {scores['MRR']:>10.4f} "
            f"{scores[f'NDCG@{k}']:>10.4f} {scores['n']:>6} {sec_per_query:>10.3f}"
        )


def print_type_breakdown(rows_by_system: dict[str, list[dict[str, Any]]], k: int) -> None:
    print("\n" + "=" * 92)
    print(f"Type breakdown by system (NDCG@{k})")
    print("=" * 92)

    all_types = sorted({row["type"] for rows in rows_by_system.values() for row in rows})
    systems = [name for name in ["Hybrid", "Hybrid+Feature", "Hybrid+CE", "Hybrid+CE+Feature"] if name in rows_by_system]
    print(f"{'type':<18} " + " ".join(f"{system:>18}" for system in systems))
    print("-" * 92)
    for qtype in all_types:
        cells = []
        for system in systems:
            group = [row for row in rows_by_system[system] if row["type"] == qtype]
            scores = compute_scores(group, k)
            cells.append(f"{scores[f'NDCG@{k}']:>18.4f}")
        print(f"{qtype:<18} " + " ".join(cells))


def print_miss_samples(rows_by_system: dict[str, list[dict[str, Any]]], k: int, limit: int) -> None:
    base_rows = rows_by_system.get("Hybrid", [])
    if not base_rows:
        return

    candidate_misses = [row for row in base_rows if not row["candidate_hit"]]
    if candidate_misses:
        top_years = Counter((row.get("top_date") or "unknown")[:4] for row in candidate_misses)
        print("\n" + "=" * 92)
        print(f"Candidate-pool misses ({min(limit, len(candidate_misses))}/{len(candidate_misses)})")
        print("=" * 92)
        print("Top-1 year distribution:", dict(sorted(top_years.items())))
        for idx, row in enumerate(candidate_misses[:limit], start=1):
            print(f"{idx}. [{row['type']}] {row['question']}")
            print(f"   GT : {row['gt_date']} | {row['notice_title']}")
            print(f"   Top: {row.get('top_date', '')} | {row.get('top_title', '')}")

    print("\n" + "=" * 92)
    print(f"Top-{k} miss counts")
    print("=" * 92)
    for system_name, rows in rows_by_system.items():
        misses = [row for row in rows if not row["hit"]]
        print(f"{system_name:<24} {len(misses):>5}/{len(rows)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="Top-k for retrieval metrics.")
    parser.add_argument("--candidate-k", type=int, default=100, help="Fixed hybrid candidate pool size.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Vector weight for hybrid retrieval.")
    parser.add_argument("--ce-model", default=DEFAULT_CE_MODEL, help="Cross-encoder reranker model.")
    parser.add_argument("--ce-batch-size", type=int, default=8, help="Cross-encoder prediction batch size.")
    parser.add_argument("--ce-max-chars", type=int, default=1800, help="Max document characters sent to CE.")
    parser.add_argument(
        "--ce-min-free-gb",
        type=float,
        default=DEFAULT_MIN_CE_FREE_GB,
        help="Minimum free GiB required before downloading an uncached cross-encoder model.",
    )
    parser.add_argument(
        "--ce-on-fail",
        choices=["skip", "abort"],
        default="skip",
        help="What to do when the cross-encoder cannot be loaded.",
    )
    parser.add_argument("--skip-ce", action="store_true", help="Evaluate only Hybrid and Hybrid+Feature.")
    parser.add_argument("--profile-json", default="{}", help="User profile JSON for feature reranker.")
    parser.add_argument("--miss-limit", type=int, default=10, help="Number of miss samples to print.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N examples.")
    parser.add_argument(
        "--allow-missing-gt",
        action="store_true",
        help="Continue even if some ground-truth URLs are missing from Pinecone.",
    )
    args = parser.parse_args()

    if args.candidate_k < args.k:
        raise SystemExit("--candidate-k must be greater than or equal to --k")
    if not 0.0 <= args.alpha <= 1.0:
        raise SystemExit("--alpha must be between 0.0 and 1.0")

    try:
        profile = json.loads(args.profile_json)
        if not isinstance(profile, dict):
            raise ValueError
    except ValueError as exc:
        raise SystemExit("--profile-json must be a JSON object") from exc

    corpus = load_json(CORPUS_PATH)
    qa_list = load_jsonl(QA_PATH)
    examples, skipped = build_eval_examples(corpus, qa_list)
    if args.limit is not None:
        examples = examples[: args.limit]

    print("=" * 92)
    print("Fixed-candidate reranker-stage evaluation")
    print("=" * 92)
    print(f"QA file        : {QA_PATH}")
    print(f"Corpus file    : {CORPUS_PATH}")
    print(f"QA examples    : {len(examples)} (skipped={skipped})")
    print(f"Unique GT URLs : {len(set(ex['gt_url'] for ex in examples))}")
    print(f"Hybrid alpha   : {args.alpha:.2f}")
    print(f"Candidate K    : {args.candidate_k}")
    print(f"Metric K       : {args.k}")
    print(f"Pinecone count : {get_vector_collection().count():,}")

    present, total = check_ground_truth_coverage(ex["gt_url"] for ex in examples)
    print(f"GT URL coverage: {present}/{total}")
    if present < total:
        print("[WARN] Some ground-truth URLs are not present in Pinecone; metrics will be deflated.")
        if not args.allow_missing_gt:
            print("Abort: re-index the target corpus into Pinecone or pass --allow-missing-gt to continue.")
            return

    ce_model = None
    if not args.skip_ce:
        print(f"Cross-encoder  : {args.ce_model}", flush=True)
        try:
            ce_model = load_cross_encoder(args.ce_model, args.ce_min_free_gb)
        except Exception as exc:
            message = f"[WARN] Cross-encoder load failed: {exc}"
            if args.ce_on_fail == "abort":
                raise SystemExit(message) from exc
            print(message, flush=True)
            print("[WARN] Continuing without CE systems. Use --skip-ce to silence this warning.", flush=True)

    rows_by_system, latency = evaluate_examples(
        examples,
        alpha=args.alpha,
        candidate_k=args.candidate_k,
        k=args.k,
        profile=profile,
        ce_model=ce_model,
        ce_batch_size=args.ce_batch_size,
        ce_max_chars=args.ce_max_chars,
    )

    print_score_table(rows_by_system, latency, args.k)
    print_type_breakdown(rows_by_system, args.k)
    print_miss_samples(rows_by_system, args.k, args.miss_limit)


if __name__ == "__main__":
    main()
