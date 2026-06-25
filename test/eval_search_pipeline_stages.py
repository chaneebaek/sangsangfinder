"""
Evaluate the app search pipeline stages with the existing URL-ground-truth metrics.

Compares:
  - Hybrid
  - Hybrid + Query Routing
  - Hybrid + Query Routing + Feature Reranker

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
import contextlib
import io
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("VECTOR_DB", "pinecone")
if "--disable-feature-llm" in sys.argv:
    os.environ["GEMINI_API_KEY_PAID_TIER"] = ""
if "--disable-routing-llm" in sys.argv:
    os.environ["GEMINI_API_KEY"] = ""

from api.core.models import get_vector_collection  # noqa: E402
from api.services import feature_reranker as feature_reranker_module  # noqa: E402
from api.services.feature_reranker import rerank_notices  # noqa: E402
from api.services.search_service import (  # noqa: E402
    DEFAULT_ALPHA,
    ROUTED_ALPHA,
    _apply_intent_boosts,
    _hybrid_candidate_search,
    _route_query_with_gemini,
    _routed_candidates_before_feature_rerank,
)


QA_DATA_DIR = ROOT / "qa_dataset_generation" / "data"
CORPUS_PATH = QA_DATA_DIR / "test_notices_2025.json"
QA_PATH = QA_DATA_DIR / "qa_test_2025.jsonl"


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


def rank_of(ranked_urls: list[str], gt_url: str) -> int | None:
    for idx, url in enumerate(ranked_urls):
        if url == gt_url:
            return idx + 1
    return None


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


def urls(rows: list[dict[str, Any]]) -> list[str]:
    return [row["url"] for row in rows if row.get("url")]


def quiet_call(fn, *args, quiet: bool, **kwargs):
    if not quiet:
        return fn(*args, **kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def evaluate_examples(
    examples: list[dict[str, Any]],
    *,
    alpha: float,
    candidate_k: int,
    k: int,
    profile: dict[str, Any],
    skip_feature_rerank: bool,
    quiet: bool,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, float], list[dict[str, Any]]]:
    rows_by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    latency: dict[str, float] = defaultdict(float)
    routing_diags: list[dict[str, Any]] = []

    for idx, ex in enumerate(examples, start=1):
        query = ex["question"]
        print(f"[{idx}/{len(examples)}] {query}", flush=True)

        started = time.perf_counter()
        hybrid_candidates, hybrid_diag = quiet_call(
            _hybrid_candidate_search,
            query=query,
            top_k=k,
            alpha=alpha,
            candidate_k=candidate_k,
            quiet=quiet,
        )
        latency["Hybrid"] += time.perf_counter() - started

        started = time.perf_counter()
        route = quiet_call(_route_query_with_gemini, query, quiet=quiet)
        search_query = route["search_query"] if route.get("is_user_short") else query
        routed_alpha = ROUTED_ALPHA.get(route["intent"], DEFAULT_ALPHA)
        latency["RouteClassify"] += time.perf_counter() - started

        started = time.perf_counter()
        rewrite_only, _ = quiet_call(
            _hybrid_candidate_search,
            query=search_query,
            top_k=k,
            alpha=alpha,
            candidate_k=candidate_k,
            quiet=quiet,
        )
        latency["RewriteOnly"] += time.perf_counter() - started

        started = time.perf_counter()
        alpha_only, _ = quiet_call(
            _hybrid_candidate_search,
            query=query,
            top_k=k,
            alpha=routed_alpha,
            candidate_k=candidate_k,
            quiet=quiet,
        )
        latency["AlphaOnly"] += time.perf_counter() - started

        started = time.perf_counter()
        intent_boost_only = _apply_intent_boosts(hybrid_candidates, route, hybrid_diag)
        latency["IntentBoostOnly"] += time.perf_counter() - started

        started = time.perf_counter()
        _, routed_candidates, routing_diag = quiet_call(
            _routed_candidates_before_feature_rerank,
            query=query,
            top_k=k,
            alpha=alpha,
            category_filter=None,
            candidate_k=candidate_k,
            profile=profile,
            quiet=quiet,
        )
        latency["Hybrid+Routing"] += time.perf_counter() - started
        routing_diags.append(routing_diag)

        system_rankings = {
            "Hybrid": hybrid_candidates,
            "RewriteOnly": rewrite_only,
            "AlphaOnly": alpha_only,
            "IntentBoostOnly": intent_boost_only,
            "Hybrid+Routing": routed_candidates,
        }
        if not skip_feature_rerank:
            started = time.perf_counter()
            feature_ranked = quiet_call(
                rerank_notices,
                routed_candidates,
                profile=profile,
                top_k=candidate_k,
                quiet=quiet,
            )
            latency["Hybrid+Routing+Feature"] += time.perf_counter() - started
            system_rankings["Hybrid+Routing+Feature"] = feature_ranked

        for system_name, ranked in system_rankings.items():
            top = ranked[0] if ranked else {}
            ranked_urls = urls(ranked)
            route = routing_diag.get("route", {})
            rows_by_system[system_name].append(
                {
                    **ex,
                    "ranked_urls": ranked_urls,
                    "top_title": top.get("title", ""),
                    "top_url": top.get("url", ""),
                    "top_date": top.get("date", ""),
                    "top_score": top.get("score"),
                    "candidate_hit": ex["gt_url"] in urls(hybrid_candidates),
                    "hit": ex["gt_url"] in ranked_urls[:k],
                    "gt_rank": rank_of(ranked_urls, ex["gt_url"]),
                    "routing_intent": route.get("intent", "unknown"),
                    "routing_action": routing_diag.get("routing_action", "unknown"),
                    "routing_fallback_used": bool(routing_diag.get("fallback_used")),
                }
            )

    return rows_by_system, latency, routing_diags


def print_score_table(rows_by_system: dict[str, list[dict[str, Any]]], latency: dict[str, float], k: int) -> None:
    print("\n" + "=" * 96)
    print(f"Search pipeline stage comparison (K={k})")
    print("=" * 96)
    print(f"{'system':<30} {f'Recall@{k}':>10} {'MRR':>10} {f'NDCG@{k}':>10} {'n':>6} {'sec/query':>10}")
    print("-" * 96)
    for system_name in system_order(rows_by_system):
        rows = rows_by_system.get(system_name, [])
        scores = compute_scores(rows, k)
        sec_per_query = latency.get(system_name, 0.0) / max(1, len(rows))
        print(
            f"{system_name:<30} {scores[f'Recall@{k}']:>10.4f} {scores['MRR']:>10.4f} "
            f"{scores[f'NDCG@{k}']:>10.4f} {scores['n']:>6} {sec_per_query:>10.3f}"
        )


def print_type_breakdown(rows_by_system: dict[str, list[dict[str, Any]]], k: int) -> None:
    print("\n" + "=" * 96)
    print(f"Type breakdown by system (NDCG@{k})")
    print("=" * 96)
    systems = system_order(rows_by_system)
    print(f"{'type':<18} " + " ".join(f"{system:>24}" for system in systems))
    print("-" * 96)
    all_types = sorted({row["type"] for rows in rows_by_system.values() for row in rows})
    for qtype in all_types:
        cells = []
        for system in systems:
            group = [row for row in rows_by_system[system] if row["type"] == qtype]
            scores = compute_scores(group, k)
            cells.append(f"{scores[f'NDCG@{k}']:>24.4f}")
        print(f"{qtype:<18} " + " ".join(cells))


def print_routing_summary(routing_diags: list[dict[str, Any]]) -> None:
    routes = [diag.get("route", {}) for diag in routing_diags]
    methods = Counter(route.get("method", "unknown") for route in routes)
    intents = Counter(route.get("intent", "unknown") for route in routes)
    actions = Counter(diag.get("routing_action", "unknown") for diag in routing_diags)
    fallback_count = sum(1 for diag in routing_diags if diag.get("fallback_used"))
    final_alphas = Counter(f"{diag.get('final_alpha', DEFAULT_ALPHA):.2f}" for diag in routing_diags)

    print("\n" + "=" * 96)
    print("Query routing summary")
    print("=" * 96)
    print(f"method counts     : {dict(sorted(methods.items()))}")
    print(f"intent counts     : {dict(sorted(intents.items()))}")
    print(f"action counts     : {dict(sorted(actions.items()))}")
    print(f"fallback used     : {fallback_count}/{len(routing_diags)}")
    print(f"final alpha counts: {dict(sorted(final_alphas.items()))}")


def print_delta_summary(rows_by_system: dict[str, list[dict[str, Any]]], k: int) -> None:
    print("\n" + "=" * 96)
    print(f"Top-{k} hit movement")
    print("=" * 96)
    base_rows = rows_by_system["Hybrid"]
    for system_name in system_order(rows_by_system):
        if system_name == "Hybrid":
            continue
        gained = 0
        lost = 0
        same_hit = 0
        same_miss = 0
        for base, candidate in zip(base_rows, rows_by_system[system_name]):
            if not base["hit"] and candidate["hit"]:
                gained += 1
            elif base["hit"] and not candidate["hit"]:
                lost += 1
            elif base["hit"] and candidate["hit"]:
                same_hit += 1
            else:
                same_miss += 1
        print(
            f"{system_name:<30} gained={gained:<4} lost={lost:<4} "
            f"same_hit={same_hit:<4} same_miss={same_miss:<4}"
        )


def print_routing_movement_breakdown(rows_by_system: dict[str, list[dict[str, Any]]]) -> None:
    base_rows = rows_by_system.get("Hybrid", [])
    routed_rows = rows_by_system.get("Hybrid+Routing", [])
    if not base_rows or not routed_rows:
        return

    groups: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"gained": 0, "lost": 0, "same_hit": 0, "same_miss": 0, "n": 0}
    )
    for base, routed in zip(base_rows, routed_rows):
        key = (routed.get("routing_intent", "unknown"), routed.get("routing_action", "unknown"))
        groups[key]["n"] += 1
        if not base["hit"] and routed["hit"]:
            groups[key]["gained"] += 1
        elif base["hit"] and not routed["hit"]:
            groups[key]["lost"] += 1
        elif base["hit"] and routed["hit"]:
            groups[key]["same_hit"] += 1
        else:
            groups[key]["same_miss"] += 1

    print("\n" + "=" * 96)
    print("Hybrid+Routing movement by intent/action")
    print("=" * 96)
    print(f"{'intent':<16} {'action':<26} {'n':>6} {'gained':>8} {'lost':>8} {'same_hit':>10} {'same_miss':>10}")
    print("-" * 96)
    for (intent, action), counts in sorted(groups.items()):
        print(
            f"{intent:<16} {action:<26} {counts['n']:>6} {counts['gained']:>8} "
            f"{counts['lost']:>8} {counts['same_hit']:>10} {counts['same_miss']:>10}"
        )


def print_rank_delta_summary(rows_by_system: dict[str, list[dict[str, Any]]]) -> None:
    base_rows = rows_by_system.get("Hybrid", [])
    routed_rows = rows_by_system.get("Hybrid+Routing", [])
    if not base_rows or not routed_rows:
        return

    groups: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"rank_up": 0, "rank_down": 0, "same_rank": 0, "base_missing": 0, "routed_missing": 0, "n": 0}
    )
    for base, routed in zip(base_rows, routed_rows):
        key = (routed.get("routing_intent", "unknown"), routed.get("routing_action", "unknown"))
        groups[key]["n"] += 1
        base_rank = base.get("gt_rank")
        routed_rank = routed.get("gt_rank")
        if base_rank is None and routed_rank is None:
            continue
        if base_rank is None:
            groups[key]["base_missing"] += 1
        elif routed_rank is None:
            groups[key]["routed_missing"] += 1
        elif routed_rank < base_rank:
            groups[key]["rank_up"] += 1
        elif routed_rank > base_rank:
            groups[key]["rank_down"] += 1
        else:
            groups[key]["same_rank"] += 1

    print("\n" + "=" * 96)
    print("GT rank delta by intent/action")
    print("=" * 96)
    print(f"{'intent':<16} {'action':<30} {'n':>6} {'up':>6} {'down':>6} {'same':>6} {'base_miss':>10} {'route_miss':>10}")
    print("-" * 96)
    for (intent, action), counts in sorted(groups.items()):
        print(
            f"{intent:<16} {action:<30} {counts['n']:>6} {counts['rank_up']:>6} "
            f"{counts['rank_down']:>6} {counts['same_rank']:>6} "
            f"{counts['base_missing']:>10} {counts['routed_missing']:>10}"
        )


def system_order(rows_by_system: dict[str, list[dict[str, Any]]]) -> list[str]:
    preferred = [
        "Hybrid",
        "RewriteOnly",
        "AlphaOnly",
        "IntentBoostOnly",
        "Hybrid+Routing",
        "Hybrid+Routing+Feature",
    ]
    return [name for name in preferred if name in rows_by_system]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5, help="Top-k for retrieval metrics.")
    parser.add_argument("--candidate-k", type=int, default=50, help="Fixed candidate pool size; app_api.py uses 50.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Hybrid vector weight for the baseline.")
    parser.add_argument("--profile-json", default="{}", help="User profile JSON for the feature reranker.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N examples.")
    parser.add_argument("--verbose-search-logs", action="store_true", help="Keep per-query search score logs.")
    parser.add_argument(
        "--disable-feature-llm",
        action="store_true",
        help="Use cached/rule fallback feature extraction instead of calling Gemini during feature reranking.",
    )
    parser.add_argument(
        "--skip-feature-rerank",
        action="store_true",
        help="Do not compute the Hybrid+Routing+Feature stage.",
    )
    parser.add_argument(
        "--disable-routing-llm",
        action="store_true",
        help="Use heuristic query routing instead of calling Gemini during query routing.",
    )
    parser.add_argument(
        "--no-feature-cache-write",
        action="store_true",
        help="Do not persist newly extracted/fallback feature rows during evaluation.",
    )
    parser.add_argument(
        "--allow-missing-gt",
        action="store_true",
        help="Continue even if some ground-truth URLs are missing from Pinecone.",
    )
    args = parser.parse_args()
    if args.no_feature_cache_write:
        feature_reranker_module._save_cache = lambda cache: None

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

    print("=" * 96)
    print("App search pipeline stage evaluation")
    print("=" * 96)
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

    rows_by_system, latency, routing_diags = evaluate_examples(
        examples,
        alpha=args.alpha,
        candidate_k=args.candidate_k,
        k=args.k,
        profile=profile,
        skip_feature_rerank=args.skip_feature_rerank,
        quiet=not args.verbose_search_logs,
    )

    print_score_table(rows_by_system, latency, args.k)
    print_type_breakdown(rows_by_system, args.k)
    print_delta_summary(rows_by_system, args.k)
    print_routing_movement_breakdown(rows_by_system)
    print_rank_delta_summary(rows_by_system)
    print_routing_summary(routing_diags)


if __name__ == "__main__":
    main()
