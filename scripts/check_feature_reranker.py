#!/usr/bin/env python3
"""Compare hybrid search Top-N with feature-reranked Top-N for one query."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.core.config import SEARCH_ALPHA  # noqa: E402
from api.services.search_service import hybrid_search  # noqa: E402


DEFAULT_PROFILE = {
    "grade": "3학년",
    "track": "모바일소프트웨어트랙",
    "college": "IT공과대학",
    "interests": ["장학금", "비교과", "취업"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print the hybrid-search Top-N and feature-reranked Top-N for the same query. "
            "The reranker is applied over candidate-k hybrid candidates."
        )
    )
    parser.add_argument("query", help="검색 쿼리")
    parser.add_argument("--top-k", type=int, default=20, help="출력할 결과 수 (default: 20)")
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=50,
        help="reranker에 넘길 hybrid 후보 수 (default: 50)",
    )
    parser.add_argument("--alpha", type=float, default=SEARCH_ALPHA, help=f"hybrid alpha (default: {SEARCH_ALPHA})")
    parser.add_argument(
        "--category",
        default=None,
        help='카테고리 필터. 여러 개는 comma로 구분. 예: "장학금,학자금/근로장학"',
    )
    parser.add_argument(
        "--profile-json",
        default=None,
        help="reranker 사용자 프로필 JSON 문자열. 미지정 시 기본 학부생 프로필 사용",
    )
    parser.add_argument(
        "--empty-profile",
        action="store_true",
        help="기본 프로필을 쓰지 않고 빈 profile로 reranker 실행",
    )
    parser.add_argument(
        "--show-internal-score-log",
        action="store_true",
        help="hybrid_search 내부 score 로그도 함께 출력",
    )
    return parser.parse_args()


def parse_category(raw: str | None) -> str | list[str] | None:
    if not raw or raw == "전체":
        return None
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        return None
    return values if len(values) > 1 else values[0]


def parse_profile(args: argparse.Namespace) -> dict[str, Any]:
    if args.empty_profile:
        return {}
    if not args.profile_json:
        return dict(DEFAULT_PROFILE)
    try:
        profile = json.loads(args.profile_json)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--profile-json 파싱 실패: {exc}") from exc
    if not isinstance(profile, dict):
        raise SystemExit("--profile-json은 JSON object여야 합니다.")
    return profile


def run_search(*, show_internal_score_log: bool, **kwargs: Any) -> tuple[list[dict[str, Any]], float]:
    started_at = perf_counter()
    if show_internal_score_log:
        return hybrid_search(**kwargs), (perf_counter() - started_at) * 1000

    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        results = hybrid_search(**kwargs)
    return results, (perf_counter() - started_at) * 1000


def clean_cell(value: Any, max_len: int | None = None) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").replace("|", r"\|").strip()
    if max_len and len(text) > max_len:
        return text[: max_len - 1] + "…"
    return text


def score(value: Any) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return ""


def audience_label(item: dict[str, Any]) -> str:
    features = ((item.get("feature_reranker") or {}).get("features") or {})
    return ",".join(features.get("target_audiences") or [])


def extraction_label(item: dict[str, Any]) -> str:
    features = ((item.get("feature_reranker") or {}).get("features") or {})
    extraction = features.get("extraction") or {}
    method = extraction.get("method") or ""
    if extraction.get("llm_failed"):
        return f"{method}:failed" if method else "failed"
    return str(method)


def print_hybrid_table(items: list[dict[str, Any]]) -> None:
    print("\n## Hybrid Search Top 20 (Before Feature Reranker)")
    print("| rank | score | date | category | title |")
    print("| ---: | ---: | --- | --- | --- |")
    for rank, item in enumerate(items, start=1):
        print(
            "| "
            f"{rank} | "
            f"{score(item.get('score'))} | "
            f"{clean_cell(item.get('date') or item.get('posted_at'))} | "
            f"{clean_cell(item.get('category'))} | "
            f"{clean_cell(item.get('title'), 90)} |"
        )


def print_reranked_table(items: list[dict[str, Any]], hybrid_rank_by_url: dict[str, int]) -> None:
    print("\n## Feature Reranker Top 20 (After Feature Reranker)")
    print("| rank | hybrid_rank | change | base | rerank | boost | audience | extraction | date | title |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |")
    for rank, item in enumerate(items, start=1):
        meta = item.get("feature_reranker") or {}
        before_rank = hybrid_rank_by_url.get(str(item.get("url") or ""))
        if before_rank:
            change = before_rank - rank
            change_text = f"+{change}" if change > 0 else str(change)
        else:
            change_text = "new"
        print(
            "| "
            f"{rank} | "
            f"{before_rank or ''} | "
            f"{change_text} | "
            f"{score(item.get('base_score'))} | "
            f"{score(item.get('score'))} | "
            f"{score(meta.get('feature_boost'))} | "
            f"{clean_cell(audience_label(item))} | "
            f"{clean_cell(extraction_label(item))} | "
            f"{clean_cell(item.get('date') or item.get('posted_at'))} | "
            f"{clean_cell(item.get('title'), 90)} |"
        )


def print_rank_changes(reranked: list[dict[str, Any]], hybrid_rank_by_url: dict[str, int]) -> None:
    moved = []
    for rank, item in enumerate(reranked, start=1):
        before_rank = hybrid_rank_by_url.get(str(item.get("url") or ""))
        if before_rank and before_rank != rank:
            moved.append((abs(before_rank - rank), before_rank, rank, item))
    moved.sort(reverse=True, key=lambda row: row[0])

    print("\n## Biggest Rank Changes")
    if not moved:
        print("순위 변화가 없습니다.")
        return
    print("| hybrid_rank | rerank_rank | delta | title |")
    print("| ---: | ---: | ---: | --- |")
    for _, before_rank, after_rank, item in moved[:10]:
        delta = before_rank - after_rank
        delta_text = f"+{delta}" if delta > 0 else str(delta)
        print(f"| {before_rank} | {after_rank} | {delta_text} | {clean_cell(item.get('title'), 100)} |")


def main() -> None:
    args = parse_args()
    category = parse_category(args.category)
    profile = parse_profile(args)

    print(f"# Feature Reranker Check")
    print(f"- query: {args.query}")
    print(f"- top_k: {args.top_k}")
    print(f"- candidate_k: {args.candidate_k}")
    print(f"- alpha: {args.alpha}")
    print(f"- category: {category or '전체'}")
    print(f"- profile: {json.dumps(profile, ensure_ascii=False, sort_keys=True)}")

    hybrid_top, hybrid_only_ms = run_search(
        show_internal_score_log=args.show_internal_score_log,
        query=args.query,
        top_k=args.top_k,
        alpha=args.alpha,
        category_filter=category,
        candidate_k=args.candidate_k,
        feature_rerank=False,
        profile=profile,
    )
    reranked_top, reranked_total_ms = run_search(
        show_internal_score_log=args.show_internal_score_log,
        query=args.query,
        top_k=args.top_k,
        alpha=args.alpha,
        category_filter=category,
        candidate_k=args.candidate_k,
        feature_rerank=True,
        profile=profile,
    )

    candidate_hybrid, candidate_hybrid_ms = run_search(
        show_internal_score_log=False,
        query=args.query,
        top_k=args.candidate_k,
        alpha=args.alpha,
        category_filter=category,
        candidate_k=args.candidate_k,
        feature_rerank=False,
        profile=profile,
    )
    hybrid_rank_by_url = {
        str(item.get("url") or ""): rank
        for rank, item in enumerate(candidate_hybrid, start=1)
        if item.get("url")
    }

    print("\n## Search Timing")
    print("| mode | elapsed_ms | note |")
    print("| --- | ---: | --- |")
    print(f"| hybrid_only_top_{args.top_k} | {hybrid_only_ms:.1f} | feature_rerank=False |")
    print(f"| reranker_top_{args.top_k}_from_{args.candidate_k} | {reranked_total_ms:.1f} | feature_rerank=True |")
    print(f"| hybrid_candidate_map_top_{args.candidate_k} | {candidate_hybrid_ms:.1f} | used for rank-change comparison |")

    print_hybrid_table(hybrid_top)
    print_reranked_table(reranked_top, hybrid_rank_by_url)
    print_rank_changes(reranked_top, hybrid_rank_by_url)


if __name__ == "__main__":
    main()
