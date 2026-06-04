"""
Hybrid search (BM25 + dense vector) and LLM reply generation.
Extracted from app.py: hybrid_search, _build_bm25_index, generate_llm_reply.
"""
from __future__ import annotations

import json
import re
from time import perf_counter

from ..core.config import (
    GEMINI_API_KEY,
    GEMINI_API_KEY_PAID_TIER,
    GEMINI_REPLY_TIMEOUT_SECONDS,
    GEMINI_ROUTER_TIMEOUT_SECONDS,
    QUERY_ROUTING_ENABLED,
    SEARCH_ALPHA,
)
from ..core.models import get_embed_model, get_vector_collection, get_index_fingerprint, load_notices_cache
from ..core.utils import tokenize_ko
from .feature_reranker import rerank_notices
from .gemini_rest import generate_content_text

# Module-level BM25 cache (replaces @st.cache_data from app.py)
_bm25_cache: dict[str, tuple[tuple[int, int, str], tuple]] = {}

DEFAULT_ALPHA = 0.5
ROUTED_ALPHA = {
    "factual": 0.4,
    "procedural": 0.3,
    "conditional": 0.5,
}
ROUTING_CONFIDENCE_THRESHOLD = 0.80
ROUTING_SCORE_GAP_THRESHOLD = 0.08
CONDITIONAL_KEYWORDS = ("대상", "제외", "자격", "조건", "유의사항", "유의 사항", "단,", "단 ", "경우", "해당", "가능", "불가")
PROCEDURAL_KEYWORDS = ("신청방법", "신청 방법", "신청", "접수", "제출", "제출서류", "제출 서류", "절차", "링크", "양식", "첨부", "종합정보시스템")
FACTUAL_KEYWORDS = ("기간", "일시", "일정", "마감", "장소", "금액", "인원", "시간", "발표", "확인", "날짜")
QUERY_STOPWORDS = {
    "언제", "어디", "어떻게", "무엇", "가능", "신청", "방법", "기간", "마감", "장소", "대상", "조건",
    "알려줘", "인가요", "하나요", "되나요", "있나요", "어떤", "이후", "이전", "관련", "공지",
}
QUERY_EXPANSIONS = {
    "종정시": "종합정보시스템",
    "비교과": "비교과 프로그램",
    "장학": "장학금",
    "기숙": "기숙사",
    "유예": "학사학위취득 유예",
    "취소": "취소 신청",
}


def _category_filter_key(category_filter: str | list[str] | None) -> str:
    if isinstance(category_filter, list):
        values = [cat for cat in category_filter if cat and cat != "전체"]
        return "|".join(sorted(values)) if values else "전체"
    return category_filter if category_filter and category_filter != "전체" else "전체"


def _category_filter_where(category_filter: str | list[str] | None) -> dict | None:
    if isinstance(category_filter, list):
        values = [cat for cat in category_filter if cat and cat != "전체"]
        return {"category": {"$in": values}} if values else None
    return {"category": category_filter} if category_filter and category_filter != "전체" else None


def _build_bm25_index(category_filter: str | list[str] | None):
    from rank_bm25 import BM25Okapi

    key        = _category_filter_key(category_filter)
    collection = get_vector_collection()
    fingerprint = get_index_fingerprint()
    cached = _bm25_cache.get(key)
    if cached and cached[0] == fingerprint:
        return cached[1]

    where      = _category_filter_where(category_filter)
    all_data   = collection.get(include=["documents", "metadatas"], where=where)

    documents = all_data["documents"]
    metadatas = all_data["metadatas"]
    ids       = all_data["ids"]

    if not documents:
        return None, [], [], []

    tokenized_docs = [tokenize_ko(doc) for doc in documents]
    bm25           = BM25Okapi(tokenized_docs)
    _bm25_cache[key] = (fingerprint, (bm25, ids, documents, metadatas))
    return bm25, ids, documents, metadatas


def invalidate_bm25_cache() -> None:
    _bm25_cache.clear()


def _markdown_log_cell(value: object) -> str:
    return str(value or "").replace("\r", " ").replace("\n", " ").replace("|", r"\|")


def _parse_year_month(date_value: object) -> tuple[int, int] | None:
    match = re.search(r"(\d{4})\D{0,3}(\d{1,2})", str(date_value or ""))
    if not match:
        return None
    year, month = int(match.group(1)), int(match.group(2))
    if not 1 <= month <= 12:
        return None
    return year, month


def _build_recency_scores(
    metadatas: list[dict],
    meta_map: dict[str, dict],
) -> tuple[dict[str, float], dict[str, int]]:
    parsed_months = [_parse_year_month(meta.get("date")) for meta in metadatas if meta]
    parsed_months = [ym for ym in parsed_months if ym]
    if not parsed_months:
        return {}, {}

    latest_year, latest_month = max(parsed_months, key=lambda ym: ym[0] * 12 + ym[1])
    latest_index = latest_year * 12 + latest_month
    recency_scores: dict[str, float] = {}
    month_diffs: dict[str, int] = {}

    for did, meta in meta_map.items():
        ym = _parse_year_month(meta.get("date"))
        if not ym:
            month_diffs[did] = 999
            recency_scores[did] = 0
            continue

        month_diff = max(0, latest_index - (ym[0] * 12 + ym[1]))
        month_diffs[did] = month_diff
        recency_scores[did] = 1 / (1 + month_diff / 3)

    return recency_scores, month_diffs


def _normalize_user_query(query: str) -> str:
    normalized = re.sub(r"\s+", " ", query or "").strip()
    replacements = {
        "언제야": "언제인가요",
        "어케": "어떻게",
        "어떡해": "어떻게",
        "뭐야": "무엇인가요",
        "얼마야": "얼마인가요",
        "가능해": "가능한가요",
        "되나": "가능한가요",
        "해도돼": "해도 되나요",
        "신청 어케": "신청 방법",
        "비교과": "비교과 프로그램",
        "종정시": "종합정보시스템",
    }
    for src, dst in replacements.items():
        normalized = normalized.replace(src, dst)
    return normalized


def _looks_like_user_short(query: str) -> bool:
    stripped = (query or "").strip()
    if not stripped:
        return False
    token_count = len(stripped.split())
    casual_markers = ("언제야", "얼마야", "뭐야", "어디서", "어케", "알려줘", "되나", "가능해", "해?")
    return token_count <= 5 or any(marker in stripped for marker in casual_markers)


def _heuristic_intent(query: str) -> tuple[str, float]:
    q = query or ""
    if any(k in q for k in ("가능", "대상", "조건", "자격", "제외", "받을 수", "해도 되", "경우")):
        return "conditional", 0.86
    if any(k in q for k in ("방법", "절차", "신청", "제출", "링크", "어떻게", "종합정보시스템")):
        return "procedural", 0.86
    if any(k in q for k in ("언제", "기간", "마감", "장소", "어디", "얼마", "대상", "일시", "날짜")):
        return "factual", 0.86
    return "기타", 0.70


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        cleaned = str(value or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def _query_terms(query: str) -> list[str]:
    terms = re.findall(r"[가-힣A-Za-z0-9]{2,}", query or "")
    return _dedupe_keep_order([term for term in terms if term not in QUERY_STOPWORDS])


def _expanded_terms(query: str) -> list[str]:
    expanded = []
    for src, dst in QUERY_EXPANSIONS.items():
        if src in (query or ""):
            expanded.append(dst)
    return _dedupe_keep_order(expanded)


def _section_targets_for_intent(intent: str) -> list[str]:
    if intent == "procedural":
        return list(PROCEDURAL_KEYWORDS)
    if intent == "factual":
        return list(FACTUAL_KEYWORDS)
    if intent == "conditional":
        return list(CONDITIONAL_KEYWORDS)
    return []


def _build_query_features(query: str, rewritten: str, intent: str) -> dict:
    source_text = " ".join([query or "", rewritten or ""])
    must_keep_terms = _query_terms(query)
    expanded_terms = _expanded_terms(source_text)
    time_terms = re.findall(r"\d{4}\s*년|\d{1,2}\s*월|\d{1,2}\s*일|[12]학기|여름|겨울|하계|동계", source_text)
    return {
        "must_keep_terms": must_keep_terms,
        "expanded_terms": expanded_terms,
        "entities": _dedupe_keep_order(must_keep_terms[:8] + expanded_terms),
        "time_terms": _dedupe_keep_order(time_terms),
        "section_targets": _section_targets_for_intent(intent),
    }


def _build_search_query(original: str, rewritten: str, features: dict) -> str:
    terms = [original, rewritten]
    terms.extend(features.get("must_keep_terms", []))
    terms.extend(features.get("expanded_terms", []))
    return " ".join(_dedupe_keep_order([term for term in terms if term]))


def _extract_json_object(text: str) -> dict | None:
    if not text:
        return None
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _route_query_with_gemini(query: str) -> dict:
    if not GEMINI_API_KEY:
        normalized = _normalize_user_query(query)
        intent, confidence = _heuristic_intent(normalized)
        features = _build_query_features(query, normalized, intent)
        return {
            "original_query": query,
            "search_query": _build_search_query(query, normalized, features),
            "is_user_short": _looks_like_user_short(query),
            "intent": intent,
            "confidence": confidence,
            "method": "heuristic",
            "features": features,
        }

    try:
        prompt = f"""너는 한성대학교 공지 검색 쿼리 라우터다.
사용자 질문이 짧은 구어체(user_short)이면 생략된 목적어를 복원하고 오타/약어/구어체를 표준 검색 질의로 정규화하라.
최종 의도는 factual, procedural, conditional, 기타 중 하나로 분류하라.

JSON만 반환:
{{
  "is_user_short": true/false,
  "rewritten_query": "검색에 사용할 한국어 질의",
  "intent": "factual|procedural|conditional|기타",
  "confidence": 0.0,
  "must_keep_terms": ["원문에서 보존할 고유명사/약어/프로그램명"],
  "expanded_terms": ["검색 recall을 높일 확장어"],
  "entities": ["프로그램명, 부서명, 장학금명 등"],
  "time_terms": ["연도, 학기, 월, 날짜 표현"],
  "section_targets": ["신청방법, 기간, 대상처럼 본문에서 찾아야 할 섹션 힌트"]
}}

사용자 질문: {query}"""
        response_text = generate_content_text(
            api_key=GEMINI_API_KEY,
            prompt=prompt,
            timeout=GEMINI_ROUTER_TIMEOUT_SECONDS,
            temperature=0,
            response_mime_type="application/json",
        )
        parsed = _extract_json_object(response_text) or {}
    except Exception as exc:
        print(f"[query-router] Gemini 분류 실패, heuristic 사용: {exc}", flush=True)
        normalized = _normalize_user_query(query)
        intent, confidence = _heuristic_intent(normalized)
        features = _build_query_features(query, normalized, intent)
        return {
            "original_query": query,
            "search_query": _build_search_query(query, normalized, features),
            "is_user_short": _looks_like_user_short(query),
            "intent": intent,
            "confidence": confidence,
            "method": "heuristic_fallback",
            "features": features,
        }

    intent = str(parsed.get("intent") or "기타").strip()
    if intent not in {"factual", "procedural", "conditional", "기타"}:
        intent = "기타"
    rewritten = _normalize_user_query(str(parsed.get("rewritten_query") or query))
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    parsed_features = {
        "must_keep_terms": parsed.get("must_keep_terms") if isinstance(parsed.get("must_keep_terms"), list) else [],
        "expanded_terms": parsed.get("expanded_terms") if isinstance(parsed.get("expanded_terms"), list) else [],
        "entities": parsed.get("entities") if isinstance(parsed.get("entities"), list) else [],
        "time_terms": parsed.get("time_terms") if isinstance(parsed.get("time_terms"), list) else [],
        "section_targets": parsed.get("section_targets") if isinstance(parsed.get("section_targets"), list) else [],
    }
    heuristic_features = _build_query_features(query, rewritten, intent)
    features = {
        key: _dedupe_keep_order([*heuristic_features.get(key, []), *parsed_features.get(key, [])])
        for key in ("must_keep_terms", "expanded_terms", "entities", "time_terms", "section_targets")
    }
    return {
        "original_query": query,
        "search_query": _build_search_query(query, rewritten, features),
        "is_user_short": bool(parsed.get("is_user_short", _looks_like_user_short(query))),
        "intent": intent,
        "confidence": max(0.0, min(1.0, confidence)),
        "method": "gemini",
        "features": features,
    }


def _top_score_gap(candidates: list[dict]) -> float:
    if len(candidates) < 2:
        return 0.0
    scores = sorted((_safe_float(c.get("score"), 0.0) for c in candidates), reverse=True)
    return scores[0] - scores[1]


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _keyword_boost(text: str, keywords: tuple[str, ...], weight: float) -> float:
    hits = sum(1 for keyword in keywords if keyword in (text or ""))
    return min(weight * hits, weight * 3)


def _conditional_match_score(query: str, text: str) -> float:
    query_terms = set(re.findall(r"[가-힣A-Za-z0-9]{2,}", query or ""))
    condition_terms = {
        term for term in re.findall(r"[가-힣A-Za-z0-9]{2,}", text or "")
        if any(marker in term for marker in ("학년", "재학생", "휴학생", "졸업", "소득", "성적", "전공", "트랙", "학부", "대상"))
    }
    if not query_terms or not condition_terms:
        return 0.0
    overlap = len(query_terms & condition_terms)
    return min(0.12, overlap * 0.04)


def _term_hit_count(text: str, terms: list[str]) -> int:
    return sum(1 for term in terms if term and term in text)


def _feature_bonus_cap(intent: str) -> float:
    if intent == "procedural":
        return 0.21
    if intent == "conditional":
        return 0.13
    return 0.18


def _score_query_features(item: dict, route: dict) -> tuple[float, dict[str, float]]:
    features = route.get("features") or {}
    intent = route.get("intent")
    title = str(item.get("title") or "")
    content = str(item.get("content") or item.get("body") or "")
    category = str(item.get("category") or "")
    text = " ".join([title, content, category])

    entity_hits = _term_hit_count(title, features.get("entities", []))
    body_hits = _term_hit_count(text, features.get("must_keep_terms", []))
    expanded_hits = _term_hit_count(text, features.get("expanded_terms", []))
    time_hits = _term_hit_count(text, features.get("time_terms", []))
    section_hits = _term_hit_count(text, features.get("section_targets", []))

    parts = {
        "title_entity": min(entity_hits * 0.025, 0.075),
        "body_exact": min(body_hits * 0.010, 0.050),
        "expanded": min(expanded_hits * 0.012, 0.036),
        "time": min(time_hits * 0.015, 0.045),
        "section": min(section_hits * 0.012, 0.060),
    }

    if intent == "procedural":
        parts["intent_section"] = min(section_hits * 0.018, 0.120)
    elif intent == "factual":
        parts["intent_section"] = min((section_hits + time_hits) * 0.016, 0.080)
    elif intent == "conditional":
        condition_score = _conditional_match_score(route.get("search_query") or route.get("original_query") or "", text)
        parts["intent_section"] = min(section_hits * 0.020 + condition_score, 0.060)
    else:
        parts["intent_section"] = 0.0

    bonus = min(sum(parts.values()), _feature_bonus_cap(str(intent)))
    return bonus, {key: round(value, 6) for key, value in parts.items() if value > 0}


def _apply_intent_boosts(candidates: list[dict], route: dict, meta: dict | None = None) -> list[dict]:
    intent = route.get("intent")
    query = route.get("search_query") or route.get("original_query") or ""
    boosted: list[dict] = []
    for item in candidates:
        row = dict(item)
        text = " ".join(str(row.get(key, "")) for key in ("title", "content", "category"))
        bonus, boost_parts = _score_query_features(row, route)

        if (route.get("is_user_short") or intent == "conditional") and intent == "conditional":
            eligibility_boost = _keyword_boost(text, CONDITIONAL_KEYWORDS, 0.07)
            logic_boost = _conditional_match_score(query, text)
            bonus = min(bonus + eligibility_boost + logic_boost, _feature_bonus_cap(str(intent)))
            boost_parts["routing_eligibility_keyword"] = round(eligibility_boost, 6)
            boost_parts["routing_condition_match"] = round(logic_boost, 6)

        base_score = _safe_float(row.get("score"), 0.0)
        row["score"] = round(base_score + bonus, 6)
        row["query_routing_boost"] = {
            "intent": intent,
            "base_score": round(base_score, 6),
            "bonus": round(bonus, 6),
            "parts": boost_parts,
        }
        boosted.append(row)
    boosted.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return boosted


def _feature_union_candidates(
    baseline: list[dict],
    routed: list[dict],
) -> list[dict]:
    merged: dict[str, dict] = {}

    for source, rows in (("baseline", baseline), ("routed", routed)):
        for rank, item in enumerate(rows, start=1):
            key = str(item.get("url") or item.get("_id") or item.get("title") or "")
            if not key:
                continue

            if key not in merged:
                row = dict(item)
                row["pre_union_score"] = _safe_float(item.get("score"), 0.0)
                row["routing_sources"] = [source]
                row["routing_source_ranks"] = {source: rank}
                if source == "routed":
                    row["score"] = round(_safe_float(row.get("score"), 0.0) - 0.04, 6)
                    row["routed_only_penalty"] = 0.04
                merged[key] = row
                continue

            row = merged[key]
            row["routing_source_ranks"][source] = rank
            if source not in row["routing_sources"]:
                row["routing_sources"].append(source)

            item_score = _safe_float(item.get("score"), 0.0)
            if item_score > _safe_float(row.get("pre_union_score"), 0.0):
                preserved = {
                    "routing_sources": row["routing_sources"],
                    "routing_source_ranks": row["routing_source_ranks"],
                }
                row.update(item)
                row.update(preserved)
                row["pre_union_score"] = item_score

    results = list(merged.values())
    results.sort(key=lambda row: row.get("score", 0.0), reverse=True)
    return results


def _hybrid_candidate_search(
    query: str,
    top_k: int = 5,
    alpha: float = SEARCH_ALPHA,
    category_filter: str | list[str] | None = None,
    candidate_k: int | None = None,
) -> tuple[list[dict], dict]:
    model      = get_embed_model()
    collection = get_vector_collection()
    where      = _category_filter_where(category_filter)

    bm25, ids, documents, metadatas = _build_bm25_index(category_filter)
    if bm25 is None:
        return [], {}

    q_emb     = model.encode(query).tolist()
    candidate_limit = max(top_k, candidate_k or top_k)
    n_results = min(max(candidate_limit * 5, top_k * 5), len(documents))
    vr        = collection.query(
        query_embeddings=[q_emb],
        n_results=n_results,
        include=["metadatas", "distances"],
        where=where,
    )

    vector_scores: dict[str, float] = {}
    raw_dist = vr["distances"][0]
    if raw_dist:
        max_sim = 1 - min(raw_dist)
        min_sim = 1 - max(raw_dist)
        for vid, dist in zip(vr["ids"][0], raw_dist):
            sim  = 1 - dist
            norm = (sim - min_sim) / (max_sim - min_sim + 1e-9)
            vector_scores[vid] = norm

    bm25_raw    = bm25.get_scores(tokenize_ko(query))
    bm25_max    = max(bm25_raw) if max(bm25_raw) > 0 else 1
    bm25_scores = {did: s / bm25_max for did, s in zip(ids, bm25_raw)}

    all_ids = set(vector_scores) | set(bm25_scores)
    meta_map  = dict(zip(ids, metadatas))
    doc_map   = dict(zip(ids, documents))
    recency_scores, month_diffs = _build_recency_scores(metadatas, meta_map)
    base = {
        did: alpha * vector_scores.get(did, 0) + (1 - alpha) * bm25_scores.get(did, 0)
        for did in all_ids
    }
    final = {
        did: base[did] * (1 + 0.15 * recency_scores.get(did, 0))
        for did in all_ids
    }
    seen_urls: dict[str, dict] = {}
    seen_score_ids: list[str] = []
    for did in sorted(final, key=lambda x: final[x], reverse=True):
        meta = meta_map.get(did)
        if not meta:
            continue
        url = meta["url"]
        if url not in seen_urls:
            seen_score_ids.append(did)
            seen_urls[url] = {
                **meta,
                "_id": did,
                "score": round(final[did], 4),
                "_base_score": round(base.get(did, 0), 6),
                "_vector_score": round(vector_scores.get(did, 0), 6),
                "_bm25_score": round(bm25_scores.get(did, 0), 6),
                "_recency_score": round(recency_scores.get(did, 0), 6),
                "_month_diff": month_diffs.get(did, 999),
                "content": doc_map.get(did, ""),
            }
        if len(seen_urls) >= candidate_limit:
            break

    score_log_rows = seen_score_ids[:5]
    if score_log_rows:
        print(
            "| tag | query | rank | final | base | vector | bm25 | recency | month_diff | date | title |\n"
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
            flush=True,
        )
    for rank, did in enumerate(score_log_rows, start=1):
        meta = meta_map.get(did, {})
        print(
            "| search-score | "
            f"{_markdown_log_cell(query)} | "
            f"{rank} | "
            f"{final.get(did, 0):.4f} | "
            f"{base.get(did, 0):.4f} | "
            f"{vector_scores.get(did, 0):.4f} | "
            f"{bm25_scores.get(did, 0):.4f} | "
            f"{recency_scores.get(did, 0):.4f} | "
            f"{month_diffs.get(did, 999)} | "
            f"{_markdown_log_cell(meta.get('date', ''))} | "
            f"{_markdown_log_cell(meta.get('title', ''))} |",
            flush=True,
        )

    candidates = list(seen_urls.values())
    diagnostics = {
        "alpha": alpha,
        "top_score_gap": _top_score_gap(candidates),
        "recency_scores": recency_scores,
        "month_diffs": month_diffs,
    }
    return candidates, diagnostics


def _routed_candidates_before_feature_rerank(
    query: str,
    top_k: int,
    alpha: float,
    category_filter: str | list[str] | None,
    candidate_k: int | None,
    profile: dict | None,
) -> tuple[str, list[dict], dict]:
    if not QUERY_ROUTING_ENABLED:
        candidate_limit = max(top_k, candidate_k or top_k)
        candidates, _ = _hybrid_candidate_search(
            query=query,
            top_k=top_k,
            alpha=alpha,
            category_filter=category_filter,
            candidate_k=candidate_limit,
        )
        diagnostics = {
            "route": {
                "original_query": query,
                "search_query": query,
                "intent": "disabled",
                "confidence": 0.0,
                "method": "disabled",
                "features": {},
            },
            "requested_alpha": alpha,
            "routed_alpha": alpha,
            "final_alpha": alpha,
            "fallback_used": True,
            "routing_action": "disabled",
            "top_score_gap": 0.0,
            "baseline_candidate_count": len(candidates),
            "routed_candidate_count": 0,
            "candidate_count": len(candidates),
        }
        for item in candidates:
            item["query_routing"] = diagnostics
        return query, candidates, diagnostics

    route = _route_query_with_gemini(query)
    search_query = route.get("search_query") or query
    candidate_limit = max(top_k, candidate_k or top_k)
    routed_alpha = ROUTED_ALPHA.get(route["intent"], DEFAULT_ALPHA)
    baseline_candidates, _ = _hybrid_candidate_search(
        query=query,
        top_k=top_k,
        alpha=alpha,
        category_filter=category_filter,
        candidate_k=candidate_limit,
    )
    baseline_boosted = _apply_intent_boosts(baseline_candidates, route)
    route_gate_eligible = (
        route["intent"] in ROUTED_ALPHA
        and route["confidence"] >= ROUTING_CONFIDENCE_THRESHOLD
    )

    if not route_gate_eligible:
        candidates = baseline_boosted
        diagnostics = {
            "route": route,
            "requested_alpha": alpha,
            "routed_alpha": routed_alpha,
            "final_alpha": alpha,
            "fallback_used": True,
            "routing_action": "baseline_feature_only",
            "top_score_gap": 0.0,
            "baseline_candidate_count": len(baseline_candidates),
            "routed_candidate_count": 0,
            "candidate_count": len(candidates),
        }
        print(
            "| query-routing | "
            f"{_markdown_log_cell(query)} | "
            f"search_query={_markdown_log_cell(search_query)} | "
            f"intent={route['intent']} | "
            f"confidence={route['confidence']:.3f} | "
            f"routed_alpha={routed_alpha:.2f} | "
            f"final_alpha={alpha:.2f} | "
            "fallback=True | "
            "action=baseline_feature_only | "
            "gap=0.0000 | ",
            flush=True,
        )
        for item in candidates:
            item["query_routing"] = diagnostics
        return query, candidates, diagnostics

    routed_raw, routed_diag = _hybrid_candidate_search(
        query=search_query,
        top_k=top_k,
        alpha=routed_alpha,
        category_filter=category_filter,
        candidate_k=candidate_limit,
    )
    routed_boosted = _apply_intent_boosts(routed_raw, route, routed_diag)

    use_routed_alpha = (
        route["intent"] == "factual"
        and routed_diag.get("top_score_gap", 0.0) >= ROUTING_SCORE_GAP_THRESHOLD
    )

    if use_routed_alpha:
        candidates = _feature_union_candidates(baseline_boosted, routed_boosted)
        final_alpha = routed_alpha
        fallback_used = False
        routing_action = "baseline_routed_feature_union"
    else:
        candidates = baseline_boosted
        final_alpha = alpha
        fallback_used = True
        routing_action = "baseline_feature_only"

    diagnostics = {
        "route": route,
        "requested_alpha": alpha,
        "routed_alpha": routed_alpha,
        "final_alpha": final_alpha,
        "fallback_used": fallback_used,
        "routing_action": routing_action,
        "top_score_gap": round(routed_diag.get("top_score_gap", 0.0), 6),
        "baseline_candidate_count": len(baseline_candidates),
        "routed_candidate_count": len(routed_boosted),
        "candidate_count": len(candidates),
    }
    print(
        "| query-routing | "
        f"{_markdown_log_cell(query)} | "
        f"search_query={_markdown_log_cell(search_query)} | "
        f"intent={route['intent']} | "
        f"confidence={route['confidence']:.3f} | "
        f"routed_alpha={routed_alpha:.2f} | "
        f"final_alpha={final_alpha:.2f} | "
        f"fallback={fallback_used} | "
        f"action={routing_action} | "
        f"gap={diagnostics['top_score_gap']:.4f} | ",
        flush=True,
    )
    for item in candidates:
        item["query_routing"] = diagnostics
    return search_query, candidates, diagnostics


def hybrid_search(
    query: str,
    top_k: int = 5,
    alpha: float = SEARCH_ALPHA,
    category_filter: str | list[str] | None = None,
    candidate_k: int | None = None,
    feature_rerank: bool = False,
    profile: dict | None = None,
) -> list[dict]:
    search_started_at = perf_counter()
    if feature_rerank:
        search_query, candidates, routing_diag = _routed_candidates_before_feature_rerank(
            query=query,
            top_k=top_k,
            alpha=alpha,
            category_filter=category_filter,
            candidate_k=candidate_k,
            profile=profile,
        )
        hybrid_elapsed_ms = (perf_counter() - search_started_at) * 1000
        rerank_started_at = perf_counter()
        reranked = rerank_notices(candidates, profile=profile or {}, top_k=top_k)
        rerank_elapsed_ms = (perf_counter() - rerank_started_at) * 1000
        total_elapsed_ms = (perf_counter() - search_started_at) * 1000
        for item in reranked:
            item["query_routing"] = routing_diag
        print(
            "| search-timing | "
            f"{_markdown_log_cell(search_query)} | "
            f"hybrid_candidates={len(candidates)} | "
            f"top_k={top_k} | "
            f"hybrid_ms={hybrid_elapsed_ms:.1f} | "
            f"reranker_ms={rerank_elapsed_ms:.1f} | "
            f"total_ms={total_elapsed_ms:.1f} |",
            flush=True,
        )
        return reranked

    candidates, _ = _hybrid_candidate_search(
        query=query,
        top_k=top_k,
        alpha=alpha,
        category_filter=category_filter,
        candidate_k=candidate_k,
    )
    result = candidates[:top_k]
    total_elapsed_ms = (perf_counter() - search_started_at) * 1000
    print(
        "| search-timing | "
        f"{_markdown_log_cell(query)} | "
        f"hybrid_candidates={len(candidates)} | "
        f"top_k={top_k} | "
        f"hybrid_ms={total_elapsed_ms:.1f} | "
        "reranker_ms=0.0 | "
        f"total_ms={total_elapsed_ms:.1f} |",
        flush=True,
    )
    return result


def generate_llm_reply(
    user_query: str,
    results: list[dict],
    profile: dict,
    is_first: bool = False,
) -> str:
    if not GEMINI_API_KEY_PAID_TIER:
        if results:
            return f"총 {len(results)}개의 관련 공지를 찾았습니다."
        return "관련 공지를 찾지 못했습니다. GEMINI_API_KEY_PAID_TIER를 설정해 주세요."

    if not results:
        return "관련 공지를 찾지 못했습니다. 다른 키워드로 검색해보세요."

    notices     = load_notices_cache()
    body_map    = {n["url"]: n.get("body", "") for n in notices}
    context_parts = []
    for i, r in enumerate(results[:5], 1):
        body = (r.get("content") or body_map.get(r["url"], ""))[:800]
        context_parts.append(
            f"[공지 {i}]\n제목: {r['title']}\n날짜: {r['date']}\n내용: {body if body else '(본문 없음)'}"
        )
    context = "\n\n".join(context_parts)

    name     = profile.get("name", "")
    greeting = f"{name}님, 안녕하세요. " if is_first and name else ""

    prompt = f"""당신은 한성대학교 공지사항 안내 도우미입니다.

아래 공지사항 본문을 바탕으로 사용자 질문에 직접적이고 구체적으로 답변하세요.
- 날짜, 금액, 조건 등 구체적인 정보가 있으면 반드시 포함하세요.
- "공지를 참고하세요" 같은 말은 절대 하지 마세요. 정보를 직접 알려주세요.
- 2~3문장으로 간결하게 답변하세요.
- 답변 시작: "{greeting}"{"(인사 없이 바로 답변)" if not is_first else ""}

[공지 본문]
{context}

[질문]
{user_query}"""

    try:
        reply_started_at = perf_counter()
        response_text = generate_content_text(
            api_key=GEMINI_API_KEY_PAID_TIER,
            prompt=prompt,
            timeout=GEMINI_REPLY_TIMEOUT_SECONDS,
        )
        reply_elapsed_ms = (perf_counter() - reply_started_at) * 1000
        print(f"[reply-generator] Gemini 응답 생성 성공: {reply_elapsed_ms:.1f}ms", flush=True)
        return response_text
    except Exception as e:
        print(f"[reply-generator] Gemini 응답 생성 실패, fallback 사용: {e}", flush=True)
        return f"외부 요인으로 LLM 응답을 생성하지 못했습니다. 검색 결과 {len(results)}개 찾았습니다."
