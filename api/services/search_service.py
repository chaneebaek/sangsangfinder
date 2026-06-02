"""
Hybrid search (BM25 + dense vector) and LLM reply generation.
Extracted from app.py: hybrid_search, _build_bm25_index, generate_llm_reply.
"""
from __future__ import annotations

import re

from ..core.config import GEMINI_API_KEY, SEARCH_ALPHA
from ..core.models import get_embed_model, get_vector_collection, get_index_fingerprint, load_notices_cache
from ..core.utils import tokenize_ko
from .feature_reranker import rerank_notices

# Module-level BM25 cache (replaces @st.cache_data from app.py)
_bm25_cache: dict[str, tuple[tuple[int, int, str], tuple]] = {}


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


def hybrid_search(
    query: str,
    top_k: int = 5,
    alpha: float = SEARCH_ALPHA,
    category_filter: str | list[str] | None = None,
    candidate_k: int | None = None,
    feature_rerank: bool = False,
    profile: dict | None = None,
) -> list[dict]:
    model      = get_embed_model()
    collection = get_vector_collection()
    where      = _category_filter_where(category_filter)

    bm25, ids, documents, metadatas = _build_bm25_index(category_filter)
    if bm25 is None:
        return []

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
                "score": round(final[did], 4),
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
    if feature_rerank:
        return rerank_notices(candidates, profile=profile or {}, top_k=top_k)
    return candidates[:top_k]


def generate_llm_reply(
    user_query: str,
    results: list[dict],
    profile: dict,
    is_first: bool = False,
) -> str:
    if not GEMINI_API_KEY:
        if results:
            return f"총 {len(results)}개의 관련 공지를 찾았습니다."
        return "관련 공지를 찾지 못했습니다. GEMINI_API_KEY를 설정해 주세요."

    if not results:
        return "관련 공지를 찾지 못했습니다. 다른 키워드로 검색해보세요."

    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")
    except Exception as e:
        return f"[Gemini 모델 로드 오류] {e}"

    notices     = load_notices_cache()
    body_map    = {n["url"]: n.get("body", "") for n in notices}
    context_parts = []
    for i, r in enumerate(results[:3], 1):
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
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[Gemini 오류] {e}"
