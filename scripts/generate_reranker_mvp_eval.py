from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("VECTOR_DB", "pinecone")

from api.core.utils import clean_title, tokenize_ko  # noqa: E402
from api.core.config import PINECONE_INDEX_NAME, PINECONE_NAMESPACE  # noqa: E402
from api.core.models import get_embed_model, get_vector_collection  # noqa: E402


TODAY = date(2026, 5, 26)
MIN_YEAR = 2020
MAX_YEAR = 2030
DATA_DIR = ROOT / "data"
QUERY_PATH = DATA_DIR / "reranker_eval_queries.json"
OUTPUT_PATH = DATA_DIR / "reranker_mvp_scored_top50.json"
SUMMARY_PATH = DATA_DIR / "reranker_mvp_summary.json"

ACTION_KEYWORDS = {
    "apply": [
        "신청", "지원", "접수", "모집", "참여", "선발", "장학생", "장학금",
        "근로장학생", "교환학생", "현장실습", "인턴십", "프로그램", "캠프",
    ],
    "schedule": ["일정", "기간", "언제", "마감", "납부기간", "정정기간", "발표일"],
    "result": ["합격자", "선정자", "결과", "발표", "선발 결과"],
    "guide": ["방법", "안내", "절차", "기준", "요건", "조건", "수수료", "비용", "발급"],
    "download": ["양식", "서류", "파일", "제출서류", "신청서"],
    "contact": ["문의", "전화", "연락처", "담당자", "위치"],
}

ACTION_PRIORITY = ["result", "schedule", "download", "contact", "apply", "guide"]

AUDIENCE_KEYWORDS = {
    "undergraduate": [
        "학부생", "재학생", "대학생", "본교생", "재학 중인 학생", "1학년", "2학년", "3학년", "4학년",
    ],
    "freshman": ["신입생", "새내기", "입학생"],
    "international": ["유학생", "외국인"],
    "graduate": ["대학원생", "석사", "박사", "일반대학원"],
    "faculty_staff": ["교직원", "교수", "직원", "전임교원", "조교"],
    "general_youth": ["청년", "만 19세", "만19세", "만 34세", "만34세"],
    "all": ["전체", "누구나", "제한 없음", "제한없음"],
}

ROLE_KEYWORDS = {
    "participant": ["단원", "참여자", "참가자", "교육생", "수강생", "장학생", "멘티", "봉사자"],
    "leader": ["단장", "팀장", "대표"],
    "mentor": ["멘토", "튜터", "서포터즈", "홍보대사"],
    "staff": ["운영진", "운영요원", "관리자", "인솔자", "조교", "사감"],
    "applicant": ["신청자", "지원자", "입사생", "선발자"],
}

FAMILY_HINTS = [
    "해외봉사", "월드프렌즈", "WFK", "KOICA", "청년봉사단", "봉사단",
    "국가장학금", "근로장학", "근로장학생", "장학금", "장학생",
    "기숙사", "생활관", "상상빌리지", "우촌학사",
    "수강신청", "수강정정", "계절학기", "강의평가",
    "등록금", "분할납부", "휴학", "복학", "졸업", "조기졸업",
    "복수전공", "부전공", "전과", "교환학생", "어학연수",
    "현장실습", "인턴십", "취업", "특강", "상담", "증명서",
]

DATE_PATTERNS = [
    re.compile(r"(?P<y>20\d{2})[.\-/년]\s*(?P<m>\d{1,2})[.\-/월]\s*(?P<d>\d{1,2})"),
    re.compile(r"(?P<y>20\d{2})\.\s*(?P<m>\d{1,2})\.\s*(?P<d>\d{1,2})\.?"),
]


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def parse_notice_date(value: Any) -> date | None:
    text = str(value or "")
    match = re.search(r"(20\d{2})\D+(\d{1,2})\D+(\d{1,2})", text)
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def years_in_text(text: str) -> list[int]:
    return sorted({int(y) for y in re.findall(r"20\d{2}", text) if MIN_YEAR <= int(y) <= MAX_YEAR})


def infer_action(text: str) -> str:
    found = {
        action: sum(1 for keyword in keywords if keyword in text)
        for action, keywords in ACTION_KEYWORDS.items()
    }
    best = max(ACTION_PRIORITY, key=lambda action: (found.get(action, 0), -ACTION_PRIORITY.index(action)))
    if found.get(best, 0) > 0:
        return best
    return "unknown"


def infer_audiences(text: str, default_undergraduate: bool = False) -> list[str]:
    audiences = [
        label
        for label, keywords in AUDIENCE_KEYWORDS.items()
        if any(keyword in text for keyword in keywords)
    ]
    if default_undergraduate and not audiences:
        audiences.append("undergraduate")
    if not audiences:
        audiences.append("unknown")
    return audiences


def infer_role(text: str) -> str:
    hits = [
        label
        for label, keywords in ROLE_KEYWORDS.items()
        if any(keyword in text for keyword in keywords)
    ]
    if "leader" in hits:
        return "leader"
    if "staff" in hits:
        return "staff"
    if "participant" in hits:
        return "participant"
    if "applicant" in hits:
        return "applicant"
    if "mentor" in hits:
        return "mentor"
    return "unknown"


def extract_dates(text: str) -> list[date]:
    dates: set[date] = set()
    for pattern in DATE_PATTERNS:
        for match in pattern.finditer(text):
            try:
                year = int(match.group("y"))
                if MIN_YEAR <= year <= MAX_YEAR:
                    dates.add(date(year, int(match.group("m")), int(match.group("d"))))
            except ValueError:
                continue
    return sorted(dates)


def extract_deadline(text: str, published_at: date | None) -> date | None:
    dates = extract_dates(text)
    if not dates:
        return None

    context_terms = ("신청", "접수", "모집", "제출", "납부", "등록", "마감", "기간")
    contextual: list[date] = []
    for pattern in DATE_PATTERNS:
        for match in pattern.finditer(text):
            window = text[max(0, match.start() - 40) : match.end() + 40]
            if any(term in window for term in context_terms):
                try:
                    year = int(match.group("y"))
                    if MIN_YEAR <= year <= MAX_YEAR:
                        contextual.append(date(year, int(match.group("m")), int(match.group("d"))))
                except ValueError:
                    pass

    candidates = contextual or dates
    if published_at:
        futureish = [dt for dt in candidates if dt >= published_at]
        if futureish:
            return max(futureish)
    return max(candidates)


def infer_time_intent(query: str) -> dict[str, Any]:
    years = years_in_text(query)
    if any(word in query for word in ("최근", "최신", "올해", "이번", "현재", "아직", "2026")):
        intent = "current_or_latest"
    elif any(word in query for word in ("작년", "지난", "2025", "2024")):
        intent = "specific_or_past"
    else:
        intent = "unspecified"
    return {"intent": intent, "years": years}


def extract_query_profile(query: str) -> dict[str, Any]:
    action = infer_action(query)
    if action == "unknown" and any(word in query for word in ("뭐", "어떤", "있어", "찾", "알려")):
        action = "apply"
    if action == "unknown" and any(
        word in query
        for word in (
            "해외봉사", "교환학생", "현장실습", "인턴십", "장학금", "근로장학생",
            "기숙사", "생활관", "특강", "취업 프로그램", "창업 지원", "상담",
        )
    ):
        action = "apply"
    return {
        "action": action,
        "audience": infer_audiences(query, default_undergraduate=True),
        "time_intent": infer_time_intent(query),
        "entities": extract_entities(query),
    }


def extract_entities(text: str) -> dict[str, Any]:
    upper_text = text.upper()
    matched = [hint for hint in FAMILY_HINTS if hint.upper() in upper_text]
    return {
        "years": years_in_text(text),
        "keywords": matched,
    }


def normalize_family(title: str, body: str) -> str:
    text = clean_title(title)
    upper_text = f"{text} {body[:300]}".upper()
    hints = [hint for hint in FAMILY_HINTS if hint.upper() in upper_text]
    if hints:
        return " / ".join(hints[:3])

    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"20\d{2}[\w\s.월년학년도-]*", " ", text)
    text = re.sub(r"\d+\s*(기|차|회)", " ", text)
    text = re.sub(r"(모집|안내|신청|접수|발표|결과|선발|공고|공지|새글)", " ", text)
    tokens = [tok for tok in tokenize_ko(text) if len(tok) >= 2]
    return " ".join(tokens[:4]) or clean_title(title)[:30]


def notice_features(notice: dict[str, Any]) -> dict[str, Any]:
    title = notice.get("title", "")
    body = notice.get("body", "") or notice.get("content", "")
    text = f"{title}\n{body}"
    published_at = parse_notice_date(notice.get("date"))
    deadline = extract_deadline(text, published_at)
    years = years_in_text(text)
    return {
        "action_type": infer_action(text),
        "audience": infer_audiences(text),
        "deadline": deadline.isoformat() if deadline else None,
        "year": max(years) if years else (published_at.year if published_at else None),
        "published_at": published_at.isoformat() if published_at else None,
        "program_family": normalize_family(title, body),
        "role_type": infer_role(text),
    }


def score_action(query_profile: dict[str, Any], features: dict[str, Any]) -> tuple[float, str]:
    query_action = query_profile["action"]
    doc_action = features["action_type"]
    if query_action == "unknown" or doc_action == "unknown":
        return 0.55, "행동 의도 또는 공지 행동 유형이 불명확함"
    if query_action == doc_action:
        return 1.0, f"query/action 모두 {query_action}"
    compatible = {
        ("apply", "guide"),
        ("guide", "apply"),
        ("schedule", "apply"),
        ("apply", "schedule"),
    }
    if (query_action, doc_action) in compatible:
        return 0.7, f"{query_action} 의도와 {doc_action} 공지는 보조적으로 관련"
    return 0.25, f"{query_action} 의도와 {doc_action} 공지가 다름"


def score_audience(query_profile: dict[str, Any], features: dict[str, Any]) -> tuple[float, str]:
    query_aud = set(query_profile["audience"])
    doc_aud = set(features["audience"])
    if "faculty_staff" in doc_aud and not query_aud.intersection({"faculty_staff"}):
        return 0.05, "교직원/교원 대상 공지로 학부생 기본 사용자와 불일치"
    if doc_aud.intersection(query_aud):
        return 1.0, f"대상자 일치: {sorted(doc_aud.intersection(query_aud))}"
    if doc_aud.intersection({"all", "general_youth"}):
        return 0.8, "전체/청년 대상이라 학부생과 대체로 부합"
    if "unknown" in doc_aud:
        return 0.55, "공지 대상자 불명확"
    return 0.25, f"대상자 불일치 가능성: query={sorted(query_aud)}, notice={sorted(doc_aud)}"


def score_validity(query_profile: dict[str, Any], features: dict[str, Any]) -> tuple[float, str]:
    action = query_profile["action"]
    deadline_text = features.get("deadline")
    published_text = features.get("published_at")
    deadline = date.fromisoformat(deadline_text) if deadline_text else None
    published = date.fromisoformat(published_text) if published_text else None

    if action in {"result", "guide", "contact"}:
        if published and (TODAY - published).days <= 365:
            return 0.9, "결과/안내형 쿼리라 마감보다 최신 게시일을 우선"
        return 0.65, "결과/안내형 쿼리라 과거 공지도 허용"

    if deadline:
        days = (deadline - TODAY).days
        if days >= 0:
            return 1.0, f"마감일이 아직 지나지 않음: D-{days}"
        if days >= -14:
            return 0.65, f"마감 직후: D{days}"
        return 0.2, f"마감일이 오래 지남: D{days}"

    if published:
        age = (TODAY - published).days
        if age <= 90:
            return 0.75, f"마감일은 없지만 최근 게시: {age}일 전"
        if age <= 365:
            return 0.55, f"마감일은 없고 게시 후 {age}일"
        return 0.35, f"마감일 불명, 오래된 게시: {age}일 전"

    return 0.5, "날짜 정보 부족"


def score_version(
    query_profile: dict[str, Any],
    features: dict[str, Any],
    family_stats: dict[str, dict[str, Any]],
) -> tuple[float, str]:
    family = features["program_family"]
    stats = family_stats.get(family, {})
    query_years = query_profile["time_intent"]["years"]
    doc_year = features.get("year")
    latest_year = stats.get("latest_year")
    query_action = query_profile["action"]
    doc_action = features["action_type"]
    role = features["role_type"]

    score = 0.55
    reasons = []

    if query_years:
        if doc_year in query_years:
            score += 0.25
            reasons.append("쿼리 연도와 공지 연도 일치")
        else:
            score -= 0.2
            reasons.append("쿼리 연도와 공지 연도 불일치")
    elif query_profile["time_intent"]["intent"] == "current_or_latest" and latest_year:
        if doc_year == latest_year:
            score += 0.2
            reasons.append("같은 family 내 최신 연도")
        else:
            score -= 0.1
            reasons.append("같은 family 내 최신 연도 아님")

    if query_action != "unknown" and doc_action == query_action:
        score += 0.15
        reasons.append("같은 주제 내 행동 유형 적합")

    if query_action == "apply":
        if role in {"participant", "applicant"}:
            score += 0.15
            reasons.append("참여/지원자 버전")
        elif role in {"leader", "staff"} and not any(word in " ".join(query_profile["entities"]["keywords"]) for word in ("단장", "인솔")):
            score -= 0.25
            reasons.append("일반 참여 의도에서 운영/리더 역할 버전")

    score = max(0.0, min(1.0, score))
    if not reasons:
        reasons.append("같은 주제 내 버전 판단 근거 약함")
    return score, "; ".join(reasons)


def minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi):
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def build_hybrid_searcher():
    from rank_bm25 import BM25Okapi

    collection = get_vector_collection()
    all_data = collection.get(include=["documents", "metadatas"])
    ids = all_data["ids"]
    documents = all_data["documents"]
    metadatas = all_data["metadatas"]
    tokenized = [tokenize_ko(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized)
    model = get_embed_model()

    by_url: dict[str, dict[str, Any]] = {}
    for corpus_path in [DATA_DIR / "data_2025.json", DATA_DIR / "data_2026.json", DATA_DIR / "2026_notice.json"]:
        if not corpus_path.exists():
            continue
        for item in load_json(corpus_path):
            url = item.get("url")
            if url and url not in by_url:
                by_url[url] = item

    def search(query: str, top_k: int = 50, alpha: float = 0.5) -> list[dict[str, Any]]:
        q_emb = model.encode(query).tolist()
        n_results = min(top_k * 5, len(documents))
        vr = collection.query(
            query_embeddings=[q_emb],
            n_results=n_results,
            include=["metadatas", "distances"],
        )

        vector_scores: dict[str, float] = {}
        raw_dist = vr["distances"][0]
        if raw_dist:
            sims = [1 - dist for dist in raw_dist]
            normed = minmax(sims)
            vector_scores = {did: score for did, score in zip(vr["ids"][0], normed)}

        bm25_raw = bm25.get_scores(tokenize_ko(query))
        bm25_max = max(bm25_raw) if max(bm25_raw) > 0 else 1
        bm25_scores = {did: float(score / bm25_max) for did, score in zip(ids, bm25_raw)}

        meta_map = dict(zip(ids, metadatas))
        doc_map = dict(zip(ids, documents))
        latest_month = max(
            (
                parsed.year * 12 + parsed.month
                for meta in metadatas
                if (parsed := parse_notice_date(meta.get("date")))
            ),
            default=TODAY.year * 12 + TODAY.month,
        )

        rows = []
        for did in set(vector_scores) | set(bm25_scores):
            meta = meta_map.get(did) or {}
            published = parse_notice_date(meta.get("date"))
            if published:
                month_diff = max(0, latest_month - (published.year * 12 + published.month))
                recency = 1 / (1 + month_diff / 3)
            else:
                month_diff = 999
                recency = 0.0
            base = alpha * vector_scores.get(did, 0.0) + (1 - alpha) * bm25_scores.get(did, 0.0)
            final = base * (1 + 0.15 * recency)
            rows.append(
                {
                    "id": did,
                    "retrieval_score": round(final, 6),
                    "retrieval_base": round(base, 6),
                    "vector": round(vector_scores.get(did, 0.0), 6),
                    "bm25": round(bm25_scores.get(did, 0.0), 6),
                    "recency": round(recency, 6),
                    "month_diff": month_diff,
                    "metadata": meta,
                    "content": doc_map.get(did, ""),
                }
            )

        seen_urls = set()
        deduped = []
        for row in sorted(rows, key=lambda item: item["retrieval_score"], reverse=True):
            meta = row["metadata"]
            url = meta.get("url")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            full_notice = by_url.get(url, {})
            deduped.append(
                {
                    **row,
                    "title": full_notice.get("title") or meta.get("title", ""),
                    "url": url,
                    "date": full_notice.get("date") or meta.get("date", ""),
                    "category": full_notice.get("category") or meta.get("category", ""),
                    "body": full_notice.get("body", ""),
                }
            )
            if len(deduped) >= top_k:
                break
        return deduped

    return search


def score_results(query_profile: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    family_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"latest_year": None, "count": 0})
    for candidate in candidates:
        features = candidate["features"]
        family = features["program_family"]
        family_stats[family]["count"] += 1
        year = features.get("year")
        if year:
            family_stats[family]["latest_year"] = max(family_stats[family]["latest_year"] or year, year)

    retrieval_values = [candidate["retrieval_score"] for candidate in candidates]
    retrieval_norms = minmax(retrieval_values)

    scored = []
    for candidate, retrieval_norm in zip(candidates, retrieval_norms):
        features = candidate["features"]
        action_score, action_reason = score_action(query_profile, features)
        audience_score, audience_reason = score_audience(query_profile, features)
        validity_score, validity_reason = score_validity(query_profile, features)
        version_score, version_reason = score_version(query_profile, features, family_stats)
        rerank_score = (
            0.45 * retrieval_norm
            + 0.18 * action_score
            + 0.14 * audience_score
            + 0.13 * validity_score
            + 0.10 * version_score
        )
        scored.append(
            {
                **candidate,
                "retrieval_norm": round(retrieval_norm, 6),
                "reranker_scores": {
                    "action_match": round(action_score, 4),
                    "audience_match": round(audience_score, 4),
                    "validity_match": round(validity_score, 4),
                    "version_match": round(version_score, 4),
                    "rerank_score": round(rerank_score, 6),
                },
                "reranker_reasons": {
                    "action_match": action_reason,
                    "audience_match": audience_reason,
                    "validity_match": validity_reason,
                    "version_match": version_reason,
                },
                "family_stats": {
                    "candidate_count_in_top50": family_stats[features["program_family"]]["count"],
                    "latest_year_in_top50_family": family_stats[features["program_family"]]["latest_year"],
                },
            }
        )

    for rank, item in enumerate(sorted(scored, key=lambda row: row["reranker_scores"]["rerank_score"], reverse=True), 1):
        item["rerank_rank"] = rank
    return scored


def main() -> None:
    queries = load_json(QUERY_PATH)
    search = build_hybrid_searcher()

    output = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "today_for_validity": TODAY.isoformat(),
        "source_queries": str(QUERY_PATH.relative_to(ROOT)),
        "search": {
            "candidate_count": 50,
            "alpha_vector_weight": 0.5,
            "vector_db": "pinecone",
            "pinecone_index": PINECONE_INDEX_NAME,
            "pinecone_namespace": PINECONE_NAMESPACE,
        },
        "scoring_weights": {
            "retrieval_norm": 0.45,
            "action_match": 0.18,
            "audience_match": 0.14,
            "validity_match": 0.13,
            "version_match": 0.10,
        },
        "queries": [],
    }

    summary_rows = []
    for query_row in queries:
        query = query_row["query"]
        query_profile = extract_query_profile(query)
        candidates = []
        for rank, row in enumerate(search(query, top_k=50), 1):
            feature_source = {
                "title": row["title"],
                "body": row.get("body") or row.get("content", ""),
                "date": row.get("date"),
            }
            candidates.append(
                {
                    "retrieval_rank": rank,
                    "retrieval_score": row["retrieval_score"],
                    "retrieval_base": row["retrieval_base"],
                    "vector": row["vector"],
                    "bm25": row["bm25"],
                    "recency": row["recency"],
                    "month_diff": row["month_diff"],
                    "title": row["title"],
                    "url": row["url"],
                    "date": row["date"],
                    "category": row["category"],
                    "snippet": (row.get("body") or row.get("content") or "")[:220],
                    "features": notice_features(feature_source),
                }
            )

        scored = score_results(query_profile, candidates)
        output["queries"].append(
            {
                "id": query_row["id"],
                "query": query,
                "query_profile": query_profile,
                "candidates": sorted(scored, key=lambda row: row["retrieval_rank"]),
            }
        )
        top_retrieval = min(scored, key=lambda row: row["retrieval_rank"]) if scored else {}
        top_reranked = min(scored, key=lambda row: row["rerank_rank"]) if scored else {}
        summary_rows.append(
            {
                "id": query_row["id"],
                "query": query,
                "query_profile": query_profile,
                "top_retrieval": {
                    "title": top_retrieval.get("title", ""),
                    "date": top_retrieval.get("date", ""),
                    "score": top_retrieval.get("retrieval_score"),
                },
                "top_reranked": {
                    "title": top_reranked.get("title", ""),
                    "date": top_reranked.get("date", ""),
                    "score": (top_reranked.get("reranker_scores") or {}).get("rerank_score"),
                    "moved_from_rank": top_reranked.get("retrieval_rank"),
                },
            }
        )

    dump_json(OUTPUT_PATH, output)
    dump_json(
        SUMMARY_PATH,
        {
            "generated_at": output["generated_at"],
            "source": str(OUTPUT_PATH.relative_to(ROOT)),
            "query_count": len(output["queries"]),
            "summary": summary_rows,
        },
    )
    print(f"wrote {OUTPUT_PATH}")
    print(f"wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
