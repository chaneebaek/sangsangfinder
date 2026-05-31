#!/usr/bin/env python3
"""Generate a robust reranker evaluation dataset from Supabase notices.

This script samples representative notices from the Supabase `notices` table,
uses Gemini to create realistic search-query variants for each sampled notice,
and attaches heuristic hard-negative qrels from nearby/confusable notices.

It intentionally does not evaluate search quality. Its output is an offline
dataset that can be consumed by a separate retrieval/reranker evaluation script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from api.core.utils import clean_title, tokenize_ko  # noqa: E402
from crawling.supabase_store import _connect  # noqa: E402


DATA_DIR = ROOT / "data"
DEFAULT_OUTPUT_PATH = DATA_DIR / "reranker_eval_dataset_gemini.json"
DEFAULT_CACHE_PATH = DATA_DIR / "reranker_eval_dataset_gemini_cache.json"

MIN_BODY_CHARS = 80
MAX_PROMPT_BODY_CHARS = 1800
DEFAULT_SAMPLE_SIZE = 150
DEFAULT_QUERIES_PER_NOTICE = 5
DEFAULT_HARD_NEGATIVES = 6
DEFAULT_MAX_RETRIES = 3

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

FAMILY_HINTS = [
    "해외봉사", "월드프렌즈", "WFK", "KOICA", "청년봉사단", "봉사단",
    "국가장학금", "근로장학", "근로장학생", "장학금", "장학생",
    "기숙사", "생활관", "상상빌리지", "우촌학사",
    "수강신청", "수강정정", "계절학기", "강의평가",
    "등록금", "분할납부", "휴학", "복학", "졸업", "조기졸업",
    "복수전공", "부전공", "전과", "교환학생", "어학연수",
    "현장실습", "인턴십", "취업", "특강", "상담", "증명서",
]


@dataclass(frozen=True)
class Notice:
    id: int
    notice_id: str | None
    title: str
    url: str
    posted_at: str | None
    category: str
    body: str
    category_type: list[str]
    job_types: list[str]

    @property
    def year(self) -> int | None:
        if not self.posted_at:
            return None
        match = re.search(r"(20\d{2})", self.posted_at)
        return int(match.group(1)) if match else None

    @property
    def text_for_matching(self) -> str:
        return f"{self.title}\n{self.body[:1200]}"


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def infer_action(text: str) -> str:
    counts = {
        action: sum(1 for keyword in keywords if keyword in text)
        for action, keywords in ACTION_KEYWORDS.items()
    }
    action, count = max(counts.items(), key=lambda item: item[1])
    return action if count > 0 else "unknown"


def infer_family(title: str, body: str) -> str:
    text = f"{clean_title(title)} {body[:400]}".upper()
    hints = [hint for hint in FAMILY_HINTS if hint.upper() in text]
    if hints:
        return " / ".join(hints[:3])

    title_text = re.sub(r"\[[^\]]+\]", " ", clean_title(title))
    title_text = re.sub(r"20\d{2}[\w\s.월년학년도-]*", " ", title_text)
    title_text = re.sub(r"\d+\s*(기|차|회)", " ", title_text)
    title_text = re.sub(r"(모집|안내|신청|접수|발표|결과|선발|공고|공지|새글)", " ", title_text)
    tokens = [token for token in tokenize_ko(title_text) if len(token) >= 2]
    return " ".join(tokens[:4]) or clean_title(title)[:30]


def notice_signature(notice: Notice) -> str:
    raw = f"{notice.id}|{notice.url}|{notice.title}|{notice.posted_at}|{notice.body[:500]}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def load_supabase_notices() -> list[Notice]:
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select
                    id,
                    notice_id,
                    title,
                    url,
                    posted_at::text,
                    coalesce(category, ''),
                    coalesce(body, ''),
                    coalesce(category_type, '{}'),
                    coalesce(job_types, '{}')
                from notices
                where source = 'hansung'
                  and title is not null
                  and url is not null
                  and length(coalesce(body, '')) >= %s
                order by posted_at desc nulls last, id desc
                """,
                (MIN_BODY_CHARS,),
            )
            rows = cur.fetchall()

    notices = []
    for row in rows:
        notices.append(
            Notice(
                id=row[0],
                notice_id=row[1],
                title=normalize_space(row[2]),
                url=row[3],
                posted_at=row[4],
                category=row[5] or "기타",
                body=normalize_space(row[6]),
                category_type=list(row[7] or []),
                job_types=list(row[8] or []),
            )
        )
    return notices


def stratify(notice: Notice) -> tuple[str, str, str, str]:
    year = str(notice.year or "unknown")
    action = infer_action(notice.text_for_matching)
    family = infer_family(notice.title, notice.body)
    return (notice.category or "기타", year, action, family)


def sample_representative_notices(
    notices: list[Notice],
    sample_size: int,
    seed: int,
) -> list[Notice]:
    rng = random.Random(seed)
    strata: dict[tuple[str, str, str, str], list[Notice]] = defaultdict(list)
    for notice in notices:
        strata[stratify(notice)].append(notice)

    for bucket in strata.values():
        rng.shuffle(bucket)

    selected: list[Notice] = []
    seen_urls: set[str] = set()

    # First pass: take one from as many category/year/action/family buckets as possible.
    for key in sorted(strata, key=lambda item: (item[0], item[1], item[2], item[3])):
        if len(selected) >= sample_size:
            break
        for notice in strata[key]:
            if notice.url not in seen_urls:
                selected.append(notice)
                seen_urls.add(notice.url)
                break

    # Second pass: fill remaining slots while preserving deterministic randomness.
    remaining = [notice for notice in notices if notice.url not in seen_urls]
    rng.shuffle(remaining)
    for notice in remaining:
        if len(selected) >= sample_size:
            break
        selected.append(notice)
        seen_urls.add(notice.url)

    return selected


def token_overlap(a: str, b: str) -> float:
    left = {token for token in tokenize_ko(a) if len(token) >= 2}
    right = {token for token in tokenize_ko(b) if len(token) >= 2}
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def find_hard_negatives(
    target: Notice,
    all_notices: list[Notice],
    limit: int,
) -> list[dict[str, Any]]:
    target_action = infer_action(target.text_for_matching)
    target_family = infer_family(target.title, target.body)
    scored = []

    for other in all_notices:
        if other.url == target.url:
            continue

        other_action = infer_action(other.text_for_matching)
        other_family = infer_family(other.title, other.body)
        same_category = other.category == target.category
        same_family = other_family == target_family
        same_year = other.year is not None and other.year == target.year
        overlap = token_overlap(target.title, other.title)

        score = overlap
        score += 0.35 if same_category else 0.0
        score += 0.45 if same_family else 0.0
        score += 0.15 if same_year else 0.0
        score += 0.25 if other_action != target_action and other_action != "unknown" else 0.0

        if score <= 0:
            continue

        reason_bits = []
        if same_family:
            reason_bits.append("same_family")
        if same_category:
            reason_bits.append("same_category")
        if same_year:
            reason_bits.append("same_year")
        if other_action != target_action:
            reason_bits.append(f"action_mismatch:{target_action}->{other_action}")
        if overlap:
            reason_bits.append(f"title_overlap:{overlap:.2f}")

        scored.append(
            {
                "notice_id": other.id,
                "url": other.url,
                "title": other.title,
                "category": other.category,
                "posted_at": other.posted_at,
                "relevance": 1 if same_family or same_category else 0,
                "reason": ", ".join(reason_bits) or "lexically similar",
                "_score": score,
            }
        )

    scored.sort(key=lambda row: row["_score"], reverse=True)
    negatives = []
    seen_urls = set()
    for row in scored:
        if row["url"] in seen_urls:
            continue
        seen_urls.add(row["url"])
        row.pop("_score", None)
        negatives.append(row)
        if len(negatives) >= limit:
            break
    return negatives


def load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def save_cache(path: Path, cache: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp_path.replace(path)


def parse_json_response(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def build_generation_prompt(notice: Notice, queries_per_notice: int) -> str:
    return f"""
너는 한성대학교 공지 검색 시스템의 평가셋을 만드는 검색 품질 엔지니어다.
아래 공지를 정확히 찾을 수 있는 한국어 검색 질의 {queries_per_notice}개를 만들어라.

요구사항:
- 실제 학생이 검색창에 입력할 법한 질의로 작성한다.
- 서로 다른 유형을 섞는다: 짧은 키워드형, 자연어형, 조건 포함형, 약어/영문 포함형, 일정/마감/대상자 질문형.
- 정답 공지를 과하게 그대로 복사하지 말고, 그래도 이 공지가 1순위로 나와야 한다.
- 다른 유사 공지와 헷갈릴 수 있는 표현도 일부 포함한다.
- 각 질의는 4~45자 정도로 작성한다.
- JSON만 출력한다. 설명 문장은 쓰지 않는다.

출력 형식:
{{
  "queries": [
    {{
      "query": "검색 질의",
      "query_type": "keyword|natural|condition|abbreviation|schedule|audience",
      "intent": "사용자 의도 한 문장",
      "must_match": ["랭킹에서 중요하게 봐야 할 조건"]
    }}
  ]
}}

[공지]
제목: {notice.title}
카테고리: {notice.category}
게시일: {notice.posted_at or ""}
본문: {notice.body[:MAX_PROMPT_BODY_CHARS]}
""".strip()


def generate_queries_with_gemini(
    notice: Notice,
    queries_per_notice: int,
    model_name: str,
    cache: dict[str, Any],
    max_retries: int,
) -> list[dict[str, Any]]:
    cache_key = f"{notice.id}:{notice_signature(notice)}:{queries_per_notice}:{model_name}"
    if cache_key in cache:
        return cache[cache_key]["queries"]

    api_key = os.getenv("GEMINI_API_KEY_FREE_TIER")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY_FREE_TIER is required in .env")

    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    last_error: Exception | None = None
    parsed: dict[str, Any] | None = None
    prompt = build_generation_prompt(notice, queries_per_notice)
    for attempt in range(1, max_retries + 1):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.45 if attempt == 1 else 0.25,
                    top_p=0.9,
                    max_output_tokens=2400,
                    response_mime_type="application/json",
                ),
            )
            parsed = parse_json_response(response.text)
            break
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                raise
            time.sleep(min(2 * attempt, 8))

    if parsed is None:
        raise RuntimeError(f"Gemini response parse failed for notice id={notice.id}") from last_error

    queries = parsed.get("queries", [])
    if not isinstance(queries, list) or not queries:
        raise ValueError(f"Gemini returned no queries for notice id={notice.id}")

    cleaned = []
    seen = set()
    for item in queries:
        if not isinstance(item, dict):
            continue
        query = normalize_space(item.get("query"))
        if not query or query in seen:
            continue
        seen.add(query)
        cleaned.append(
            {
                "query": query,
                "query_type": normalize_space(item.get("query_type")) or "unknown",
                "intent": normalize_space(item.get("intent")),
                "must_match": [
                    normalize_space(value)
                    for value in item.get("must_match", [])
                    if normalize_space(value)
                ],
            }
        )
        if len(cleaned) >= queries_per_notice:
            break

    if not cleaned:
        raise ValueError(f"Gemini queries were invalid for notice id={notice.id}")

    cache[cache_key] = {
        "notice_id": notice.id,
        "notice_url": notice.url,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "queries": cleaned,
    }
    return cleaned


def build_dataset(
    notices: list[Notice],
    sampled: list[Notice],
    queries_per_notice: int,
    hard_negative_count: int,
    model_name: str,
    cache_path: Path,
    sleep_seconds: float,
    save_every: int,
    max_retries: int,
) -> dict[str, Any]:
    cache = load_cache(cache_path)
    query_rows = []

    for index, notice in enumerate(sampled, start=1):
        generated_queries = generate_queries_with_gemini(
            notice=notice,
            queries_per_notice=queries_per_notice,
            model_name=model_name,
            cache=cache,
            max_retries=max_retries,
        )
        hard_negatives = find_hard_negatives(notice, notices, hard_negative_count)

        for query_index, query_item in enumerate(generated_queries, start=1):
            query_id = f"n{notice.id:06d}_q{query_index:02d}"
            qrels = [
                {
                    "notice_id": notice.id,
                    "url": notice.url,
                    "title": notice.title,
                    "relevance": 3,
                    "reason": "target_notice",
                }
            ]
            qrels.extend(hard_negatives)
            query_rows.append(
                {
                    "id": query_id,
                    "query": query_item["query"],
                    "query_type": query_item["query_type"],
                    "intent": query_item["intent"],
                    "must_match": query_item["must_match"],
                    "positive_notice": {
                        "notice_id": notice.id,
                        "source_notice_id": notice.notice_id,
                        "url": notice.url,
                        "title": notice.title,
                        "category": notice.category,
                        "posted_at": notice.posted_at,
                        "action": infer_action(notice.text_for_matching),
                        "family": infer_family(notice.title, notice.body),
                    },
                    "qrels": qrels,
                }
            )

        if index % save_every == 0:
            save_cache(cache_path, cache)

        if sleep_seconds > 0 and index < len(sampled):
            time.sleep(sleep_seconds)

    save_cache(cache_path, cache)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": {
            "table": "notices",
            "sampled_notice_count": len(sampled),
            "loaded_notice_count": len(notices),
        },
        "generation": {
            "model": model_name,
            "queries_per_notice": queries_per_notice,
            "hard_negatives_per_query": hard_negative_count,
            "api_key_env": "GEMINI_API_KEY_FREE_TIER",
        },
        "labels": {
            "3": "exact target notice",
            "1": "hard negative: related/confusable but not the exact target",
            "0": "hard negative: weakly related lexical/category confounder",
        },
        "queries": query_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Gemini-based reranker evaluation queries from Supabase notices.",
    )
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--queries-per-notice", type=int, default=DEFAULT_QUERIES_PER_NOTICE)
    parser.add_argument("--hard-negatives", type=int, default=DEFAULT_HARD_NEGATIVES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--sleep-seconds", type=float, default=4.0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    args = parser.parse_args()

    if args.sample_size < 1:
        raise ValueError("--sample-size must be at least 1")
    if args.queries_per_notice < 1:
        raise ValueError("--queries-per-notice must be at least 1")
    if args.hard_negatives < 0:
        raise ValueError("--hard-negatives must be non-negative")

    notices = load_supabase_notices()
    if not notices:
        raise SystemExit("No Supabase notices loaded. Check SUPABASE_DB_URL and notices table contents.")

    sampled = sample_representative_notices(
        notices=notices,
        sample_size=min(args.sample_size, len(notices)),
        seed=args.seed,
    )
    dataset = build_dataset(
        notices=notices,
        sampled=sampled,
        queries_per_notice=args.queries_per_notice,
        hard_negative_count=args.hard_negatives,
        model_name=args.model,
        cache_path=args.cache,
        sleep_seconds=args.sleep_seconds,
        save_every=max(1, args.save_every),
        max_retries=max(1, args.max_retries),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Loaded notices: {len(notices)}")
    print(f"Sampled notices: {len(sampled)}")
    print(f"Generated queries: {len(dataset['queries'])}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
