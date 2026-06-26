#!/usr/bin/env python3
"""Reclassify stored Supabase notices with Gemini."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, datetime
from typing import Any

import google.generativeai as genai
import psycopg
from psycopg.types.json import Jsonb

SOURCE = "hansung"

VALID_CATEGORIES = [
    "취업/채용", "학사행정", "학생활동/비교과",
    "대외활동", "공모전/경진대회", "국제교류", "창업",
    "장학금", "기숙사", "ROTC",
]

CATEGORY_TYPE_MAP = {
    "취업/채용": ["경영/금융/사무", "IT/정보통신", "디자인/예술/방송", "공학/기술", "교육/법률/공공", "교내채용"],
    "학생활동/비교과": ["IT/AI/SW", "진로/취업/현장실습", "디자인/콘텐츠", "심리/상담/자기계발", "인문/어학", "행사/생활"],
    "공모전/경진대회": ["IT/AI/SW", "창업/아이디어", "디자인/콘텐츠", "글쓰기/발표/어학", "정책/사회/ESG"],
    "창업": ["창업"],
    "국제교류": ["교환학생/파견", "해외인턴/현장실습", "해외연수", "외국인학생/글로벌교류", "해외봉사"],
    "학사행정": ["수업/수강", "전공/트랙/학적", "성적/졸업", "시험/학사일정", "시설/시스템/행정"],
    "장학금": ["장학금"],
    "대외활동": ["봉사활동", "멘토링", "서포터즈/홍보대사", "기획/미디어"],
    "ROTC": ["ROTC"],
    "기숙사": ["기숙사"],
}


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _db_url() -> str:
    value = os.getenv("SUPABASE_DB_URL") or os.getenv("DATABASE_URL")
    if not value:
        raise RuntimeError("SUPABASE_DB_URL or DATABASE_URL is required.")
    return value


def _clean(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, list):
        return [_clean(item) for item in value]
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    return value


def classify_with_gemini(model: genai.GenerativeModel, title: str, body: str, max_retry: int) -> dict[str, Any]:
    prompt = f"""
한성대학교 공지사항을 아래 카테고리 중 하나로 분류하고, 해당 카테고리의 세부 유형도 분류해.
반드시 JSON 형식으로만 답해.

카테고리 목록:
{json.dumps(VALID_CATEGORIES, ensure_ascii=False)}

카테고리별 세부 유형:
{json.dumps(CATEGORY_TYPE_MAP, ensure_ascii=False)}

공지 제목: {title}
공지 본문: {(body or '')[:500]}

아래 JSON 형식으로만 출력해:
{{"category": "카테고리명", "category_type": ["세부유형1", "세부유형2"]}}

규칙:
- category는 반드시 위 카테고리 목록 중 하나
- category_type은 해당 카테고리의 세부 유형 중 해당하는 것만
- 장학금/기숙사/ROTC/창업은 category_type이 하나
"""
    last_error: Exception | None = None
    for attempt in range(1, max_retry + 1):
        try:
            res = model.generate_content(prompt)
            result = json.loads(res.text)
            category = result.get("category")
            category_type = result.get("category_type") or []
            if category not in VALID_CATEGORIES:
                raise ValueError(f"invalid category: {category!r}")
            valid_types = CATEGORY_TYPE_MAP.get(category, [])
            category_type = [item for item in category_type if item in valid_types]
            return {"category": category, "category_type": category_type}
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"  Gemini failed ({attempt}/{max_retry}): {exc}", flush=True)
            time.sleep(2)

    raise RuntimeError("Gemini classification failed") from last_error


def load_candidates(
    conn: psycopg.Connection,
    start_date: date,
    end_date: date,
    only_fallback: bool,
    limit: int | None,
) -> list[dict[str, Any]]:
    params: list[Any] = [SOURCE, start_date, end_date]
    fallback_sql = ""
    if only_fallback:
        fallback_sql = """
          and (
            category is null
            or category = '기타'
            or coalesce(jsonb_array_length(category_type), 0) = 0
          )
        """

    limit_sql = ""
    if limit:
        limit_sql = "limit %s"
        params.append(limit)

    with conn.cursor() as cur:
        cur.execute(
            f"""
            select id, title, body, posted_at, category, category_type, raw
            from notices
            where source = %s
              and posted_at between %s and %s
              {fallback_sql}
            order by posted_at desc, id desc
            {limit_sql}
            """,
            params,
        )
        rows = cur.fetchall()

    return [
        {
            "id": row[0],
            "title": row[1] or "",
            "body": row[2] or "",
            "posted_at": row[3],
            "category": row[4],
            "category_type": row[5] or [],
            "raw": row[6] or {},
        }
        for row in rows
    ]


def update_notice(conn: psycopg.Connection, notice: dict[str, Any], result: dict[str, Any]) -> None:
    raw = dict(notice.get("raw") or {})
    raw["category"] = result["category"]
    raw["category_type"] = result["category_type"]
    raw["job_types"] = result["category_type"]

    with conn.cursor() as cur:
        cur.execute(
            """
            update notices
            set category = %s,
                category_type = %s,
                job_types = %s,
                raw = %s,
                updated_at = now()
            where id = %s
            """,
            (
                result["category"],
                Jsonb(_clean(result["category_type"])),
                Jsonb(_clean(result["category_type"])),
                Jsonb(_clean(raw)),
                notice["id"],
            ),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Reclassify Supabase notices with Gemini.")
    parser.add_argument("--start-date", required=True, type=_parse_date)
    parser.add_argument("--end-date", required=True, type=_parse_date)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--all", action="store_true", help="Reclassify all rows in range, not just fallback rows.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-retry", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=1.0)
    args = parser.parse_args()

    if args.end_date < args.start_date:
        raise ValueError("--end-date must be on or after --start-date")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash", generation_config={"response_mime_type": "application/json"})

    with psycopg.connect(_db_url(), autocommit=False, connect_timeout=10, prepare_threshold=None) as conn:
        notices = load_candidates(conn, args.start_date, args.end_date, not args.all, args.limit)
        print(f"Target notices: {len(notices)}", flush=True)

        updated = 0
        failed = 0
        for index, notice in enumerate(notices, start=1):
            print(f"[{index}/{len(notices)}] id={notice['id']} {notice['posted_at']} {notice['title'][:80]}", flush=True)
            try:
                result = classify_with_gemini(model, notice["title"], notice["body"], args.max_retry)
                print(f"  -> {result['category']} {result['category_type']}", flush=True)
                if not args.dry_run:
                    update_notice(conn, notice, result)
                    conn.commit()
                updated += 1
            except Exception as exc:  # noqa: BLE001
                conn.rollback()
                failed += 1
                print(f"  failed: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(args.sleep)

    print(f"Done. updated={updated}, failed={failed}, total={len(notices)}", flush=True)
    if failed:
        raise RuntimeError(f"Reclassification failed for {failed}/{len(notices)} notices")


if __name__ == "__main__":
    main()
