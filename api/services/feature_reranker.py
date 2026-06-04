"""Feature-based soft reranker for notice search results.

The reranker keeps the original hybrid relevance score as the backbone and adds
small boosts for user eligibility, recruitment status, deadline urgency, and
recency. LLM-extracted notice features are cached because the top-50 path can be
called repeatedly for the same notices.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from datetime import date, datetime
from typing import Any

from ..core.config import GEMINI_API_KEY_PAID_TIER, GEMINI_FEATURE_TIMEOUT_SECONDS, NOTICE_FEATURE_CACHE_PATH

AUDIENCES = {"학부생", "교직원", "졸업생", "일반인", "기타"}
DEFAULT_FEATURES: dict[str, Any] = {
    "extraction": {
        "method": "unknown",
        "llm_failed": False,
    },
    "target_audiences": ["기타"],
    "undergraduate": {
        "grades": [],
        "departments": [],
        "low_income_required": None,
    },
    "start_date": None,
    "end_date": None,
    "confidence": {
        "target_audiences": 0.0,
        "undergraduate": 0.0,
        "start_date": 0.0,
        "end_date": 0.0,
    },
}


def rerank_notices(
    notices: list[dict],
    profile: dict[str, Any] | None = None,
    top_k: int = 20,
    today: date | None = None,
) -> list[dict]:
    """Return the top-k notices after feature soft boosts."""
    if not notices:
        return []

    today = today or date.today()
    profile = profile or {}
    features_by_key = extract_notice_features(notices)
    max_recency_date = _max_posted_date(notices)

    reranked: list[dict] = []
    for idx, notice in enumerate(notices):
        key = _notice_key(notice)
        features = features_by_key.get(key, copy.deepcopy(DEFAULT_FEATURES))
        base_score = _safe_float(notice.get("score"), default=0.0)
        boost_parts = _score_boosts(notice, features, profile, today, max_recency_date)
        signals = _score_signals(notice, features, today, max_recency_date)
        feature_boost = sum(boost_parts.values())
        rerank_score = base_score * (1.0 + feature_boost)

        item = dict(notice)
        item["base_score"] = round(base_score, 6)
        item["score"] = round(rerank_score, 6)
        item["feature_reranker"] = {
            "features": features,
            "signals": signals,
            "boosts": {k: round(v, 6) for k, v in boost_parts.items()},
            "feature_boost": round(feature_boost, 6),
            "rerank_score": round(rerank_score, 6),
            "original_rank": idx + 1,
        }
        reranked.append(item)

    reranked.sort(key=lambda x: (x.get("score", 0), -x["feature_reranker"]["original_rank"]), reverse=True)
    for rank, item in enumerate(reranked, start=1):
        item["feature_reranker"]["rerank_rank"] = rank
    return reranked[:top_k]


def extract_notice_features(notices: list[dict]) -> dict[str, dict[str, Any]]:
    cache = _load_cache()
    result: dict[str, dict[str, Any]] = {}
    missing: list[dict] = []

    for notice in notices:
        key = _notice_key(notice)
        content_hash = _content_hash(notice)
        cached = cache.get(key)
        if cached and cached.get("content_hash") == content_hash:
            result[key] = _normalize_features(cached.get("features"))
        else:
            missing.append(notice)

    if missing:
        extracted = _extract_with_gemini(missing)
        changed = False
        for notice in missing:
            key = _notice_key(notice)
            features = extracted.get(key) or _fallback_extract(notice)
            features = _normalize_features(features)
            result[key] = features
            cache[key] = {
                "content_hash": _content_hash(notice),
                "features": features,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            changed = True
        if changed:
            _save_cache(cache)

    return result


def _extract_with_gemini(notices: list[dict]) -> dict[str, dict[str, Any]]:
    if not GEMINI_API_KEY_PAID_TIER:
        return {_notice_key(notice): _fallback_extract(notice, llm_failed=True) for notice in notices}

    try:
        import google.generativeai as genai

        genai.configure(api_key=GEMINI_API_KEY_PAID_TIER)
        model = genai.GenerativeModel("gemini-2.5-flash")
    except Exception as exc:
        print(f"[feature-reranker] Gemini 로드 실패, fallback 사용: {exc}", flush=True)
        return {_notice_key(notice): _fallback_extract(notice, llm_failed=True) for notice in notices}

    extracted: dict[str, dict[str, Any]] = {}
    for batch in _chunks(notices, 8):
        prompt = _build_extraction_prompt(batch)
        try:
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0,
                    "response_mime_type": "application/json",
                },
                request_options={"timeout": GEMINI_FEATURE_TIMEOUT_SECONDS},
            )
            rows = _parse_json_array(response.text)
        except Exception as exc:
            print(f"[feature-reranker] Gemini 추출 실패, fallback 사용: {exc}", flush=True)
            rows = []

        row_map = {str(row.get("id")): row for row in rows if isinstance(row, dict)}
        for notice in batch:
            key = _notice_key(notice)
            row = row_map.get(key)
            extracted[key] = _mark_llm_extract(row) if row else _fallback_extract(notice, llm_failed=True)

    return extracted


def _build_extraction_prompt(notices: list[dict]) -> str:
    items = []
    for notice in notices:
        items.append({
            "id": _notice_key(notice),
            "title": notice.get("title", ""),
            "posted_date": notice.get("posted_at") or notice.get("date") or "",
            "category": notice.get("category", ""),
            "body_excerpt": _notice_text(notice)[:1800],
        })

    return f"""
한성대학교 공지에서 feature reranker용 정보를 추출하세요.
반드시 JSON 배열만 반환하세요. 마크다운, 설명, 주석은 금지입니다.

분류 규칙:
- target_audiences는 ["학부생","교직원","졸업생","일반인","기타"] 중 하나 이상입니다.
- 여러 대상이면 모두 넣습니다.
- 대상이 불명확하면 ["기타"]입니다.
- 대학생, 재학생, 휴학생, 신입생, 편입생, 졸업예정자는 학부생으로 분류합니다.
- 기업/지역주민/외부 참가자 등 학교 구성원이 아니어도 참여 가능하면 일반인을 포함합니다.
- 교원, 직원, 조교, 교수는 교직원입니다.
- start_date/end_date는 신청, 모집, 접수 기간 기준입니다. ISO YYYY-MM-DD 또는 null입니다.
- 게시일이나 행사일을 신청 시작/마감일로 착각하지 마세요.
- confidence 값은 0.0~1.0입니다.

반환 스키마:
[
  {{
    "id": "입력 id",
    "target_audiences": ["학부생"],
    "undergraduate": {{
      "grades": [1,2,3,4],
      "departments": ["컴퓨터공학부"],
      "low_income_required": true
    }},
    "start_date": "2026-06-01",
    "end_date": "2026-06-10",
    "confidence": {{
      "target_audiences": 0.8,
      "undergraduate": 0.6,
      "start_date": 0.7,
      "end_date": 0.7
    }}
  }}
]

입력:
{json.dumps(items, ensure_ascii=False)}
""".strip()


def _score_boosts(
    notice: dict,
    features: dict[str, Any],
    profile: dict[str, Any],
    today: date,
    max_recency_date: date | None,
) -> dict[str, float]:
    audiences = set(features.get("target_audiences") or ["기타"])
    confidence = features.get("confidence") or {}
    audience_conf = _bounded(_safe_float(confidence.get("target_audiences"), 0.0), 0.0, 1.0)

    user_audience = _profile_audience(profile)
    audience_boost = 0.0
    if user_audience in audiences:
        audience_boost = 0.18 * audience_conf
    elif "기타" in audiences or not audiences:
        audience_boost = 0.02 * audience_conf
    elif user_audience == "학부생" and "졸업생" in audiences and _is_senior_or_graduating(profile):
        audience_boost = 0.06 * audience_conf

    eligibility_boost = _undergraduate_boost(features, profile)

    start_date = _parse_iso_date(features.get("start_date"))
    end_date = _parse_iso_date(features.get("end_date"))
    date_conf = max(
        _bounded(_safe_float(confidence.get("start_date"), 0.0), 0.0, 1.0),
        _bounded(_safe_float(confidence.get("end_date"), 0.0), 0.0, 1.0),
    )
    status_score = _application_status_score(start_date, end_date, today)
    status_boost = 0.12 * status_score * date_conf

    deadline_conf = _bounded(_safe_float(confidence.get("end_date"), 0.0), 0.0, 1.0)
    deadline_boost = _deadline_urgency_boost(end_date, today) * deadline_conf
    recency_boost = _recency_boost(_posted_date(notice), max_recency_date)

    # Keep this as a soft reranker: features can move results, not dominate them.
    return {
        "audience": audience_boost,
        "undergraduate_eligibility": eligibility_boost,
        "application_status": status_boost,
        "deadline_urgency": deadline_boost,
        "recency": recency_boost,
    }


def _score_signals(
    notice: dict,
    features: dict[str, Any],
    today: date,
    max_recency_date: date | None,
) -> dict[str, Any]:
    start_date = _parse_iso_date(features.get("start_date"))
    end_date = _parse_iso_date(features.get("end_date"))
    posted = _posted_date(notice)
    return {
        "days_after_deadline": (today - end_date).days if end_date else None,
        "application_status_score": _application_status_score(start_date, end_date, today),
        "recency_score": round(_recency_boost(posted, max_recency_date) / 0.06, 6)
        if posted and max_recency_date
        else 0.0,
    }


def _undergraduate_boost(features: dict[str, Any], profile: dict[str, Any]) -> float:
    if _profile_audience(profile) != "학부생":
        return 0.0

    info = features.get("undergraduate") or {}
    confidence = features.get("confidence") or {}
    conf = _bounded(_safe_float(confidence.get("undergraduate"), 0.0), 0.0, 1.0)
    boost = 0.0

    profile_grade = _grade_number(profile.get("grade"))
    target_grades = [g for g in info.get("grades") or [] if isinstance(g, int)]
    if profile_grade and target_grades:
        if profile_grade in target_grades:
            boost += 0.05 * conf

    departments = [str(v).strip() for v in info.get("departments") or [] if str(v).strip()]
    user_department = str(profile.get("track") or profile.get("department") or "").strip()
    if departments and user_department:
        joined = " ".join(departments)
        if user_department in joined or any(dep in user_department for dep in departments):
            boost += 0.05 * conf

    low_income_required = info.get("low_income_required")
    if low_income_required is True:
        income_level = str(profile.get("income_level") or "")
        has_low_income = bool(re.search(r"\b([1-3])\s*분위\b|기초|차상위|저소득", income_level))
        if has_low_income:
            boost += 0.04 * conf

    return boost


def _application_status_score(start_date: date | None, end_date: date | None, today: date) -> float:
    if end_date and today > end_date:
        return 0.0
    if start_date and today < start_date:
        return 0.5
    if start_date or end_date:
        return 1.0
    return 0.4


def _deadline_urgency_boost(end_date: date | None, today: date) -> float:
    if not end_date:
        return 0.0
    days_after_deadline = (today - end_date).days
    if days_after_deadline > 0:
        return 0.0
    days_until_deadline = -days_after_deadline
    if days_until_deadline <= 1:
        return 0.10
    if days_until_deadline <= 3:
        return 0.08
    if days_until_deadline <= 7:
        return 0.06
    if days_until_deadline <= 14:
        return 0.035
    if days_until_deadline <= 30:
        return 0.015
    return 0.0


def _recency_boost(posted: date | None, max_posted: date | None) -> float:
    if not posted or not max_posted:
        return 0.0
    days_old = max(0, (max_posted - posted).days)
    return 0.06 * math.exp(-days_old / 30)


def _fallback_extract(notice: dict, llm_failed: bool = False) -> dict[str, Any]:
    text = _notice_text(notice)
    audiences: set[str] = set()
    if re.search(r"학부생|재학생|휴학생|복학생|편입생|신입생|졸업예정|대학생|학생", text):
        audiences.add("학부생")
    if re.search(r"교직원|교원|직원|교수|조교", text):
        audiences.add("교직원")
    if re.search(r"졸업생|동문|기졸업", text):
        audiences.add("졸업생")
    if re.search(r"일반인|지역주민|누구나|시민|외부", text):
        audiences.add("일반인")
    if not audiences:
        audiences.add("기타")

    dates = _extract_dates(text)
    return {
        "extraction": {
            "method": "rule_fallback",
            "llm_failed": llm_failed,
        },
        "target_audiences": sorted(audiences),
        "undergraduate": {
            "grades": _extract_grades(text),
            "departments": [],
            "low_income_required": _extract_low_income(text),
        },
        "start_date": dates[0] if dates else None,
        "end_date": dates[-1] if len(dates) >= 2 else None,
        "confidence": {
            "target_audiences": 0.35,
            "undergraduate": 0.25,
            "start_date": 0.2 if dates else 0.0,
            "end_date": 0.2 if len(dates) >= 2 else 0.0,
        },
    }


def _mark_llm_extract(raw: dict[str, Any]) -> dict[str, Any]:
    row = dict(raw)
    row["extraction"] = {
        "method": "llm",
        "llm_failed": False,
    }
    return row


def _normalize_features(raw: Any) -> dict[str, Any]:
    base = copy.deepcopy(DEFAULT_FEATURES)
    if not isinstance(raw, dict):
        return base

    extraction = raw.get("extraction")
    if isinstance(extraction, dict):
        method = str(extraction.get("method") or "unknown")
        base["extraction"] = {
            "method": method,
            "llm_failed": bool(extraction.get("llm_failed")),
        }

    audiences = raw.get("target_audiences")
    if isinstance(audiences, str):
        audiences = [audiences]
    if isinstance(audiences, list):
        clean = [str(v).strip() for v in audiences if str(v).strip() in AUDIENCES]
        base["target_audiences"] = clean or ["기타"]

    ug = raw.get("undergraduate")
    if isinstance(ug, dict):
        grades = []
        for grade in ug.get("grades") or []:
            num = _grade_number(grade)
            if num and num not in grades:
                grades.append(num)
        base["undergraduate"]["grades"] = grades
        base["undergraduate"]["departments"] = [
            str(v).strip() for v in ug.get("departments") or [] if str(v).strip()
        ]
        low_income = ug.get("low_income_required")
        base["undergraduate"]["low_income_required"] = low_income if isinstance(low_income, bool) else None

    for field in ("start_date", "end_date"):
        parsed = _parse_iso_date(raw.get(field))
        base[field] = parsed.isoformat() if parsed else None

    conf = raw.get("confidence")
    if isinstance(conf, dict):
        for field in base["confidence"]:
            base["confidence"][field] = _bounded(_safe_float(conf.get(field), 0.0), 0.0, 1.0)

    return base


def _load_cache() -> dict[str, Any]:
    try:
        with open(NOTICE_FEATURE_CACHE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(NOTICE_FEATURE_CACHE_PATH), exist_ok=True)
    tmp_path = f"{NOTICE_FEATURE_CACHE_PATH}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, NOTICE_FEATURE_CACHE_PATH)


def _notice_key(notice: dict) -> str:
    return str(notice.get("url") or notice.get("notice_id") or notice.get("id") or notice.get("title") or "")


def _content_hash(notice: dict) -> str:
    payload = {
        "title": notice.get("title", ""),
        "body": notice.get("body", "") or notice.get("content", ""),
        "date": notice.get("posted_at") or notice.get("date") or "",
        "url": notice.get("url", ""),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _notice_text(notice: dict) -> str:
    return "\n".join([
        str(notice.get("title") or ""),
        str(notice.get("category") or ""),
        str(notice.get("body") or notice.get("content") or ""),
    ])


def _parse_json_array(text: str) -> list[Any]:
    try:
        value = json.loads(text)
        return value if isinstance(value, list) else []
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", text or "")
        if not match:
            return []
        value = json.loads(match.group(0))
        return value if isinstance(value, list) else []


def _chunks(values: list[Any], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _parse_iso_date(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    match = re.search(r"(20\d{2})[-./년\s]+(\d{1,2})[-./월\s]+(\d{1,2})", text)
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def _extract_dates(text: str) -> list[str]:
    dates: list[str] = []
    for match in re.finditer(r"(20\d{2})[-./년\s]+(\d{1,2})[-./월\s]+(\d{1,2})", text):
        try:
            value = date(int(match.group(1)), int(match.group(2)), int(match.group(3))).isoformat()
        except ValueError:
            continue
        if value not in dates:
            dates.append(value)
    return dates


def _extract_grades(text: str) -> list[int]:
    grades: list[int] = []
    if re.search(r"전\s*학년|전체\s*학년|1\s*-\s*4\s*학년", text):
        return [1, 2, 3, 4]
    for match in re.finditer(r"([1-4])\s*학년", text):
        grade = int(match.group(1))
        if grade not in grades:
            grades.append(grade)
    return grades


def _extract_low_income(text: str) -> bool | None:
    if re.search(r"저소득|소득\s*분위|기초생활|차상위|한국장학재단\s*소득", text):
        return True
    return None


def _profile_audience(profile: dict[str, Any]) -> str:
    raw = str(profile.get("audience") or profile.get("target_audience") or profile.get("user_type") or "")
    if raw in AUDIENCES:
        return raw
    if profile.get("grade") or profile.get("track") or profile.get("college"):
        return "학부생"
    return "기타"


def _is_senior_or_graduating(profile: dict[str, Any]) -> bool:
    grade = _grade_number(profile.get("grade"))
    raw = " ".join(str(profile.get(k) or "") for k in ("grade", "status", "interests"))
    return grade == 4 or "졸업예정" in raw


def _grade_number(value: Any) -> int | None:
    if isinstance(value, int) and 1 <= value <= 4:
        return value
    match = re.search(r"([1-4])", str(value or ""))
    return int(match.group(1)) if match else None


def _posted_date(notice: dict) -> date | None:
    return _parse_iso_date(notice.get("posted_at") or notice.get("date"))


def _max_posted_date(notices: list[dict]) -> date | None:
    dates = [_posted_date(notice) for notice in notices]
    dates = [d for d in dates if d]
    return max(dates) if dates else None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bounded(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
