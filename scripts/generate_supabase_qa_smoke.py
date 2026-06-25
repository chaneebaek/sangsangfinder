#!/usr/bin/env python3
"""Generate a small grounded QA smoke set from 2026 Supabase notices.

The pipeline intentionally uses two independent Gemini calls:
  1. generator agent creates candidate QA pairs
  2. validator agent independently checks grounding and schema quality

Ground truth is stored by notice URL so retrieval evaluation is robust to title
duplicates, title normalization, and chunked vector storage.
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
import uuid
from dataclasses import dataclass
from datetime import datetime
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

from crawling.supabase_store import SOURCE, _connect  # noqa: E402


DATA_DIR = ROOT / "data"
DEFAULT_OUTPUT = DATA_DIR / "qa_supabase_2026_smoke.json"
DEFAULT_MODEL = "gemini-2.5-flash"
QA_TYPES = ("factual", "procedural", "conditional", "user_short")
MIN_BODY_CHARS = 250
MAX_NOTICE_BODY_CHARS = 2600


@dataclass(frozen=True)
class Notice:
    id: int
    notice_id: str | None
    title: str
    url: str
    posted_at: str
    category: str
    body: str


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def parse_json_response(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def body_hash(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def load_2026_notices(limit: int, seed: int) -> list[Notice]:
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
                    coalesce(body, '')
                from notices
                where source = %s
                  and posted_at >= date '2026-01-01'
                  and posted_at < date '2027-01-01'
                  and title is not null
                  and url is not null
                  and length(coalesce(body, '')) >= %s
                order by posted_at desc nulls last, id desc
                """,
                (SOURCE, MIN_BODY_CHARS),
            )
            rows = cur.fetchall()

    notices = [
        Notice(
            id=row[0],
            notice_id=row[1],
            title=normalize_space(row[2]),
            url=row[3],
            posted_at=row[4],
            category=row[5] or "기타",
            body=normalize_space(row[6]),
        )
        for row in rows
    ]
    rng = random.Random(seed)
    rng.shuffle(notices)
    return notices[: max(limit, 1)]


def configure_gemini(api_key_env: str, model_name: str):
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is required in .env")

    import google.generativeai as genai

    genai.configure(api_key=api_key)
    return genai, genai.GenerativeModel(model_name)


def build_generation_prompt(notice: Notice, qa_type: str) -> str:
    return f"""
You are the QA Generator Agent. Create exactly one Korean QA pair grounded only
in the notice below.

Requirements:
- type must be exactly "{qa_type}".
- factual: ask for a concrete fact such as date, amount, target, location, contact, benefit.
- procedural: ask how to apply, submit, check, participate, or complete a process.
- conditional: ask whether something is possible under a condition, exception, eligibility, deadline, or status.
- user_short: use a short casual Korean user query, but keep the answer grounded and specific.
- The answer must be fully supported by the notice.
- source_span must quote or closely copy the shortest Korean evidence span from the notice.
- Do not invent facts. If the notice cannot support this type, return {{"qa": null, "reason": "unsupported"}}.

Return JSON only:
{{
  "qa": {{
    "question": "...",
    "answer": "...",
    "type": "{qa_type}",
    "source_span": "..."
  }},
  "reason": "why the notice supports this QA"
}}

[Notice]
title: {notice.title}
url: {notice.url}
posted_at: {notice.posted_at}
category: {notice.category}
body: {notice.body[:MAX_NOTICE_BODY_CHARS]}
""".strip()


def build_validation_prompt(notice: Notice, qa: dict[str, Any]) -> str:
    return f"""
You are the QA Validator Agent. You did not participate in generation.
Independently judge whether this QA is valid for retrieval evaluation.

Pass only if all conditions are true:
- The answer is fully grounded in the notice body/title.
- The source_span supports the answer.
- The question is answerable from this exact notice.
- The QA type is one of {list(QA_TYPES)} and matches the question style.
- The ground-truth URL should be this notice URL, not just a similar notice.

Return JSON only:
{{
  "verdict": "pass" or "reject",
  "score": 1-5,
  "grounded": true/false,
  "type_correct": true/false,
  "source_span_supported": true/false,
  "reason": "short Korean reason"
}}

[Notice]
title: {notice.title}
url: {notice.url}
posted_at: {notice.posted_at}
category: {notice.category}
body: {notice.body[:MAX_NOTICE_BODY_CHARS]}

[Candidate QA]
{json.dumps(qa, ensure_ascii=False, indent=2)}
""".strip()


def generate_candidate(
    notice: Notice,
    qa_type: str,
    generator_model,
    genai,
    max_retries: int,
) -> dict[str, Any] | None:
    prompt = build_generation_prompt(notice, qa_type)
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = generator_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.55 if attempt == 1 else 0.25,
                    top_p=0.9,
                    max_output_tokens=1800,
                    response_mime_type="application/json",
                ),
            )
            parsed = parse_json_response(response.text)
            qa = parsed.get("qa") if isinstance(parsed, dict) else None
            if qa is None:
                return None
            if not isinstance(qa, dict):
                raise ValueError("qa must be an object")
            qa["question"] = normalize_space(qa.get("question"))
            qa["answer"] = normalize_space(qa.get("answer"))
            qa["type"] = normalize_space(qa.get("type"))
            qa["source_span"] = normalize_space(qa.get("source_span"))
            if not qa["question"] or not qa["answer"] or qa["type"] not in QA_TYPES:
                raise ValueError("generated QA has invalid required fields")
            return qa
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < max_retries:
                time.sleep(min(2 * attempt, 6))
    print(f"[reject] generation failed for notice id={notice.id}: {last_error}", flush=True)
    return None


def validate_candidate(
    notice: Notice,
    qa: dict[str, Any],
    validator_model,
    genai,
    max_retries: int,
) -> dict[str, Any]:
    prompt = build_validation_prompt(notice, qa)
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = validator_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    top_p=0.8,
                    max_output_tokens=1200,
                    response_mime_type="application/json",
                ),
            )
            parsed = parse_json_response(response.text)
            if not isinstance(parsed, dict):
                raise ValueError("validator response must be an object")
            return parsed
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < max_retries:
                time.sleep(min(2 * attempt, 6))
    return {
        "verdict": "reject",
        "score": 1,
        "grounded": False,
        "type_correct": False,
        "source_span_supported": False,
        "reason": f"validation_failed: {last_error}",
    }


def build_row(
    notice: Notice,
    qa: dict[str, Any],
    validation: dict[str, Any],
    generator_session_id: str,
    validator_session_id: str,
    model_name: str,
) -> dict[str, Any]:
    return {
        "id": f"supabase2026_n{notice.id}_qa{uuid.uuid4().hex[:8]}",
        "question": qa["question"],
        "answer": qa["answer"],
        "type": qa["type"],
        "source_span": qa["source_span"],
        "gt_url": notice.url,
        "notice_url": notice.url,
        "notice_db_id": notice.id,
        "notice_id": notice.notice_id,
        "notice_title": notice.title,
        "posted_at": notice.posted_at,
        "category": notice.category,
        "body_sha256_16": body_hash(notice.body),
        "generation": {
            "model": model_name,
            "agent": "generator",
            "session_id": generator_session_id,
        },
        "validation": {
            "model": model_name,
            "agent": "validator",
            "session_id": validator_session_id,
            "verdict": validation.get("verdict"),
            "score": validation.get("score"),
            "grounded": validation.get("grounded"),
            "type_correct": validation.get("type_correct"),
            "source_span_supported": validation.get("source_span_supported"),
            "reason": normalize_space(validation.get("reason")),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a 2026 Supabase QA smoke JSON.")
    parser.add_argument("--target-count", type=int, default=5)
    parser.add_argument("--candidate-notices", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY_PAID_TIER")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    args = parser.parse_args()

    if args.target_count < 1:
        raise ValueError("--target-count must be at least 1")

    genai, generator_model = configure_gemini(args.api_key_env, args.model)
    _, validator_model = configure_gemini(args.api_key_env, args.model)
    generator_session_id = f"generator-{uuid.uuid4().hex}"
    validator_session_id = f"validator-{uuid.uuid4().hex}"

    notices = load_2026_notices(limit=args.candidate_notices, seed=args.seed)
    if not notices:
        raise RuntimeError("No 2026 Supabase notices found.")

    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    type_cursor = 0

    for notice in notices:
        if len(rows) >= args.target_count:
            break
        qa_type = QA_TYPES[type_cursor % len(QA_TYPES)]
        type_cursor += 1

        qa = generate_candidate(notice, qa_type, generator_model, genai, args.max_retries)
        if qa is None:
            rejected.append({"notice_db_id": notice.id, "type": qa_type, "reason": "unsupported"})
            continue
        if qa["question"] in seen_questions:
            rejected.append({"notice_db_id": notice.id, "type": qa_type, "reason": "duplicate_question"})
            continue

        validation = validate_candidate(notice, qa, validator_model, genai, args.max_retries)
        passed = (
            validation.get("verdict") == "pass"
            and int(validation.get("score") or 0) >= 4
            and validation.get("grounded") is True
            and validation.get("type_correct") is True
            and validation.get("source_span_supported") is True
        )
        if not passed:
            rejected.append(
                {
                    "notice_db_id": notice.id,
                    "type": qa_type,
                    "question": qa.get("question"),
                    "validation": validation,
                }
            )
            continue

        seen_questions.add(qa["question"])
        rows.append(
            build_row(
                notice=notice,
                qa=qa,
                validation=validation,
                generator_session_id=generator_session_id,
                validator_session_id=validator_session_id,
                model_name=args.model,
            )
        )
        print(f"[accepted {len(rows)}/{args.target_count}] {qa['type']} | {qa['question']}", flush=True)
        if args.sleep_seconds > 0 and len(rows) < args.target_count:
            time.sleep(args.sleep_seconds)

    if len(rows) < args.target_count:
        raise RuntimeError(f"Only generated {len(rows)}/{args.target_count} accepted QA pairs.")

    payload = {
        "version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": {
            "backend": "supabase",
            "table": "notices",
            "source": SOURCE,
            "year": 2026,
            "candidate_notice_count": len(notices),
            "min_body_chars": MIN_BODY_CHARS,
        },
        "pipeline": {
            "stages": ["generation", "validation"],
            "independence": "separate Gemini calls with separate agent prompts and no shared chat history",
            "api_key_env": args.api_key_env,
            "model": args.model,
            "generator_session_id": generator_session_id,
            "validator_session_id": validator_session_id,
        },
        "ground_truth": {
            "field": "gt_url",
            "matching": "url",
            "reason": "URL matching is robust to title duplicates and chunked Pinecone storage.",
        },
        "qa_count": len(rows),
        "qas": rows,
        "rejected_preview": rejected[:10],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Wrote {len(rows)} QA pairs -> {args.output}")


if __name__ == "__main__":
    main()
