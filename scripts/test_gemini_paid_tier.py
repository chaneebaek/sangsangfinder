#!/usr/bin/env python3
"""Smoke-test GEMINI_API_KEY_PAID_TIER with tiny or feature-extraction Gemini REST requests."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REDACTED_KEY_RE = re.compile(r"(key=)[^&\s)]+")

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def _post_gemini(api_key: str, prompt: str, timeout: int, model: str, json_mode: bool = False) -> dict:
    import requests

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )
    payload: dict = {"contents": [{"parts": [{"text": prompt}]}]}
    if json_mode:
        payload["generationConfig"] = {
            "temperature": 0,
            "responseMimeType": "application/json",
        }

    response = requests.post(
        url,
        params={"key": api_key},
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def _response_text(payload: dict) -> str:
    return (
        payload.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "")
        .strip()
    )


def _load_feature_notices(limit: int) -> list[dict]:
    sys.path.insert(0, str(ROOT))
    from api.core.models import load_notices_cache

    notices = load_notices_cache()
    loaded = []
    for notice in notices:
        if notice.get("title") and (notice.get("body") or notice.get("content")):
            loaded.append(notice)
        if len(loaded) >= limit:
            break
    return loaded


def _run_tiny(api_key: str, timeout: int, model: str, json_mode: bool) -> int:
    started_at = time.perf_counter()
    try:
        prompt = (
            'Return only this JSON: {"status":"ok"}'
            if json_mode
            else "Reply with exactly: ok"
        )
        payload = _post_gemini(api_key, prompt, timeout, model, json_mode=json_mode)
    except Exception as exc:
        elapsed = time.perf_counter() - started_at
        message = REDACTED_KEY_RE.sub(r"\1<redacted>", str(exc))
        print(f"FAIL: {type(exc).__name__}: {message}")
        print(f"elapsed_seconds={elapsed:.2f}")
        return 1

    elapsed = time.perf_counter() - started_at
    print(f"OK: response={_response_text(payload)!r}")
    print(f"elapsed_seconds={elapsed:.2f}")
    return 0


def _run_feature(api_key: str, batch_size: int, timeout: int, model: str, json_mode: bool) -> int:
    sys.path.insert(0, str(ROOT))
    from api.services.feature_reranker import _build_extraction_prompt, _parse_json_array

    notices = _load_feature_notices(batch_size)
    prompt = _build_extraction_prompt(notices)
    print(f"feature_batch_size={len(notices)} prompt_chars={len(prompt)} timeout={timeout} model={model} json_mode={json_mode}")

    started_at = time.perf_counter()
    try:
        payload = _post_gemini(api_key, prompt, timeout, model, json_mode=json_mode)
    except Exception as exc:
        elapsed = time.perf_counter() - started_at
        message = REDACTED_KEY_RE.sub(r"\1<redacted>", str(exc))
        print(f"FAIL: {type(exc).__name__}: {message}")
        print(f"elapsed_seconds={elapsed:.2f}")
        return 1

    elapsed = time.perf_counter() - started_at
    text = _response_text(payload)
    rows = _parse_json_array(text)
    print(f"OK: rows={len(rows)} response_chars={len(text)}")
    print(f"elapsed_seconds={elapsed:.2f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["tiny", "feature"], default="tiny")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--json-mode", action="store_true")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY_PAID_TIER")
    if not api_key:
        print("FAIL: GEMINI_API_KEY_PAID_TIER is not set.")
        return 1

    print(f"key_present=true key_length={len(api_key)}")
    if args.mode == "feature":
        return _run_feature(api_key, args.batch_size, args.timeout, args.model, args.json_mode)
    return _run_tiny(api_key, args.timeout, args.model, args.json_mode)


if __name__ == "__main__":
    sys.exit(main())
