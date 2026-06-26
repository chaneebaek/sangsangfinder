"""Small REST client for Gemini calls with hard HTTP timeouts."""
from __future__ import annotations

import re
from typing import Any

import requests

GEMINI_GENERATE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
REDACTED_KEY_RE = re.compile(r"(key=)[^&\s)]+")


class GeminiRequestError(RuntimeError):
    """Raised when a Gemini REST request fails after sanitizing sensitive data."""


def _redact_api_key(message: object) -> str:
    return REDACTED_KEY_RE.sub(r"\1<redacted>", str(message))


def generate_content_text(
    *,
    api_key: str,
    prompt: str,
    model: str = "gemini-2.5-flash",
    timeout: int = 15,
    temperature: float | None = None,
    response_mime_type: str | None = None,
) -> str:
    generation_config: dict[str, Any] = {}
    if temperature is not None:
        generation_config["temperature"] = temperature
    if response_mime_type:
        generation_config["responseMimeType"] = response_mime_type

    payload: dict[str, Any] = {
        "contents": [{"parts": [{"text": prompt}]}],
    }
    if generation_config:
        payload["generationConfig"] = generation_config

    try:
        response = requests.post(
            GEMINI_GENERATE_URL.format(model=model),
            params={"key": api_key},
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
    except requests.HTTPError as exc:
        body = ""
        if exc.response is not None:
            body = f" body={exc.response.text[:500]}"
        raise GeminiRequestError(_redact_api_key(f"{exc}{body}")) from exc
    except Exception as exc:
        raise GeminiRequestError(_redact_api_key(exc)) from exc

    try:
        return (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
            .strip()
        )
    except Exception as exc:
        raise GeminiRequestError(f"Unexpected Gemini response shape: {exc}") from exc
