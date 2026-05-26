import re
from urllib.parse import urlparse, urlencode, parse_qs, urlunparse

from .config import (
    CATEGORY_PATTERN, SUFFIX_PATTERN,
    CATEGORY_PREFIX, CATEGORY_KEYWORDS,
    CHUNK_SIZE, CHUNK_OVERLAP,
)

_KOREAN_TOKEN_PATTERN = re.compile(r"[가-힣]+")
_DOMAIN_SPLIT_HINTS = (
    "해외", "봉사", "봉사활동", "봉사단", "국제", "교류", "파견", "교환학생",
    "어학연수", "모집", "지원", "장학", "인턴", "현장실습", "채용", "서포터즈",
    "멘토링", "교육", "특강", "공모전", "경진대회", "기숙사", "국가근로",
)


def clean_url(url: str) -> str:
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    params.pop("layout", None)
    new_query = urlencode({k: v[0] for k, v in params.items()})
    return urlunparse(parsed._replace(query=new_query))


def clean_title(raw: str) -> str:
    title = raw.replace("\n", " ").replace("\r", " ")
    title = re.sub(r"\s{2,}", " ", title).strip()
    title = CATEGORY_PATTERN.sub("", title).strip()
    title = SUFFIX_PATTERN.sub("", title).strip()
    return title


def infer_category(title: str, body: str) -> str:
    text = f"{title} {body}"
    if (
        ("봉사" in text and any(term in text for term in ("해외", "WFK", "월드프렌즈", "KOICA")))
        or any(term in title for term in ("해외봉사", "청년봉사단", "프로젝트 봉사단", "봉사단"))
    ):
        return "봉사/서포터즈"

    for prefix, cat in CATEGORY_PREFIX.items():
        if title.startswith(prefix):
            return cat
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in title for kw in keywords):
            return cat
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in body for kw in keywords):
            return cat
    return "기타"


def tokenize_ko(text: str) -> list[str]:
    tokens = re.findall(r"[\w가-힣]+", text.lower())
    expanded = list(tokens)
    for token in tokens:
        if not _KOREAN_TOKEN_PATTERN.fullmatch(token):
            continue
        expanded.extend(hint for hint in _DOMAIN_SPLIT_HINTS if hint in token)
        if 4 <= len(token) <= 12:
            max_n = min(6, len(token))
            expanded.extend(
                token[start : start + n]
                for n in range(2, max_n + 1)
                for start in range(0, len(token) - n + 1)
            )
    return expanded


def chunk_text(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start: start + CHUNK_SIZE])
        if start + CHUNK_SIZE >= len(text):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks
