from __future__ import annotations

from pydantic import BaseModel, Field


class NoticeItem(BaseModel):
    title:    str
    url:      str
    date:     str
    category: str
    score:    float
    summary:  str | None = None


# ── Search ──────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query:    str
    category: str | None = None          # None → 전체 카테고리
    top_k:    int        = Field(default=5,   ge=1, le=20)
    alpha:    float      = Field(default=0.5, ge=0.0, le=1.0)
    is_first: bool       = False          # 첫 번째 대화 여부 (인사말 포함)
    feature_rerank: bool = True
    profile: dict       = Field(default_factory=dict)


class SearchResponse(BaseModel):
    reply:   str
    results: list[NoticeItem]


# ── Recommend ───────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    college:   str
    track:     str
    grade:     str
    interests: list[str] = []
    top_k:     int       = Field(default=5, ge=1, le=20)


class RecommendResponse(BaseModel):
    results: list[NoticeItem]


# ── Health ──────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:        str
    notices_count: int
    indexed_count: int
