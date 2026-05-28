"""
Content-based notice recommendation and summarization.
Extracted from app.py: recommend_notices, summarize_notice.
"""
from __future__ import annotations

import re

from ..core.models import get_embed_model, get_vector_collection, get_summary_pipeline


def recommend_notices(user_profile: dict, top_k: int = 5) -> list[dict]:
    model      = get_embed_model()
    collection = get_vector_collection()

    interests_str = ", ".join(user_profile.get("interests", []))
    query = (
        f"{user_profile.get('college', '')} "
        f"{user_profile.get('track', '')} "
        f"{user_profile.get('grade', '')} 학생 관심사: {interests_str}"
    )

    n_docs = collection.count()
    if n_docs == 0:
        return []

    results = collection.query(
        query_embeddings=[model.encode(query).tolist()],
        n_results=min(top_k * 5, n_docs),
        include=["metadatas", "distances"],
    )

    seen_urls: dict[str, dict] = {}
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        score = round(1 - dist, 4)
        if meta.get("category") in user_profile.get("interests", []):
            score = min(score + 0.05, 1.0)
        url = meta["url"]
        if url not in seen_urls or score > seen_urls[url]["score"]:
            seen_urls[url] = {**meta, "score": score}

    items = sorted(seen_urls.values(), key=lambda x: x["score"], reverse=True)
    return items[:top_k]


def summarize_notice(title: str, body: str) -> str:
    pipe = get_summary_pipeline()
    if pipe:
        try:
            result = pipe(f"제목: {title}\n\n{body[:512]}", truncation=True)
            return result[0]["summary_text"]
        except Exception:
            pass

    sentences = re.split(r"[.!?。]\s*", body)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    return ". ".join(sentences[:2]) + "." if sentences else body[:150]
