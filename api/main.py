from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .core.models import get_vector_collection, load_notices_cache
from .schemas import (
    HealthResponse,
    NoticeItem,
    RecommendRequest,
    RecommendResponse,
    SearchRequest,
    SearchResponse,
)
from .services.recommend_service import recommend_notices, summarize_notice
from .services.search_service import (
    generate_llm_reply,
    hybrid_search,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="상상파인더 API",
    description="한성대학교 공지사항 검색 및 추천 API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
def health_check():
    notices = load_notices_cache()
    try:
        indexed = get_vector_collection().count()
    except Exception:
        indexed = 0
    return HealthResponse(
        status="ok",
        notices_count=len(notices),
        indexed_count=indexed,
    )


@app.post("/api/v1/search", response_model=SearchResponse, tags=["Search"])
def search(req: SearchRequest):
    try:
        raw_results = hybrid_search(
            query=req.query,
            top_k=req.top_k,
            alpha=req.alpha,
            category_filter=req.category,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검색 오류: {e}")

    # SearchRequest에 profile이 없으므로 빈 dict 전달 (이름 인사말 생략)
    reply = generate_llm_reply(
        user_query=req.query,
        results=raw_results,
        profile={},
        is_first=req.is_first,
    )

    items = [
        NoticeItem(
            title=r["title"],
            url=r["url"],
            date=r["date"],
            category=r.get("category", "기타"),
            score=r["score"],
        )
        for r in raw_results
    ]
    return SearchResponse(reply=reply, results=items)


@app.post("/api/v1/recommend", response_model=RecommendResponse, tags=["Recommend"])
def recommend(req: RecommendRequest):
    profile = {
        "college":   req.college,
        "track":     req.track,
        "grade":     req.grade,
        "interests": req.interests,
    }
    try:
        raw_results = recommend_notices(profile, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추천 오류: {e}")

    notices  = load_notices_cache()
    body_map = {n["url"]: n.get("body", "") for n in notices}

    items = []
    for r in raw_results:
        body    = body_map.get(r["url"], "")
        summary = summarize_notice(r["title"], body) if body else None
        items.append(NoticeItem(
            title=r["title"],
            url=r["url"],
            date=r["date"],
            category=r.get("category", "기타"),
            score=r["score"],
            summary=summary,
        ))
    return RecommendResponse(results=items)
