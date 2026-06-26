# ============================================================
# app_api.py — 상상파인더 FastAPI 백엔드
# uvicorn app_api:app --reload --port 8000
# ============================================================

import logging
import os, sys, re, json
from datetime import datetime
from typing import Optional

# ── sys.path 최우선 설정 (다른 import 전에 반드시 위치해야 함) ──
_BASE = os.path.dirname(os.path.abspath(__file__))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
logger = logging.getLogger(__name__)

# ── 외부 모듈 import ─────────────────────────────────────────
from recommend import load_notices_from_supabase, two_tower_recommend, get_supabase
from api.services.search_service import hybrid_search as pinecone_hybrid_search, generate_llm_reply as search_generate_llm_reply

PINECONE_CATEGORY_MAP = {
    "취업/채용": ["취업/채용", "인턴십"],
    "학사행정": ["학사행정"],
    "학생활동/비교과": ["학생활동/비교과", "비교과", "교육/특강"],
    "대외활동": ["대외활동", "봉사/서포터즈"],
    "공모전/경진대회": ["공모전/경진대회"],
    "국제교류": ["국제교류"],
    "창업": ["창업"],
    "장학금": ["장학금", "학자금/근로장학"],
    "기숙사": ["기숙사", "기숙사/생활관"],
    "ROTC": ["ROTC"],
}

# ── 핵심 함수 ────────────────────────────────────────────────

def search_rag_notices(query, category_filter="전체", top_k=20, profile=None):
    try:
        pinecone_categories = (
            PINECONE_CATEGORY_MAP.get(category_filter, [category_filter])
            if category_filter != "전체" else None
        )
        return pinecone_hybrid_search(
            query=query,
            top_k=top_k,
            category_filter=pinecone_categories,
            candidate_k=50,
            feature_rerank=True,
            profile=profile or {},
        )
    except Exception as e:
        print(f"[검색 오류] {e}")
        return []

def generate_llm_reply(user_query, results, profile, is_first=False):
    try:
        return search_generate_llm_reply(
            user_query=user_query,
            results=results,
            profile=profile,
            is_first=is_first,
        )
    except Exception as e:
        print(f"[LLM 오류] {e}")
        return f"총 {len(results)}개의 관련 공지를 찾았습니다." if results else "관련 공지를 찾지 못했습니다."

def _region_match(user_region: str, db_region: str) -> bool:
    """사용자 입력 지역(통일 형식)과 DB 지역값 매칭"""
    if not db_region:
        return True  # DB에 지역 조건 없으면 누구나 해당
    if not user_region:
        return False  # 사용자 지역 미입력 시 지역 조건 있는 장학금 제외

    u = user_region.strip()
    d = db_region.strip()

    if u == d:
        return True  # 완전 일치

    # 사용자가 '서울특별시 강남구' → DB가 '서울특별시'(서울 전체)면 해당
    # 사용자가 '경기도 수원시' → DB가 '경기도'(경기 전체)면 해당
    u_parts = u.split()
    if len(u_parts) >= 2 and d == u_parts[0]:
        return True

    return False

def filter_scholarships(profile):
    try:
        today        = datetime.now().strftime('%Y-%m-%d')
        income_level = profile.get('income_level')
        gpa          = profile.get('gpa')
        region       = profile.get('region')
        loan         = profile.get('loan')
        grade        = profile.get('grade', '')

        grade_num = None
        try: grade_num = int(grade.replace('학년','').strip())
        except: pass

        income_num = None
        if income_level and income_level != "모름/해당없음":
            try: income_num = int(income_level.replace("분위",""))
            except: pass

        gpa_num = None
        if gpa and gpa not in ("모름/해당없음", "", None):
            try: gpa_num = float(str(gpa))
            except: pass

        res = get_supabase().table("scholarships").select("*").eq("is_application", True).execute()
        filtered = []
        for s in (res.data or []):
            target_grade = s.get('target_grade')
            if isinstance(target_grade, str):
                try: target_grade = json.loads(target_grade)
                except: target_grade = []
            target_status = s.get('target_status')
            if isinstance(target_status, str):
                try: target_status = json.loads(target_status)
                except: target_status = []

            if s.get('end_date_type') == '명시' and s.get('end_date'):
                if s['end_date'] < today: continue
            if grade_num and target_grade:
                if grade_num not in target_grade: continue
            if target_status and '재학' not in target_status: continue
            if income_num is not None and s.get('income_max') is not None:
                if income_num > s['income_max']: continue
            if gpa_num is not None and s.get('min_gpa') is not None:
                if gpa_num < s['min_gpa']: continue
            if s.get('region') is not None:
                if not _region_match(region, s['region']): continue
            if s.get('income_required') and loan != "관심있음": continue
            filtered.append(s)

        application_notices = []
        if filtered:
            notice_ids  = [s['notice_id'] for s in filtered]
            notices_res = get_supabase().table("notices").select(
                "id,notice_id,title,url,posted_at,body,category"
            ).in_("notice_id", notice_ids).execute()
            notices_map = {n['notice_id']: n for n in (notices_res.data or [])}
            for s in filtered:
                notice = notices_map.get(s['notice_id'], {})
                if notice:
                    notice['scholarship_info'] = s
                    application_notices.append(notice)

        rel_res = get_supabase().table("scholarships").select("notice_id").eq("is_application", False).execute()
        related_notices = []
        if rel_res.data:
            rel_ids     = [r['notice_id'] for r in rel_res.data]
            rel_notices = get_supabase().table("notices").select(
                "id,notice_id,title,url,posted_at,body,category"
            ).in_("notice_id", rel_ids).order("posted_at", desc=True).limit(3).execute()
            related_notices = rel_notices.data or []

        return application_notices, related_notices
    except Exception as e:
        print(f"장학금 필터링 오류: {e}")
        return [], []

def filter_dormitory(profile):
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        gender = profile.get('gender')
        dorm_interest = profile.get('dorm_interest', [])
        res = get_supabase().table("dormitories").select("*").execute()
        filtered = []
        for d in (res.data or []):
            if d.get('end_date_type') == '명시' and d.get('end_date'):
                if d['end_date'] < today: continue
            if dorm_interest:  # 관심 기숙사 선택했을 때만 이름 필터 적용
                if not any(dorm in (d.get('name') or '') for dorm in dorm_interest): continue
            if gender == '남성' and d.get('male_quota') == 0: continue
            if gender == '여성' and d.get('female_quota') == 0: continue
            filtered.append(d)
        if not filtered: return []
        notice_ids = [d['notice_id'] for d in filtered]
        notices_res = get_supabase().table("notices").select(
            "id,notice_id,title,url,posted_at,body,category"
        ).in_("notice_id", notice_ids).execute()
        notices_map = {n['notice_id']: n for n in (notices_res.data or [])}
        results = []
        for d in filtered:
            notice = notices_map.get(d['notice_id'], {})
            if notice:
                notice['dormitory_info'] = d
                results.append(notice)
        return results
    except Exception as e:
        print(f"기숙사 필터링 오류: {e}")
        return []

def filter_rotc():
    try:
        res = get_supabase().table("notices").select(
            "id,notice_id,title,url,posted_at,body,category"
        ).eq("category", "ROTC").order("posted_at", desc=True).limit(5).execute()
        return res.data or []
    except Exception as e:
        print(f"ROTC 필터링 오류: {e}")
        return []

def sync_profile_to_supabase_user(profile):
    try:
        from app import sync_profile_to_supabase_user as _sync
        _sync(profile)
    except Exception as e:
        print(f"프로필 동기화 오류: {e}")

# ── FastAPI 앱 ───────────────────────────────────────────────
app = FastAPI(title="상상파인더 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory=os.path.join(_BASE, "images")), name="images")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(
        os.path.join(_BASE, "images", "icon_상상부기.png"),
        media_type="image/png",
    )

@app.get("/")
def root():
    return FileResponse(os.path.join(_BASE, "sangsang_finder.html"))

# ── 요청 모델 ────────────────────────────────────────────────
class Profile(BaseModel):
    name: str = ""
    college: str = ""
    track: str = ""
    grade: str = ""
    interests: list[str] = []
    income_level: Optional[str] = None
    gpa: Optional[str] = None
    region: Optional[str] = None
    loan: Optional[str] = None
    gender: Optional[str] = None
    dorm_interest: list[str] = []
    rotc_interest: bool = False
    phone: Optional[str] = None

class ChatRequest(BaseModel):
    query: str
    category: str = "전체"
    profile: Profile
    history: list[dict] = []
    is_first: bool = False

class RecommendRequest(BaseModel):
    profile: Profile

# ── 챗봇 API ─────────────────────────────────────────────────
@app.post("/api/chat")
async def chat(req: ChatRequest):
    profile = req.profile.dict()
    results = search_rag_notices(req.query, req.category, top_k=20, profile=profile)
    reply = generate_llm_reply(req.query, results, profile, req.is_first)
    reranker_warning = any(
        ((r.get("feature_reranker") or {}).get("features") or {}).get("extraction", {}).get("llm_failed")
        for r in results
    )
    if reranker_warning:
        logger.warning("LLM 연결에 실패하여 규칙 기반으로 계산한 결과입니다.")
    notices = []
    for r in results[:3]:
        notices.append({
            "title":    r.get("title", ""),
            "category": r.get("category", ""),
            "date":     (r.get("posted_at") or r.get("date") or "")[:10],
            "url":      r.get("url") or r.get("link", "#"),
            "summary":  (r.get("body") or r.get("content") or "")[:120],
        })
    return {
        "reply": reply,
        "results": notices,
    }

# ── 추천 API ─────────────────────────────────────────────────
@app.post("/api/recommend")
async def recommend(req: RecommendRequest):
    p = req.profile.dict()
    if p.get("phone"):
        try: sync_profile_to_supabase_user(p)
        except: pass

    interests = p.get("interests", [])

    def fmt(n):
        return {
            "title":    n.get("title", ""),
            "category": n.get("category", ""),
            "date":     (n.get("posted_at") or n.get("date") or "")[:10],
            "url":      n.get("url") or n.get("link", "#"),
            "summary":  (n.get("body") or "")[:120],
        }

    # ── Two-Tower 일반 추천 ────────────────────────────────────
    recs = []
    try:
        recs = two_tower_recommend(
            college=p.get("college", ""),
            track=p.get("track", ""),
            year=p.get("grade", ""),
            interests=interests,
            top_k=10,
        )
    except Exception as e:
        print(f"two_tower 오류: {e}")

    seen = set()
    general = []
    for res in recs:
        n = res.get('notice', res)
        if not n.get("title"): continue
        key = n.get("url") or n.get("title", "")
        if key not in seen:
            seen.add(key)
            general.append(fmt(n))

    # ── 장학금 섹션 ───────────────────────────────────────────
    scholarships = []
    scholarship_related = []
    if "장학금" in interests:
        try:
            sch, rel = filter_scholarships(p)
            for n in sch:
                if not n.get("title"): continue
                key = n.get("url") or n.get("title", "")
                if key not in seen:
                    seen.add(key)
                    scholarships.append(fmt(n))
            # 관련 안내는 필터 결과 없어도 최신 3개 무조건
            for n in rel[:3]:
                if not n.get("title"): continue
                key = n.get("url") or n.get("title", "")
                if key not in seen:
                    seen.add(key)
                    scholarship_related.append(fmt(n))
        except Exception as e:
            print(f"장학금 필터 오류: {e}")

    # ── 기숙사 섹션 ───────────────────────────────────────────
    dormitories = []
    if "기숙사" in interests:
        try:
            for n in filter_dormitory(p)[:3]:
                if not n.get("title"): continue
                key = n.get("url") or n.get("title", "")
                if key not in seen:
                    seen.add(key)
                    dormitories.append(fmt(n))
        except Exception as e:
            print(f"기숙사 필터 오류: {e}")

    # ── ROTC 섹션 (rotc_interest 무관하게 관심사에 ROTC 있으면 표시) ──
    rotc = []
    if "ROTC" in interests:
        try:
            for n in filter_rotc()[:2]:
                if not n.get("title"): continue
                key = n.get("url") or n.get("title", "")
                if key not in seen:
                    seen.add(key)
                    rotc.append(fmt(n))
        except Exception as e:
            print(f"ROTC 필터 오류: {e}")

    return {
        "scholarships": scholarships,           # is_application=True 장학금
        "scholarship_related": scholarship_related,  # 장학금 관련 안내
        "dormitories": dormitories,
        "rotc": rotc,
        "notices": general[:10],
    }


# ── 프로필 동기화 API ─────────────────────────────────────────
@app.post("/api/sync-profile")
async def sync_profile_endpoint(req: RecommendRequest):
    try:
        sync_profile_to_supabase_user(req.profile.dict())
        print(f"[sync-profile] {req.profile.name} 저장 완료")
        return {"status": "ok"}
    except Exception as e:
        print(f"[sync-profile 오류] {e}")
        return {"status": "error", "message": str(e)}
