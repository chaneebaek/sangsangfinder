# ============================================================
# app.py — 상상파인더 UI
# ============================================================

import os, re, json, hashlib, warnings
import numpy as np
from datetime import datetime
import html

import streamlit as st
from recommend import (
    load_notices_from_supabase,
    load_two_tower_model,
    two_tower_recommend,
    classify_job_type,
    get_supabase,
)
from api.services.search_service import hybrid_search as pinecone_hybrid_search

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# ── 경로 설정 ─────────────────────────────────────────────────
_BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
EMBED_MODEL_PATH    = os.path.join(_BASE_DIR, "models", "embed_finetuned")
SUMMARY_MODEL_PATH  = os.path.join(_BASE_DIR, "models", "summary_finetuned")
CLASSIFY_MODEL_PATH = os.path.join(_BASE_DIR, "models", "classify_finetuned")
BASE_MODEL_EMBED    = "jhgan/ko-sroberta-multitask"
PROFILE_CACHE_PATH  = os.path.join(_BASE_DIR, "data", "profile_cache.json")
GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY")

os.makedirs(os.path.join(_BASE_DIR, "data"), exist_ok=True)

# ── 신버전 카테고리 ───────────────────────────────────────────
CATEGORIES = [
    "취업/채용", "학사행정", "학생활동/비교과",
    "대외활동", "공모전/경진대회", "국제교류", "창업",
    "장학금", "기숙사", "ROTC"
]

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

CATEGORY_PREFIX = {
    "채용정보":   "취업/채용",
    "강소기업채용": "취업/채용",
    "인턴쉽":     "취업/채용",
    "현장실습":   "취업/채용",
    "교외장학금": "장학금",
    "국가장학금": "장학금",
    "공모전":     "공모전/경진대회",
    "정보":       "공모전/경진대회",
    "창업정보":   "창업",
    "창업행사":   "창업",
}

CATEGORY_KEYWORDS = {
    "ROTC": [
        "ROTC", "학군사관", "학군단", "현역병 모집", "예비군", "전문사관", "재병역판정검사"
    ],
    "기숙사": [
        "기숙사", "생활관", "상상빌리지", "우촌학사", "임대기숙사", "사감",
        "입사생 선발", "대학생주택", "학사관"
    ],
    "학생활동/비교과": [
        "비교과", "동아리", "D-School", "포럼", "대동제", "영상제", "입학식",
        "HS CREW", "상상파크", "라이프 디자인", "문화탐방", "만우절", "오찬 소통",
        "Lunch with", "천원의 아침밥", "ESG", "진로집단상담", "리더십 탐험",
        "학생축제", "문화제", "페스티벌", "소모임", "디즈니 프로그램",
        "새내기 새로배움터", "새로배움터", "총학생회", "사진전", "진로 설명회",
        "특강", "교육생", "아카데미", "KDT", "K-디지털", "강좌", "교육과정",
        "역량강화", "평생교육", "RISE", "마이크로디그리", "TOPCIT", "연구방법론",
        "초청강연", "특별강연", "핵심역량진단", "폭력예방교육", "필수교육",
        "전문과정", "SW마에스트로", "코딩 캠프", "직업흥미검사", "심리증진",
        "연구윤리", "워크숍", "진로지도시스템", "진로 캠프", "심폐소생술",
        "저작권", "청년인생설계", "과학살롱", "기초학문"
    ],
    "대외활동": [
        "서포터즈", "서포터스", "봉사", "멘토", "봉사자", "기자단", "자원활동",
        "멘토단", "멘토링", "자원봉사", "홍보대사", "하랑", "소통-e", "앰버서더",
        "방송국 HBS", "홍보단", "수습기자", "모니터링단", "자문단", "바로알림단",
        "기획단", "체험단", "발굴단", "순찰대", "제작단", "자원지도자",
        "볼런톤", "청백리포터"
    ],
    "취업/채용": [
        "채용", "신입", "공채", "취업", "채용박람회", "취업박람회", "모집공고",
        "직무", "채용연계", "추천채용", "인턴", "인턴십", "일경험", "체험형",
        "현장실습", "IPP"
    ],
    "장학금": [
        "장학", "장학생", "장학재단", "장학금", "기부장학", "장학사업",
        "스칼라십", "장학지원", "학자금대출", "학자금", "이자지원", "국가근로",
        "면학근로", "근로장학", "대출이자", "등록금 납부"
    ],
    "학사행정": [
        "수강신청", "수강정정", "졸업", "휴학", "복학", "학점", "트랙변경",
        "성적", "폐강", "복수전공", "부전공", "휴복학", "재입학", "연계전공",
        "Micro Degree", "MD과정", "교양영어", "이수신청", "트랙선택", "계절학기",
        "수업평가", "학위취득유예", "수강포기", "서면신청", "교차전부", "편입생",
        "전부(과)", "학위수여식", "오리엔테이션", "반편성고사", "합격자 공고",
        "합격자 발표", "선발 결과", "이수 면제", "수업운영 안내", "출결",
        "중간고사", "기말고사", "전공과목 변경", "다전공 신청", "학석사연계",
        "학사경고", "자기설계전공", "교양필수"
    ],
    "창업": [
        "창업", "창업동아리", "창업지원", "창업멘토링", "스타트업", "아이디어톤",
        "입주기업", "예비창업", "학생 CEO", "CEO 발굴"
    ],
    "국제교류": [
        "교환학생", "어학연수", "파견", "글로벌버디", "국제교류", "해외", "어학",
        "글로컬", "글로벌 튜터링", "글로벌 Conversation", "단기연수", "K-Move",
        "WEST 연수", "글로벌 동행"
    ],
    "공모전/경진대회": [
        "공모전", "경진대회", "챌린지", "해커톤", "대회", "공모", "문학상"
    ],
}

_CATEGORY_PATTERN = re.compile(r"^(한성공지|국제|학사|비교과|장학|취업|진로|창업|기타|현장실습|교육프로그램|행사|일반공지)\s*")
_SUFFIX_PATTERN   = re.compile(r"\s*(새글|hot|NEW)\s*$", re.IGNORECASE)
_KOREAN_TOKEN_PATTERN = re.compile(r"[가-힣]+")
_DOMAIN_SPLIT_HINTS = (
    "해외", "봉사", "봉사활동", "봉사단", "국제", "교류", "파견", "교환학생",
    "어학연수", "모집", "지원", "장학", "인턴", "현장실습", "채용", "서포터즈",
    "멘토링", "교육", "특강", "공모전", "경진대회", "기숙사", "국가근로",
)

COLLEGE_MAP = {
    "크리에이티브인문예술대학": ["영미문화콘텐츠트랙", "영미언어정보트랙", "한국어교육트랙", "역사문화큐레이션트랙", "역사콘텐츠트랙", "지식정보문화트랙", "디지털인문정보학트랙", "동양화전공", "서양화전공", "한국무용전공", "현대무용전공", "발레전공"],
    "미래융합사회과학대학": ["국제무역트랙", "글로벌비지니스트랙", "기업ㆍ경제분석트랙", "경제금융투자트랙", "공공행정트랙", "법&정책트랙", "부동산트랙", "스마트도시ㆍ교통계획트랙", "기업경영트랙", "비지니스애널리틱스트랙", "회계ㆍ재무경영트랙"],
    "디자인대학": ["패션마케팅트랙", "패션디자인트랙", "패션크리에이티브디렉션트랙", "미디어디자인트랙", "시각디자인트랙", "영상ㆍ애니메이션디자인트랙", "UX/UI디자인트랙", "인테리어디자인트랙", "VMDㆍ전시디자인트랙", "게임그래픽디자인트랙", "뷰티디자인매니지먼트학과"],
    "IT공과대학": ["모바일소프트웨어트랙", "빅데이터트랙", "디지털콘텐츠ㆍ가상현실트랙", "웹공학트랙", "전자트랙", "시스템반도체트랙", "기계시스템디자인트랙", "AI로봇융합트랙", "산업공학트랙", "응용산업데이터공학트랙"],
    "창의융합대학": ["상상력인재학부", "문학문화콘텐츠학과", "AI응용학과", "융합보안학과", "미래모빌리티학과"],
    "글로벌인재대학": ["한국언어문화교육학과", "글로벌K비지니스학과", "영상엔터테인먼트학과", "패션뷰티크리에이션학과", "SW융합학과", "글로벌벤처창업학과"],
    "미래플러스대학": ["융합행정학과", "호텔외식경영학과", "뷰티디자인학과", "비지니스컨설팅학과", "ICT융합디자인학과", "AIㆍ소프트웨어학과", "뷰티매니지먼트학과", "디지털콘텐츠디자인학과", "인테리어디자인학과", "스마트제조혁신컨설팅학과"],
}

DEPT_URLS = {
    "한국언어문화교육학과":"https://www.hansung.ac.kr/global/1511/subview.do","글로벌K비지니스학과":"https://www.hansung.ac.kr/global/1516/subview.do","영상엔터테인먼트학과":"https://www.hansung.ac.kr/global/1521/subview.do","패션뷰티크리에이션학과":"https://www.hansung.ac.kr/global/1526/subview.do","SW융합학과":"https://www.hansung.ac.kr/global/1531/subview.do","글로벌벤처창업학과":"https://www.hansung.ac.kr/global/6807/subview.do","융합행정학과":"https://www.hansung.ac.kr/futureplus/731/subview.do","호텔외식경영학과":"https://www.hansung.ac.kr/futureplus/734/subview.do","뷰티디자인학과":"https://www.hansung.ac.kr/futureplus/737/subview.do","비지니스컨설팅학과":"https://www.hansung.ac.kr/futureplus/740/subview.do","ICT융합디자인학과":"https://www.hansung.ac.kr/futureplus/743/subview.do","AIㆍ소프트웨어학과":"https://www.hansung.ac.kr/futureplus/746/subview.do","뷰티매니지먼트학과":"https://www.hansung.ac.kr/futureplus/749/subview.do","디지털콘텐츠디자인학과":"https://www.hansung.ac.kr/futureplus/754/subview.do","인테리어디자인학과":"https://www.hansung.ac.kr/futureplus/759/subview.do","스마트제조혁신컨설팅학과":"https://www.hansung.ac.kr/futureplus/764/subview.do",
    "상상력인재학부":"https://www.hansung.ac.kr/CreCon/2761/subview.do","문학문화콘텐츠학과":"https://www.hansung.ac.kr/CreCon/2768/subview.do","AI응용학과":"https://www.hansung.ac.kr/CreCon/2777/subview.do","융합보안학과":"https://www.hansung.ac.kr/CreCon/2787/subview.do","미래모빌리티학과":"https://www.hansung.ac.kr/CreCon/2796/subview.do",
    "국제무역트랙":"https://www.hansung.ac.kr/SclScn/5260/subview.do","글로벌비지니스트랙":"https://www.hansung.ac.kr/SclScn/5267/subview.do","기업ㆍ경제분석트랙":"https://www.hansung.ac.kr/SclScn/5274/subview.do","경제금융투자트랙":"https://www.hansung.ac.kr/SclScn/5281/subview.do","공공행정트랙":"https://www.hansung.ac.kr/SclScn/5295/subview.do","법&정책트랙":"https://www.hansung.ac.kr/SclScn/5303/subview.do","부동산트랙":"https://www.hansung.ac.kr/SclScn/5313/subview.do","스마트도시ㆍ교통계획트랙":"https://www.hansung.ac.kr/SclScn/5321/subview.do","기업경영트랙":"https://www.hansung.ac.kr/SclScn/5328/subview.do","비지니스애널리틱스트랙":"https://www.hansung.ac.kr/SclScn/5336/subview.do","회계ㆍ재무경영트랙":"https://www.hansung.ac.kr/SclScn/5344/subview.do",
    "모바일소프트웨어트랙":"https://www.hansung.ac.kr/Engineering/4887/subview.do","빅데이터트랙":"https://www.hansung.ac.kr/Engineering/4894/subview.do","디지털콘텐츠ㆍ가상현실트랙":"https://www.hansung.ac.kr/Engineering/4901/subview.do","웹공학트랙":"https://www.hansung.ac.kr/Engineering/4908/subview.do","전자트랙":"https://www.hansung.ac.kr/Engineering/4915/subview.do","시스템반도체트랙":"https://www.hansung.ac.kr/Engineering/4922/subview.do","기계시스템디자인트랙":"https://www.hansung.ac.kr/Engineering/4929/subview.do","AI로봇융합트랙":"https://www.hansung.ac.kr/Engineering/4936/subview.do","산업공학트랙":"https://www.hansung.ac.kr/Engineering/4992/subview.do","응용산업데이터공학트랙":"https://www.hansung.ac.kr/Engineering/5020/subview.do",
    "패션마케팅트랙":"https://www.hansung.ac.kr/Design/5103/subview.do","패션디자인트랙":"https://www.hansung.ac.kr/Design/5110/subview.do","패션크리에이티브디렉션트랙":"https://www.hansung.ac.kr/Design/5117/subview.do","미디어디자인트랙":"https://www.hansung.ac.kr/Design/5124/subview.do","시각디자인트랙":"https://www.hansung.ac.kr/Design/5145/subview.do","영상ㆍ애니메이션디자인트랙":"https://www.hansung.ac.kr/Design/5131/subview.do","UX/UI디자인트랙":"https://www.hansung.ac.kr/Design/5173/subview.do","인테리어디자인트랙":"https://www.hansung.ac.kr/Design/5159/subview.do","VMDㆍ전시디자인트랙":"https://www.hansung.ac.kr/Design/5152/subview.do","게임그래픽디자인트랙":"https://www.hansung.ac.kr/Design/5166/subview.do","뷰티디자인매니지먼트학과":"https://www.hansung.ac.kr/Design/5180/subview.do",
    "영미문화콘텐츠트랙":"https://www.hansung.ac.kr/HmnArt/5641/subview.do","영미언어정보트랙":"https://www.hansung.ac.kr/HmnArt/5577/subview.do","한국어교육트랙":"https://www.hansung.ac.kr/HmnArt/5584/subview.do","역사문화큐레이션트랙":"https://www.hansung.ac.kr/HmnArt/5627/subview.do","역사콘텐츠트랙":"https://www.hansung.ac.kr/HmnArt/5634/subview.do","지식정보문화트랙":"https://www.hansung.ac.kr/HmnArt/5613/subview.do","디지털인문정보학트랙":"https://www.hansung.ac.kr/HmnArt/5620/subview.do","동양화전공":"https://www.hansung.ac.kr/HmnArt/5648/subview.do","서양화전공":"https://www.hansung.ac.kr/HmnArt/5655/subview.do","한국무용전공":"https://www.hansung.ac.kr/HmnArt/5662/subview.do","현대무용전공":"https://www.hansung.ac.kr/HmnArt/5669/subview.do","발레전공":"https://www.hansung.ac.kr/HmnArt/5676/subview.do",
}

# ============================================================
# 유틸
# ============================================================

def _load_image_b64(filename):
    import base64
    try:
        with open(os.path.join(_BASE_DIR, filename), "rb") as f:
            return base64.b64encode(f.read()).decode()
    except: return ""

def get_logo_base64(): return _load_image_b64("logo.png")
def get_hsu_base64():  return _load_image_b64("hsu.png")

def tokenize_ko(text):
    tokens = re.findall(r"[\w가-힣]+", text.lower())
    expanded = list(tokens)
    for token in tokens:
        if not _KOREAN_TOKEN_PATTERN.fullmatch(token):
            continue
        expanded.extend(hint for hint in _DOMAIN_SPLIT_HINTS if hint in token)
        if 4 <= len(token) <= 12:
            max_n = min(6, len(token))
            expanded.extend(
                token[start:start + n]
                for n in range(2, max_n + 1)
                for start in range(0, len(token) - n + 1)
            )
    return expanded

def infer_category(title, body):
    text = f"{title} {body}"
    if (
        ("봉사" in text and any(term in text for term in ("해외", "WFK", "월드프렌즈", "KOICA")))
        or any(term in title for term in ("해외봉사", "청년봉사단", "프로젝트 봉사단", "봉사단"))
    ):
        return "봉사/서포터즈"

    for prefix, cat in CATEGORY_PREFIX.items():
        if title.startswith(prefix): return cat
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(kw in title for kw in kws): return cat
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(kw in body for kw in kws): return cat
    return "기타"

# ============================================================
# 모델 로더
# ============================================================

@st.cache_resource
@st.cache_resource
@st.cache_resource
@st.cache_resource
@st.cache_data(ttl=600, show_spinner=False)
def _markdown_log_cell(value):
    return str(value or "").replace("\r", " ").replace("\n", " ").replace("|", r"\|")

def summarize_notice(title, body):
    import html as _html
    body = body or ''
    clean_body = re.sub(r'<[^>]+>', '', body)
    result = clean_body[:200] + '...' if len(clean_body) > 200 else clean_body
    return _html.escape(result)

def get_gemini_model(api_key):
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.5-flash")
    except Exception as e:
        st.error(f"[Gemini 오류] {e}"); return None

def generate_llm_reply(user_query, results, profile, is_first=False):
    model = get_gemini_model(GEMINI_API_KEY) if GEMINI_API_KEY else None
    if not model: return f"총 {len(results)}개의 관련 공지를 찾았습니다." if results else "관련 공지를 찾지 못했습니다."
    if not results: return "관련 공지를 찾지 못했습니다. 다른 키워드로 검색해보세요."
    notices_data = load_notices_from_supabase()
    body_map     = {n["url"]: n.get("body","") for n in notices_data}
    context = "\n\n".join([
        f"[공지 {i+1}]\n제목: {r['title']}\n날짜: {r['date']}\n내용: {(r.get('content') or body_map.get(r['url'],''))[:800]}"
        for i, r in enumerate(results[:3])
    ])
    greeting = f"{profile.get('name','')}님, 안녕하세요. " if is_first else ""
    prompt = f"""당신은 한성대학교 공지사항 안내 도우미입니다.
아래 공지사항 본문을 바탕으로 사용자 질문에 직접적이고 구체적으로 답변하세요.
- 날짜, 금액, 조건 등 구체적인 정보가 있으면 반드시 포함하세요.
- "공지를 참고하세요" 같은 말은 절대 하지 마세요.
- 2~3문장으로 간결하게 답변하세요.
- 답변 시작: "{greeting}"

[공지 본문]
{context}

[질문]
{user_query}"""
    try: return get_gemini_model(GEMINI_API_KEY).generate_content(prompt).text.strip()
    except Exception as e: return f"[Gemini 오류] {e}"

# ============================================================
# CSS
# ============================================================

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700&display=swap');
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background-color: #f2f2f7 !important; font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Noto Sans KR", sans-serif !important; }
.block-container { padding-left: 2rem !important; padding-right: 2rem !important; max-width: 860px !important; margin: 0 auto !important; }
[data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"], [data-testid="stSidebarCollapseButton"], button[kind="header"], .st-emotion-cache-h5rgaw, [data-testid="stSidebar"] > div:first-child > div:first-child button { display: none !important; }
section[data-testid="stSidebar"] { width: 260px !important; min-width: 260px !important; transform: translateX(0) !important; visibility: visible !important; background: #ffffff !important; border-right: 1px solid rgba(0,0,0,0.08) !important; }
section[data-testid="stSidebar"] > div:first-child { width: 260px !important; padding: 8px 16px 16px 16px !important; overflow: hidden !important; }
[data-testid="stSidebar"] > div:first-child > div:first-child { padding-top: 0 !important; margin-top: -3rem !important; }
.stButton > button { background: #0a84ff !important; color: white !important; border: none !important; border-radius: 8px !important; font-size: 13px !important; font-weight: 500 !important; padding: 7px 14px !important; transition: all 0.15s ease !important; }
.stButton > button:hover { background: #409cff !important; transform: scale(1.01) !important; }
.stTextInput > div > div > input { background: white !important; border: 1px solid rgba(0,0,0,0.12) !important; border-radius: 12px !important; font-size: 15px !important; padding: 12px 16px !important; }
.stTextInput > div > div > input:focus { border-color: #0a84ff !important; box-shadow: 0 0 0 3px rgba(10,132,255,0.15) !important; outline: none !important; }
.stSelectbox > div > div, .stMultiSelect > div > div { background: white !important; border-radius: 10px !important; border: 1px solid rgba(0,0,0,0.12) !important; }
:root { --primary-color: #0a84ff !important; }
[data-baseweb="tag"] { background-color: rgba(10,132,255,0.12) !important; }
[data-baseweb="tag"] span { color: #0a84ff !important; }
.stTabs [data-baseweb="tab-list"] { background: rgba(120,120,128,0.12) !important; border-radius: 12px !important; padding: 3px !important; gap: 2px !important; width: fit-content !important; margin: 0 auto 16px auto !important; }
.stTabs [data-baseweb="tab"] { border-radius: 10px !important; font-size: 14px !important; font-weight: 500 !important; color: #3c3c43 !important; padding: 8px 20px !important; }
.stTabs [aria-selected="true"] { background: white !important; color: #1d1d1f !important; box-shadow: 0 1px 4px rgba(0,0,0,0.12) !important; font-weight: 600 !important; }
.stTabs [data-baseweb="tab-border"] { background-color: transparent !important; }
.chat-bubble-user { background: #0a84ff; color: white; border-radius: 20px 20px 5px 20px; padding: 10px 16px; max-width: 55%; width: fit-content; margin-left: auto; margin-bottom: 10px; font-size: 14px; line-height: 1.5; word-break: break-word; }
.chat-bubble-bot { background: #f0f4ff; color: #1d1d1f; border-radius: 20px 20px 20px 5px; padding: 12px 18px; max-width: 75%; width: fit-content; margin-bottom: 10px; font-size: 14px; line-height: 1.55; border: 1px solid #dce8ff; word-break: break-word; }
.mac-bar { display: flex; align-items: center; gap: 7px; padding: 10px 16px; background: #e8eef5; border-radius: 10px 10px 0 0; margin: -14px -14px 0 -14px; border-bottom: 1px solid rgba(0,0,0,0.07); }
.mac-dot { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }
.mac-dot-red { background: #ff5f57; } .mac-dot-yellow { background: #febc2e; } .mac-dot-green { background: #28c840; }
[data-testid="stMarkdownContainer"]:has(.mac-bar) { margin-bottom: -1rem !important; line-height: 0 !important; }
[data-testid="stVerticalBlockBorderWrapper"]:has(.chat-bubble-user), [data-testid="stVerticalBlockBorderWrapper"]:has(.chat-bubble-bot), [data-testid="stVerticalBlockBorderWrapper"]:has(.mac-bar) { background: #f9f9fb !important; border-radius: 13px !important; min-height: 0 !important; border: 1px solid rgba(0,0,0,0.07) !important; padding: 14px !important; }
.notice-card { background: white; border-radius: 14px; padding: 16px 20px; margin-bottom: 12px; box-shadow: 0 1px 5px rgba(0,0,0,0.06); border: 1px solid rgba(0,0,0,0.05); transition: box-shadow 0.15s ease; }
.notice-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.10); }
.notice-tag { display: inline-block; background: rgba(10,132,255,0.1); color: #0a84ff; border-radius: 7px; padding: 2px 9px; font-size: 11px; font-weight: 700; margin-right: 8px; }
.notice-title { font-size: 15px; font-weight: 600; color: #1d1d1f; margin: 6px 0 2px 0; line-height: 1.4; }
.notice-date { font-size: 12px; color: #86868b; }
.notice-summary { font-size: 13px; color: #3c3c43; margin-top: 8px; line-height: 1.55; }
.sb-label { font-size: 10px; font-weight: 700; color: rgba(0,0,0,0.35); text-transform: uppercase; letter-spacing: 0.1em; margin: 18px 0 6px 2px; }
.sb-info-row { display: flex; align-items: flex-start; gap: 8px; padding: 5px 0; }
.sb-info-key { font-size: 12px; color: rgba(0,0,0,0.4); min-width: 52px; }
.sb-info-val { font-size: 13px; font-weight: 500; color: #1d1d1f; line-height: 1.4; }
[data-testid="stVerticalBlockBorderWrapper"] { background: white !important; border-radius: 20px !important; padding: 32px 36px !important; box-shadow: 0 4px 24px rgba(0,0,0,0.08) !important; border: none !important; }
hr { border: none; border-top: 1px solid rgba(0,0,0,0.08) !important; margin: 14px 0 !important; }
#MainMenu, footer, header { visibility: hidden; }
</style>
"""

# ============================================================
# 알림 대상자 반환
# ============================================================

def _notification_user_id(profile: dict) -> str:
    phone = profile.get("phone") or ""
    seed = phone or profile.get("name") or json.dumps(profile, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]

def sync_profile_to_supabase_user(profile: dict) -> None:
    """로컬 온보딩 프로필을 Supabase users 테이블에 업서트한다."""
    if not profile.get("name"):
        return

    user_row = {
        "user_id":       _notification_user_id(profile),
        "name":          profile.get("name", ""),
        "phone":         profile.get("phone"),
        "interests":     profile.get("interests") or [],
        "track":         profile.get("track", ""),
        "college":       profile.get("college", ""),
        "grade":         profile.get("grade", ""),
        "income_level":  profile.get("income_level"),
        "gpa":           profile.get("gpa"),
        "region":        profile.get("region"),
        "loan":          profile.get("loan"),
        "gender":        profile.get("gender"),
        "dorm_interest": profile.get("dorm_interest") or [],
        "rotc_interest": profile.get("rotc_interest", False),
    }

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
    try:
        if supabase_url and supabase_key and not supabase_url.startswith("your_"):
            try:
                get_supabase().table("users").upsert(user_row, on_conflict="user_id").execute()
            except Exception as rest_error:
                if os.getenv("SUPABASE_DB_URL"):
                    print(f"Supabase REST users 동기화 실패, DB 직접 연결로 재시도: {rest_error}")
                    _sync_profile_to_supabase_user_via_db(user_row)
                else:
                    raise
        else:
            _sync_profile_to_supabase_user_via_db(user_row)
    except Exception as e:
        print(f"Supabase users 동기화 실패: {e}")

def _sync_profile_to_supabase_user_via_db(user_row: dict) -> None:
    """SUPABASE_DB_URL만 있는 로컬 환경에서 users 테이블 생성 후 업서트한다."""
    from crawling.supabase_store import ensure_schema, _connect

    ensure_schema()
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                insert into users (user_id, name, phone, interests, track, updated_at)
                values (%s, %s, %s, %s, %s, now())
                on conflict (user_id) do update set
                    name = excluded.name,
                    phone = excluded.phone,
                    interests = excluded.interests,
                    track = excluded.track,
                    updated_at = now()
                """,
                (
                    user_row["user_id"],
                    user_row["name"],
                    user_row["phone"],
                    user_row["interests"],
                    user_row["track"],
                ),
            )
        conn.commit()

def get_notification_targets(notices: list[dict], users: list[dict] | None = None) -> list[dict]:
    """
    신규 공지 리스트를 받아 알림 대상자 정보 반환.
    알림 담당자에게 전달할 데이터 구조:
    [
        {
            "notice_id":     "222025",
            "title":         "공지 제목",
            "category":      "장학금",
            "category_type": ["장학금"],
            "url":           "https://...",
            "user_name":     "홍길동",
            "phone":         "010-1234-5678",
            "track":         "영미문화콘텐츠트랙",
            "interests":     ["장학금", "취업/채용"],
        },
        ...
    ]
    """
    if users is None:
        try:
            res   = get_supabase().table("users").select("user_id,name,phone,interests,track").execute()
            users = res.data or []
        except Exception as e:
            print(f"users 테이블 로드 오류: {e}")
            return []

    targets   = []

    for notice in notices:
        category = notice.get('category', '')
        category_type = notice.get('category_type') or notice.get('job_types') or []
        if isinstance(category_type, str):
            try:
                category_type = json.loads(category_type)
            except:
                category_type = []

        for user in users:
            if not user or not user.get('phone'):
                continue

            interests = user.get('interests', [])
            if isinstance(interests, str):
                try:
                    interests = json.loads(interests)
                except:
                    interests = [interests]

            interest_set = set(interests)
            matched = category in interest_set or bool(interest_set.intersection(category_type))
            if not matched:
                continue

            targets.append({
                "notice_db_id":   notice.get('id'),
                "notice_id":      notice.get('notice_id', ''),
                "title":          notice.get('title', ''),
                "category":       category,
                "category_type":  category_type,
                "url":            notice.get('url', ''),
                "user_id":        user.get('user_id') or user.get('id') or user.get('phone'),
                "user_name":      user.get('name', ''),
                "phone":          user.get('phone'),
                "track":          user.get('track', ''),
                "interests":      interests,
            })

    return targets

# ============================================================
# 필터링 함수
# ============================================================

def filter_scholarships(profile) -> tuple:
    """(신청 공지 리스트, 관련 공지 리스트) 반환"""
    try:
        today        = datetime.now().strftime('%Y-%m-%d')
        income_level = profile.get('income_level')
        gpa          = profile.get('gpa')
        region       = profile.get('region')
        loan         = profile.get('loan')
        grade        = profile.get('grade', '')

        grade_num = None
        try:
            grade_num = int(grade.replace('학년', '').strip())
        except:
            pass

        income_num = None
        if income_level and income_level != "모름/해당없음":
            try:
                income_num = int(income_level.replace("분위", ""))
            except:
                pass

        gpa_num = None
        if gpa and gpa != "모름/해당없음":
            try:
                gpa_num = float(gpa.split(" ")[0])
            except:
                pass

        # 신청 공지 필터링
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
                if s['end_date'] < today:
                    continue
            if grade_num and target_grade:
                if grade_num not in target_grade:
                    continue
            if target_status and '재학' not in target_status:
                continue
            if income_num is not None and s.get('income_max') is not None:
                if income_num > s['income_max']:
                    continue
            if gpa_num is not None and s.get('min_gpa') is not None:
                if gpa_num < s['min_gpa']:
                    continue
            if s.get('region') is not None and region not in [None, "모름/해당없음"]:
                if s['region'] != region:
                    continue
            if s.get('income_required') and loan != "대출 있음":
                continue
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

        # 관련 공지 최신순 3개
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
        import traceback; traceback.print_exc()
        return [], []


def filter_dormitory(profile) -> list:
    try:
        today         = datetime.now().strftime('%Y-%m-%d')
        gender        = profile.get('gender')
        dorm_interest = profile.get('dorm_interest', [])

        res = get_supabase().table("dormitories").select("*").execute()
        filtered = []
        for d in (res.data or []):
            if d.get('end_date_type') == '명시' and d.get('end_date'):
                if d['end_date'] < today:
                    continue
            if dorm_interest:
                if not any(dorm in (d.get('name') or '') for dorm in dorm_interest):
                    continue
            if gender == '남성' and d.get('male_quota') == 0:
                continue
            if gender == '여성' and d.get('female_quota') == 0:
                continue
            filtered.append(d)

        if not filtered:
            return []

        notice_ids  = [d['notice_id'] for d in filtered]
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
        import traceback; traceback.print_exc()
        return []


def filter_rotc() -> list:
    try:
        res = get_supabase().table("notices").select(
            "id,notice_id,title,url,posted_at,body,category"
        ).eq("category", "ROTC").order("posted_at", desc=True).limit(5).execute()
        return res.data or []
    except Exception as e:
        print(f"ROTC 필터링 오류: {e}")
        return []

# ============================================================
# 온보딩
# ============================================================

def render_onboarding():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    st.markdown("<div style='height:48px'></div>", unsafe_allow_html=True)
    col_logo, col_form = st.columns([1, 2], gap="large")
    with col_logo:
        st.markdown("<div style='height:60px'></div>", unsafe_allow_html=True)
        logo_b64 = get_logo_base64()
        logo_img = f'<img src="data:image/png;base64,{logo_b64}" style="width:90px;height:auto;object-fit:contain;display:block;">' if logo_b64 else '<div style="font-size:56px;">🔍</div>'
        st.markdown(f'<div style="padding-left:8px;">{logo_img}<div style="font-size:28px;font-weight:700;color:#1d1d1f;letter-spacing:-0.03em;margin-top:16px;">상상파인더</div><div style="font-size:14px;color:#86868b;margin-top:6px;line-height:1.6;">한성대 공지를<br>스마트하게 검색하세요.</div></div>', unsafe_allow_html=True)
    with col_form:
        with st.container(border=True):
            st.markdown('<div style="font-size:17px;font-weight:700;color:#1d1d1f;margin-bottom:3px;">반갑습니다 👋</div><div style="font-size:13px;color:#86868b;margin-bottom:14px;">기본 정보를 알려주세요.</div>', unsafe_allow_html=True)

            r1c1, r1c2 = st.columns(2)
            with r1c1: name    = st.text_input("이름", placeholder="홍길동", key="ob_name")
            with r1c2: college = st.selectbox("단과대", list(COLLEGE_MAP.keys()), key="ob_college")
            r2c1, r2c2 = st.columns(2)
            with r2c1: track = st.selectbox("트랙 / 학과", COLLEGE_MAP.get(college, ["기타"]), key="ob_track")
            with r2c2: grade = st.selectbox("학년", ["1학년","2학년","3학년","4학년"], key="ob_grade")
            interests = st.multiselect(
                "관심사",
                ["취업/채용","학사행정","학생활동/비교과","대외활동","공모전/경진대회","국제교류","창업","장학금","기숙사","ROTC"],
                placeholder="관심 카테고리를 선택하세요",
                key="ob_interests"
            )

            # ── 장학금 추가 항목 ───────────────────────────────
            income_level  = None
            gpa           = None
            region        = None
            loan          = None
            if "장학금" in interests:
                st.markdown("<div style='font-size:13px;font-weight:600;color:#1d1d1f;margin-top:12px;margin-bottom:6px;'>📌 장학금 맞춤 필터</div>", unsafe_allow_html=True)
                sc1, sc2 = st.columns(2)
                with sc1:
                    income_level = st.selectbox(
                        "소득분위",
                        ["모름/해당없음","1분위","2분위","3분위","4분위","5분위","6분위","7분위","8분위","9분위","10분위"],
                        key="ob_income"
                    )
                with sc2:
                    gpa = st.selectbox(
                        "최근 학점",
                        ["모름/해당없음","4.0 이상","3.5 이상","3.0 이상","2.5 이상","2.0 이상","2.0 미만"],
                        key="ob_gpa"
                    )
                sc3, sc4 = st.columns(2)
                with sc3:
                    region = st.selectbox(
                        "거주 지역",
                        ["서울","경기/인천","지방","해외"],
                        key="ob_region"
                    )
                with sc4:
                    loan = st.selectbox(
                        "학자금 대출 여부",
                        ["해당없음","대출 있음"],
                        key="ob_loan"
                    )

            # ── 기숙사 추가 항목 ───────────────────────────────
            gender        = None
            dorm_interest = None
            if "기숙사" in interests:
                st.markdown("<div style='font-size:13px;font-weight:600;color:#1d1d1f;margin-top:12px;margin-bottom:6px;'>📌 기숙사 맞춤 필터</div>", unsafe_allow_html=True)
                dc1, dc2 = st.columns(2)
                with dc1:
                    gender = st.selectbox(
                        "성별",
                        ["선택안함","남성","여성"],
                        key="ob_gender"
                    )
                with dc2:
                    dorm_interest = st.multiselect(
                        "관심 기숙사",
                        ["상상빌리지","우촌학사","동소문행복기숙사","에피소드","임대기숙사"],
                        key="ob_dorm"
                    )

            # ── ROTC 추가 항목 ─────────────────────────────────
            rotc_interest = False
            if "ROTC" in interests:
                st.markdown("<div style='font-size:13px;font-weight:600;color:#1d1d1f;margin-top:12px;margin-bottom:6px;'>📌 ROTC</div>", unsafe_allow_html=True)
                rotc_interest = st.checkbox("ROTC 후보생 모집 공지 받기", value=True, key="ob_rotc")

            # ── 전화번호 (알림용) ──────────────────────────────
            st.markdown("<div style='font-size:13px;color:#86868b;margin-top:12px;margin-bottom:4px;'>📱 공지 알림 수신 전화번호 <span style='font-size:11px;'>(선택, 빈칸이면 알림 미사용)</span></div>", unsafe_allow_html=True)
            pc1, pc2, pc3 = st.columns(3)
            with pc1: phone1 = st.text_input("", placeholder="010",  key="ob_phone1", max_chars=3,  label_visibility="collapsed")
            with pc2: phone2 = st.text_input("", placeholder="0000", key="ob_phone2", max_chars=4,  label_visibility="collapsed")
            with pc3: phone3 = st.text_input("", placeholder="0000", key="ob_phone3", max_chars=4,  label_visibility="collapsed")
            phone1 = re.sub(r'\D', '', phone1)
            phone2 = re.sub(r'\D', '', phone2)
            phone3 = re.sub(r'\D', '', phone3)
            phone  = f"{phone1}-{phone2}-{phone3}" if phone1 and phone2 and phone3 else None

            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            if st.button("시작하기 →", use_container_width=True):
                if not name.strip():
                    st.markdown('<div style="background:#e8f1ff;color:#0a84ff;border-radius:10px;padding:10px 16px;font-size:14px;font-weight:500;border:1px solid #b3d1ff;">✏️ 이름을 입력해 주세요.</div>', unsafe_allow_html=True)
                else:
                    profile_data = {
                        "name":          name.strip(),
                        "college":       college,
                        "track":         track,
                        "grade":         grade,
                        "interests":     interests,
                        "income_level":  income_level,
                        "gpa":           gpa,
                        "region":        region,
                        "loan":          loan,
                        "gender":        gender,
                        "dorm_interest": dorm_interest or [],
                        "rotc_interest": rotc_interest,
                        "phone":         phone,
                    }
                    st.session_state.profile  = profile_data
                    st.session_state.onboarded = True
                    sync_profile_to_supabase_user(profile_data)
                    st.rerun()

# ============================================================
# 사이드바
# ============================================================

def render_sidebar(profile):
    with st.sidebar:
        logo_b64 = get_logo_base64()
        logo_img = f'<img src="data:image/png;base64,{logo_b64}" style="width:32px;height:32px;object-fit:contain;flex-shrink:0;">' if logo_b64 else '🔍'
        st.markdown(f'<div style="display:flex;align-items:center;gap:10px;padding-bottom:10px;">{logo_img}<div><div style="font-size:14px;font-weight:700;color:#1d1d1f;">상상파인더</div><div style="font-size:10px;color:#86868b;margin-top:1px;">Hansung Notice Finder</div></div></div><hr/>', unsafe_allow_html=True)
        st.markdown('<div class="sb-label">내 정보</div>', unsafe_allow_html=True)
        info_html = "".join([f'<div class="sb-info-row"><span class="sb-info-key">{k}</span><span class="sb-info-val">{v}</span></div>' for k, v in [("이름", profile.get("name","")), ("단과대학", profile.get("college","")), ("트랙/학과", profile.get("track","")), ("학년", profile.get("grade",""))]])
        st.markdown(info_html, unsafe_allow_html=True)
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown('<div class="sb-label">바로가기</div>', unsafe_allow_html=True)
        links = [("🏫","한성대학교","https://www.hansung.ac.kr/hansung/index.do"),("💻","한성 e-class","https://learn.hansung.ac.kr/"),("📋","종합정보시스템","https://info.hansung.ac.kr/"),("📊","스마트자기관리시스템","https://hsportal.hansung.ac.kr/"),("📚","학술정보관","https://hsel.hansung.ac.kr/")]
        track = profile.get("track","")
        if track in DEPT_URLS: links.append(("🎓", track, DEPT_URLS[track]))
        link_html = "".join([f'<a href="{url}" target="_blank" style="display:flex;align-items:center;gap:9px;padding:8px 10px;border-radius:9px;text-decoration:none;color:#1d1d1f;font-size:13px;font-weight:500;margin-bottom:2px;" onmouseover="this.style.background=\'rgba(0,0,0,0.05)\'" onmouseout="this.style.background=\'transparent\'"><span style="font-size:15px;">{icon}</span><span>{label}</span><span style="margin-left:auto;font-size:11px;color:#aeaeb2;">↗</span></a>' for icon, label, url in links])
        st.markdown(link_html, unsafe_allow_html=True)
        st.markdown("<hr/>", unsafe_allow_html=True)
        _, btn_col, _ = st.columns([0.3, 2.4, 0.3])
        with btn_col:
            if st.button("내 정보 다시 입력", use_container_width=True):
                st.session_state.onboarded = False; st.session_state.profile = {}; st.session_state.chat_history = []
                st.rerun()

# ============================================================
# 챗봇
# ============================================================

def search_rag_notices(query: str, category_filter: str = "전체", top_k: int = 5) -> list[dict]:
    try:
        pinecone_categories = (
            PINECONE_CATEGORY_MAP.get(category_filter, [category_filter])
            if category_filter != "전체"
            else None
        )
        return pinecone_hybrid_search(
            query=query,
            top_k=top_k,
            category_filter=pinecone_categories,
        )
    except Exception as e:
        st.error(f"[Pinecone 검색 오류] {e}")
        return []

def render_chatbot(profile):
    with st.container(border=True):
        st.markdown('<div class="mac-bar"><div class="mac-dot mac-dot-red"></div><div class="mac-dot mac-dot-yellow"></div><div class="mac-dot mac-dot-green"></div></div><div style="margin-top:12px;"></div>', unsafe_allow_html=True)
        if not st.session_state.chat_history:
            name = profile.get("name",""); hsu_b64 = get_hsu_base64()
            hsu_img = f'<img src="data:image/png;base64,{hsu_b64}" style="width:60px;height:60px;object-fit:contain;display:block;margin:0 auto 14px auto;">' if hsu_b64 else '<div style="font-size:38px;margin-bottom:14px;">🔍</div>'
            st.markdown(f'<div style="text-align:center;padding:24px 0 20px;color:#86868b;">{hsu_img}<div style="font-size:17px;font-weight:600;color:#1d1d1f;margin-bottom:6px;">안녕하세요, {name}님!</div><div style="font-size:13px;line-height:1.6;">한성대 공지를 자연어로 검색해보세요.<br/><span style="color:#aeaeb2;">"장학금 신청 기간 알려줘" &nbsp;·&nbsp; "취업박람회 언제야?" &nbsp;·&nbsp; "비교과 프로그램 추천해줘"</span></div></div>', unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f'<div class="chat-bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-bubble-bot">{msg["content"]}</div>', unsafe_allow_html=True)
                    if msg.get("results"):
                        for idx, r in enumerate(msg["results"][:3], start=1):
                            title_safe = html.escape(str(r.get("title", "")))
                            cat_safe   = html.escape(str(r.get("category", "기타")))
                            date_safe  = html.escape(str(r.get("date", "")))
                            url_safe   = html.escape(str(r.get("url", "#")), quote=True)
                            st.markdown(
                                '<div class="notice-card">'
                                f'<span style="font-size:11px;color:#86868b;font-weight:500;">참고 공지 Top{idx}</span>&nbsp;'
                                f'<span class="notice-tag">{cat_safe}</span>'
                                f'<span class="notice-date">{date_safe}</span>'
                                f'<div class="notice-title">{title_safe}</div>'
                                '<div style="margin-top:8px;">'
                                f'<a href="{url_safe}" target="_blank" style="font-size:12px;color:#0a84ff;text-decoration:none;font-weight:600;">공지 바로가기 →</a>'
                                '</div>'
                                '</div>',
                                unsafe_allow_html=True,
                            )

    with st.form("chat_form", clear_on_submit=True):
        c0, c1, c2 = st.columns([1.2, 4, 0.8])
        with c0: cat_filter = st.selectbox("카테고리", ["전체"]+CATEGORIES, label_visibility="collapsed", key="chat_cat")
        with c1: user_input = st.text_input("메시지", placeholder="무엇이 궁금하세요?", label_visibility="collapsed")
        with c2: submitted  = st.form_submit_button("전송", use_container_width=True)

    if submitted and user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        cat_filter = st.session_state.get("chat_cat", "전체")
        results = search_rag_notices(user_input, cat_filter, top_k=5)
        with st.spinner("답변 생성 중..."):
            reply = generate_llm_reply(user_input, results, st.session_state.profile, is_first=len(st.session_state.chat_history)==1)
        st.session_state.chat_history.append({"role": "bot", "content": reply, "results": results})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("대화 초기화"): st.session_state.chat_history = []; st.rerun()

# ============================================================
# 추천 게시물
# ============================================================

def render_notice_card(notice):
    jts        = classify_job_type(notice)
    job_str    = " · ".join([t['job_type'] for t in jts]) if jts else ""
    summary    = summarize_notice(notice.get('title',''), notice.get('body',''))
    title_safe = html.escape(notice.get('title', ''))
    cat_safe   = html.escape(notice.get('category', '기타'))
    date_val   = re.sub(r'<[^>]+>', '', str(notice.get('date', notice.get('posted_at','')))).strip()[:10]
    job_html   = f'<span style="font-size:11px;color:#86868b;">{html.escape(job_str)}</span>' if job_str else ""
    sum_html   = f'<div class="notice-summary">{summary}</div>' if summary else ""

    st.markdown(
        '<div class="notice-card">'
        '<div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">'
        f'<span class="notice-tag">{cat_safe}</span>{job_html}'
        f'<span class="notice-date" style="margin-left:auto;">{date_val}</span>'
        '</div>'
        f'<div class="notice-title">{title_safe}</div>'
        f'{sum_html}'
        '<div style="margin-top:10px;display:flex;align-items:center;justify-content:space-between;">'
        f'<a href="{notice.get("url","#")}" target="_blank" style="font-size:12px;color:#0a84ff;text-decoration:none;font-weight:600;">공지 바로가기 →</a>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )


def render_recommend(profile):
    st.markdown(
        f'<div style="background:white;border-radius:14px;padding:16px 20px;margin-bottom:20px;'
        f'box-shadow:0 1px 5px rgba(0,0,0,0.06);display:flex;align-items:center;gap:16px;">'
        f'<div style="font-size:30px;">🎓</div>'
        f'<div><div style="font-size:15px;font-weight:700;color:#1d1d1f;">'
        f'{profile.get("college","")} &nbsp;·&nbsp; {profile.get("track","")} &nbsp;·&nbsp; {profile.get("grade","")}'
        f'</div><div style="font-size:13px;color:#86868b;margin-top:3px;">'
        f'관심사: {", ".join(profile.get("interests",[])) or "없음"}'
        f'</div></div></div>',
        unsafe_allow_html=True
    )

    col_l, col_c, col_r = st.columns([2, 1.5, 2])
    with col_c:
        btn_rec = st.button("맞춤 공지 추천받기", type="primary", use_container_width=True)

    if btn_rec:
        scholarship_results  = []
        scholarship_related  = []
        dorm_results         = []
        rotc_results         = []
        interests = profile.get('interests', [])

        # ── Two-Tower 추천 (필터링 카테고리 제외) ─────────────
        FILTER_CATS  = ["장학금", "기숙사", "ROTC"]
        rec_interests = [i for i in interests if i not in FILTER_CATS]

        with st.spinner("추천 중..."):
            recs = two_tower_recommend(
                college   = profile.get('college', ''),
                track     = profile.get('track', ''),
                year      = profile.get('grade', ''),
                interests = rec_interests,
                top_k     = 10,
            ) if rec_interests else []

        # ── 장학금 필터링 ──────────────────────────────────────
        if "장학금" in interests:
            with st.spinner("장학금 필터링 중..."):
                scholarship_results, scholarship_related = filter_scholarships(profile)

        # ── 기숙사 필터링 ──────────────────────────────────────
        if "기숙사" in interests:
            with st.spinner("기숙사 필터링 중..."):
                dorm_results = filter_dormitory(profile)

        # ── ROTC 필터링 ────────────────────────────────────────
        if "ROTC" in interests and profile.get('rotc_interest', False):
            with st.spinner("ROTC 공지 로딩 중..."):
                rotc_results = filter_rotc()

        total = len(recs) + len(scholarship_results) + len(scholarship_related) + len(dorm_results) + len(rotc_results)
        if total == 0:
            st.info("추천 결과가 없습니다.")
            return

        # ── 맞춤 추천 출력 ─────────────────────────────────────
        if recs:
            st.markdown(f"<div style='font-size:14px;font-weight:600;color:#1d1d1f;margin:16px 0 8px;'>🎯 맞춤 추천 공지 ({len(recs)}개)</div>", unsafe_allow_html=True)
            for rec in recs:
                render_notice_card(rec['notice'])

        # ── 장학금 신청 공지 ───────────────────────────────────
        if scholarship_results:
            st.markdown(f"<div style='font-size:14px;font-weight:600;color:#1d1d1f;margin:16px 0 8px;'>💰 신청 가능한 장학금 ({len(scholarship_results)}개)</div>", unsafe_allow_html=True)
            for notice in scholarship_results:
                render_notice_card(notice)

        # ── 장학금 관련 공지 ───────────────────────────────────
        if scholarship_related:
            st.markdown(f"<div style='font-size:14px;font-weight:600;color:#1d1d1f;margin:16px 0 8px;'>📢 장학금 관련 공지 ({len(scholarship_related)}개)</div>", unsafe_allow_html=True)
            for notice in scholarship_related:
                render_notice_card(notice)

        # ── 기숙사 출력 ────────────────────────────────────────
        if dorm_results:
            st.markdown(f"<div style='font-size:14px;font-weight:600;color:#1d1d1f;margin:16px 0 8px;'>🏠 기숙사 ({len(dorm_results)}개)</div>", unsafe_allow_html=True)
            for notice in dorm_results:
                render_notice_card(notice)

        # ── ROTC 출력 ──────────────────────────────────────────
        if rotc_results:
            st.markdown(f"<div style='font-size:14px;font-weight:600;color:#1d1d1f;margin:16px 0 8px;'>🎖️ ROTC ({len(rotc_results)}개)</div>", unsafe_allow_html=True)
            for notice in rotc_results:
                render_notice_card(notice)

# ============================================================
# 엔트리포인트
# ============================================================

def main():
    from PIL import Image
    try: hsu_icon = Image.open(os.path.join(_BASE_DIR, "hsu.png"))
    except: hsu_icon = "🔍"

    st.set_page_config(page_title="상상파인더", page_icon=hsu_icon, layout="wide", initial_sidebar_state="expanded")

    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    if "notices"      not in st.session_state: st.session_state.notices      = []
    if "onboarded" not in st.session_state:
        st.session_state.profile  = {}
        st.session_state.onboarded = False

    if not st.session_state.onboarded:
        render_onboarding(); return

    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    profile = st.session_state.profile

    if not st.session_state.notices:
        with st.spinner("공지 불러오는 중..."):
            notices = load_notices_from_supabase()
        if notices:
            st.session_state.notices = notices

    render_sidebar(profile)
    tab_chat, tab_rec = st.tabs(["  💬 챗봇 검색  ", "  ✨ 추천 게시물  "])
    with tab_chat: render_chatbot(profile)
    with tab_rec:  render_recommend(profile)

if __name__ == "__main__":
    main()
