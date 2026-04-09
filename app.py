
# ============================================================
# app.py — 상상파인더 (온보딩 → 사이드바 + 챗봇/추천게시물)
# ============================================================

import os, re, time, json, hashlib, warnings
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urlencode, parse_qs, urlunparse
from datetime import datetime

import streamlit as st

warnings.filterwarnings("ignore")

# ── 설정 ──────────────────────────────────────────────────────────────
EMBED_MODEL_PATH    = "/Users/dohyun/Desktop/캡스톤/0407/models/embed_finetuned"
SUMMARY_MODEL_PATH  = "/Users/dohyun/Desktop/캡스톤/0407/models/summary_finetuned"
CLASSIFY_MODEL_PATH = "/Users/dohyun/Desktop/캡스톤/0407/models/classify_finetuned"
BASE_MODEL_EMBED    = "jhgan/ko-sroberta-multitask"
CHROMA_DB_PATH      = "/Users/dohyun/Desktop/캡스톤/0407/chroma_db"
NOTICES_CACHE_PATH  = "/Users/dohyun/Desktop/캡스톤/0407/data/notices_cache.json"

BOARD_LIST_URL = "https://www.hansung.ac.kr/bbs/hansung/2127/artclList.do"
HEADERS        = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
TARGET_YEAR    = str(datetime.now().year)

CATEGORIES = ["장학", "비교과", "수업", "취업", "기타"]

CATEGORY_KEYWORDS = {
    "장학":   ["장학", "등록금", "지원금", "장학생", "장학금"],
    "비교과": ["비교과", "프로그램", "특강", "세미나", "경진대회", "챌린지", "동아리"],
    "수업":   ["수업", "강의", "학사", "수강", "시간표", "휴강", "과목", "강좌"],
    "취업":   ["취업", "채용", "인턴", "박람회", "직무", "기업", "공채", "면접"],
}

_CATEGORY_PATTERN = re.compile(
    r"^(한성공지|국제|학사|비교과|장학|취업|진로|창업|기타|현장실습|교육프로그램|행사|일반공지)\s*"
)
_SUFFIX_PATTERN = re.compile(r"\s*(새글|hot|NEW)\s*$", re.IGNORECASE)

os.makedirs("/Users/dohyun/Desktop/캡스톤/0407/data", exist_ok=True)
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

# 단과대 / 트랙·학과 매핑
COLLEGE_MAP = {
    "크리에이티브인문예술대학" : ["영미문화콘텐츠트랙", "영미언어정보트랙", "한국어교육트랙", "역사문화큐레이션트랙", "역사콘텐츠트랙", "지식정보문화트랙", "디지털인문정보학트랙","동양화전공", "서양화전공", "한국무용전공","현대무용전공","발레전공"],
    "미래융합사회과학대학": ["국제무역트랙", "글로벌비지니스트랙", "기업ㆍ경제분석트랙", "경제금융투자트랙","공공행정트랙","법&정책트랙", "부동산트랙", "스마트도시ㆍ교통계획트랙","기업경영트랙","비지니스애널리틱스트랙","회계ㆍ재무경영트랙"],
    "디자인대학" :["패션마케팅트랙", "패션디자인트랙","패션크리에이티브디렉션트랙","미디어디자인트랙","시각디자인트랙","영상ㆍ애니메이션디자인트랙","UX/UI디자인트랙","인테리어디자인트랙","VMDㆍ전시디자인트랙","게임그래픽디자인트랙","뷰티디자인매니지먼트학과"],
    "IT공과대학": ["모바일소프트웨어트랙", "빅데이터트랙", "디지털콘텐츠ㆍ가상현실트랙", "웹공학트랙", "전자트랙","시스템반도체트랙","기계시스템디자인트랙","AI로봇융합트랙","산업공학트랙","응용산업데이터공학트랙"],
    "창의융합대학": ["상상력인재학부", "문학문화콘텐츠학과", "AI응용학과","융합보안학과","미래모빌리티학과"],
    "글로벌인재대학" : ["한국언어문화교육학과","글로벌K비지니스학과","영상엔터테인먼트학과","패션뷰티크리에이션학과","SW융합학과","글로벌벤처창업학과"],
    "미래플러스대학" : ["융합행정학과", "호텔외식경영학과","뷰티디자인학과","비지니스컨설팅학과","ICT융합디자인학과","AIㆍ소프트웨어학과","뷰티매니지먼트학과","디지털콘텐츠디자인학과","인테리어디자인학과","스마트제조혁신컨설팅학과"]
}


# ============================================================
# 유틸
# ============================================================

def clean_url(url: str) -> str:
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    params.pop("layout", None)
    new_query = urlencode({k: v[0] for k, v in params.items()})
    return urlunparse(parsed._replace(query=new_query))


def clean_title(raw: str) -> str:
    title = raw.replace("\n", " ").replace("\r", " ")
    title = re.sub(r"\s{2,}", " ", title).strip()
    title = _CATEGORY_PATTERN.sub("", title).strip()
    title = _SUFFIX_PATTERN.sub("", title).strip()
    return title


def infer_category(title: str, body: str) -> str:
    text = title + " " + body
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return cat
    return "기타"


def tokenize_ko(text: str) -> list:
    return re.findall(r"[\w가-힣]+", text.lower())


# ============================================================
# 크롤러
# ============================================================

def _ocr_image(img_url: str) -> str:
    try:
        import pytesseract
        from PIL import Image
        from io import BytesIO
        import platform
        if platform.system() == "Windows":
            pytesseract.pytesseract.tesseract_cmd = (
                r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            )
        res = requests.get(img_url, headers=HEADERS, timeout=10)
        res.raise_for_status()
        img  = Image.open(BytesIO(res.content))
        text = pytesseract.image_to_string(img, lang="kor+eng")
        return text.strip()
    except Exception as e:
        print(f"    ⚠️ OCR 실패 ({img_url[:50]}): {e}")
        return ""


def _extract_pdf(pdf_url: str) -> str:
    try:
        import pdfplumber
        from io import BytesIO
        res = requests.get(pdf_url, headers=HEADERS, timeout=15)
        res.raise_for_status()
        with pdfplumber.open(BytesIO(res.content)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages[:10]]
        return " ".join(pages).strip()
    except Exception as e:
        print(f"    ⚠️ PDF 추출 실패 ({pdf_url[:50]}): {e}")
        return ""


def get_post_content(url: str) -> str:
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        div  = soup.select_one(".txt")
        parts = []
        if div:
            text = div.get_text(" ", strip=True)
            if text:
                parts.append(text)
            for img in div.find_all("img"):
                src = img.get("src", "")
                if not src:
                    continue
                if src.startswith("/"):
                    src = "https://www.hansung.ac.kr" + src
                ocr_text = _ocr_image(src)
                if ocr_text:
                    parts.append(f"[이미지 OCR] {ocr_text}")
        BASE = "https://www.hansung.ac.kr"
        for a in soup.find_all("a", href=True):
            href     = a["href"]
            filename = a.get_text(strip=True).lower()
            if "download.do" in href and filename.endswith(".pdf"):
                pdf_url  = BASE + href if href.startswith("/") else href
                pdf_text = _extract_pdf(pdf_url)
                if pdf_text:
                    parts.append(f"[첨부PDF] {pdf_text[:1000]}")
        return " ".join(parts)
    except Exception as e:
        print(f"  ⚠️ 본문 크롤링 실패: {e}")
        return ""


def get_list_page(page: int):
    try:
        res = requests.get(BOARD_LIST_URL, params={"page": page},
                           headers=HEADERS, timeout=10)
        res.raise_for_status()
        soup  = BeautifulSoup(res.text, "html.parser")
        items = []
        for tr in soup.find_all("tr"):
            if not tr.find_all("td"):
                continue
            tr_classes = tr.get("class") or []
            if "notice" in tr_classes:
                continue
            date_el = tr.select_one(".td-date")
            link_el = tr.select_one(".td-title a")
            if not date_el or not link_el:
                continue
            date_text = date_el.get_text(strip=True)
            if not date_text.startswith(TARGET_YEAR):
                if page >= 5:
                    return items, True
                else:
                    continue
            href = link_el.get("href", "")
            if href.startswith("/"):
                href = "https://www.hansung.ac.kr" + href
            items.append({
                "title": clean_title(link_el.get_text()),
                "url":   clean_url(href),
                "date":  date_text,
            })
        return items, False
    except Exception as e:
        print(f"  ⚠️ 목록 파싱 실패 (page={page}): {e}")
        return [], False


def crawl_all() -> list:
    all_items, page = [], 1
    while True:
        items, done = get_list_page(page)
        if items:
            all_items.extend(items)
        if done or not items:
            break
        page += 1
        time.sleep(0.3)
    for item in all_items:
        item["body"]     = get_post_content(item["url"])
        item["category"] = classify_notice(item["title"], item["body"])
        time.sleep(0.2)
    with open(NOTICES_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)
    return all_items


def load_notices_cache() -> list:
    if os.path.exists(NOTICES_CACHE_PATH):
        with open(NOTICES_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


# ============================================================
# 모델 로더
# ============================================================

@st.cache_resource
def get_embed_model():
    path = EMBED_MODEL_PATH if os.path.exists(EMBED_MODEL_PATH) else BASE_MODEL_EMBED
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(path, device="cpu")


@st.cache_resource
def get_summary_pipeline():
    if not os.path.exists(SUMMARY_MODEL_PATH):
        return None
    from transformers import pipeline
    return pipeline("summarization", model=SUMMARY_MODEL_PATH,
                    tokenizer=SUMMARY_MODEL_PATH, max_new_tokens=128, device=-1)


@st.cache_resource
def get_classifier():
    if not os.path.exists(CLASSIFY_MODEL_PATH):
        return None, None
    from transformers import pipeline
    clf = pipeline("text-classification", model=CLASSIFY_MODEL_PATH,
                   tokenizer=CLASSIFY_MODEL_PATH, device=-1)
    label_map_path = f"{CLASSIFY_MODEL_PATH}/label_map.json"
    label_map = {}
    if os.path.exists(label_map_path):
        with open(label_map_path) as f:
            label_map = json.load(f)
    return clf, label_map


@st.cache_resource
def get_chroma():
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client.get_or_create_collection(
        name="hansung_notices",
        metadata={"hnsw:space": "cosine"}
    )


# ============================================================
# 분류
# ============================================================

def classify_notice(title: str, body: str) -> str:
    clf, label_map = get_classifier()
    if clf is None:
        return infer_category(title, body)
    try:
        result   = clf(f"{title} {body[:200]}", truncation=True)[0]
        label_id = result["label"].replace("LABEL_", "")
        return label_map.get(label_id, "기타")
    except Exception:
        return infer_category(title, body)


# ============================================================
# ChromaDB 임베딩 & 저장
# ============================================================

def index_notices(notices: list):
    model      = get_embed_model()
    collection = get_chroma()
    new_count  = update_count = 0
    for item in notices:
        doc_id    = hashlib.md5(item["url"].encode()).hexdigest()
        body      = item.get("body", "")
        category  = classify_notice(item["title"], body)
        text      = f"제목: {item['title']}\n\n{body}"
        embedding = model.encode(text).tolist()
        existing  = collection.get(ids=[doc_id])["ids"]
        if existing:
            collection.update(
                ids=[doc_id], embeddings=[embedding], documents=[text],
                metadatas=[{"title": item["title"], "url": item["url"],
                            "date": item["date"], "category": category}]
            )
            update_count += 1
        else:
            collection.add(
                ids=[doc_id], embeddings=[embedding], documents=[text],
                metadatas=[{"title": item["title"], "url": item["url"],
                            "date": item["date"], "category": category}]
            )
            new_count += 1


# ============================================================
# 하이브리드 검색
# ============================================================

@st.cache_data(ttl=600, show_spinner=False)
def _build_bm25_index(category_filter: str):
    from rank_bm25 import BM25Okapi
    collection = get_chroma()
    where      = {"category": category_filter} \
                 if category_filter and category_filter != "전체" else None
    all_data   = collection.get(include=["documents", "metadatas"], where=where)
    documents  = all_data["documents"]
    metadatas  = all_data["metadatas"]
    ids        = all_data["ids"]
    if not documents:
        return None, [], [], []
    tokenized_docs = [tokenize_ko(doc) for doc in documents]
    bm25           = BM25Okapi(tokenized_docs)
    return bm25, ids, documents, metadatas


def hybrid_search(query: str, top_k: int = 5, alpha: float = 0.7,
                  category_filter: str = None) -> list:
    model      = get_embed_model()
    collection = get_chroma()
    cat_key    = category_filter if category_filter and category_filter != "전체" else "전체"
    where      = {"category": category_filter} \
                 if category_filter and category_filter != "전체" else None
    bm25, ids, documents, metadatas = _build_bm25_index(cat_key)
    if bm25 is None:
        return []
    q_emb     = model.encode(query).tolist()
    n_results = min(top_k * 2, len(documents))
    vr        = collection.query(
        query_embeddings=[q_emb], n_results=n_results,
        include=["metadatas", "distances"], where=where,
    )
    vector_scores = {}
    raw_dist = vr["distances"][0]
    if raw_dist:
        max_sim = 1 - min(raw_dist)
        min_sim = 1 - max(raw_dist)
        for vid, dist in zip(vr["ids"][0], raw_dist):
            sim  = 1 - dist
            norm = (sim - min_sim) / (max_sim - min_sim + 1e-9)
            vector_scores[vid] = norm
    bm25_raw    = bm25.get_scores(tokenize_ko(query))
    bm25_max    = max(bm25_raw) if max(bm25_raw) > 0 else 1
    bm25_scores = {did: s / bm25_max for did, s in zip(ids, bm25_raw)}
    all_ids = set(vector_scores) | set(bm25_scores)
    final   = {
        did: alpha * vector_scores.get(did, 0) + (1 - alpha) * bm25_scores.get(did, 0)
        for did in all_ids
    }
    top_ids  = sorted(final, key=lambda x: final[x], reverse=True)[:top_k]
    meta_map = dict(zip(ids, metadatas))
    return [
        {**meta_map[did], "score": round(final[did], 4)}
        for did in top_ids if did in meta_map
    ]


# ============================================================
# 요약
# ============================================================

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


# ============================================================
# 콘텐츠 기반 추천
# ============================================================

def recommend_notices(user_profile: dict, top_k: int = 5) -> list:
    model      = get_embed_model()
    collection = get_chroma()
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
        n_results=min(top_k, n_docs),
        include=["metadatas", "distances"],
    )
    items = []
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        score = round(1 - dist, 4)
        if meta.get("category") in user_profile.get("interests", []):
            score = min(score + 0.05, 1.0)
        items.append({**meta, "score": score})
    items.sort(key=lambda x: x["score"], reverse=True)
    return items


# ============================================================
# CSS
# ============================================================

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #f2f2f7 !important;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display",
                 "Noto Sans KR", sans-serif !important;
}

/* 사이드바 */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[kind="header"],
.st-emotion-cache-h5rgaw,
[data-testid="stSidebar"] > div:first-child > div:first-child button { display: none !important; }
section[data-testid="stSidebar"] {
    width: 260px !important;
    min-width: 260px !important;
    transform: translateX(0) !important;
    visibility: visible !important;
    background: #ffffff !important;
    border-right: 1px solid rgba(0,0,0,0.08) !important;
}
section[data-testid="stSidebar"] > div:first-child {
    width: 260px !important;
    padding: 24px 16px !important;
}
[data-testid="stSidebar"] * {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text",
                 "Noto Sans KR", sans-serif !important;
}

/* 버튼 */
.stButton > button {
    background: #0a84ff !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
    transition: all 0.15s ease !important;
    letter-spacing: -0.01em !important;
}
.stButton > button:hover {
    background: #409cff !important;
    transform: scale(1.01) !important;
}

/* 인풋 */
.stTextInput > div > div > input {
    background: white !important;
    border: 1px solid rgba(0,0,0,0.12) !important;
    border-radius: 12px !important;
    font-size: 15px !important;
    padding: 12px 16px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
}
.stTextInput > div > div > input:focus {
    border-color: #0a84ff !important;
    box-shadow: 0 0 0 3px rgba(10,132,255,0.15) !important;
    outline: none !important;
}

/* 셀렉트박스 */
.stSelectbox > div > div {
    background: white !important;
    border-radius: 10px !important;
    border: 1px solid rgba(0,0,0,0.12) !important;
}

/* 멀티셀렉트 */
.stMultiSelect > div > div {
    background: white !important;
    border-radius: 10px !important;
    border: 1px solid rgba(0,0,0,0.12) !important;
}
/* 멀티셀렉트 체크박스 파란색 */
.stMultiSelect [data-baseweb="checkbox"] svg { color: #0a84ff !important; fill: #0a84ff !important; }
.stMultiSelect [aria-selected="true"] { background: rgba(10,132,255,0.08) !important; color: #0a84ff !important; }
/* 라디오 파란색 */
.stRadio [data-baseweb="radio"] div { border-color: #0a84ff !important; }
.stRadio [data-baseweb="radio"] [data-checked="true"] div { background: #0a84ff !important; border-color: #0a84ff !important; }
/* 전체 accent 빨강 → 파란색 override (config.toml 없이도 동작) */
:root {
    --primary-color: #0a84ff !important;
    --primary-hue: 211 !important;
}
/* 라디오 버튼 */
[data-baseweb="radio"] [data-checked="true"] > div:first-child {
    background-color: #0a84ff !important;
    border-color: #0a84ff !important;
}
[data-baseweb="radio"] > div:first-child {
    border-color: #0a84ff !important;
}
/* 멀티셀렉트 태그 */
[data-baseweb="tag"] { background-color: rgba(10,132,255,0.12) !important; }
[data-baseweb="tag"] span { color: #0a84ff !important; }
/* 체크박스 */
[data-baseweb="checkbox"] [data-checked="true"] > div {
    background-color: #0a84ff !important;
    border-color: #0a84ff !important;
}
/* 포커스 링 */
*:focus-visible { outline-color: #0a84ff !important; }
/* 탭 underline */
[data-baseweb="tab-highlight"] { background-color: #0a84ff !important; }
/* 프로그레스/스피너 */
[data-testid="stProgress"] > div > div { background-color: #0a84ff !important; }
/* 슬라이더 */
[data-baseweb="slider"] [role="slider"] { background-color: #0a84ff !important; border-color: #0a84ff !important; }
[data-baseweb="slider"] div[data-testid="stSliderTrackFill"] { background-color: #0a84ff !important; }

/* 라디오 */
.stRadio > div { gap: 4px !important; }

/* 탭 */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(120,120,128,0.12) !important;
    border-radius: 12px !important;
    padding: 3px !important;
    gap: 2px !important;
    width: fit-content !important;
    margin: 0 auto 16px auto !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #3c3c43 !important;
    padding: 8px 20px !important;
    font-family: -apple-system, "SF Pro Text", "Noto Sans KR", sans-serif !important;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    color: #1d1d1f !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.12) !important;
    font-weight: 600 !important;
}
/* 탭 하단 액센트 라인 파스텔 블루로 */
.stTabs [data-baseweb="tab-highlight"] {
    background-color: #7ab8f5 !important;
}
.stTabs [data-baseweb="tab-border"] {
    background-color: transparent !important;
}

/* 채팅 */
.chat-bubble-user {
    background: #0a84ff;
    color: white;
    border-radius: 20px 20px 5px 20px;
    padding: 12px 18px;
    max-width: 72%;
    margin-left: auto;
    margin-bottom: 10px;
    font-size: 15px;
    line-height: 1.55;
    word-break: break-word;
}
.chat-bubble-bot {
    background: white;
    color: #1d1d1f;
    border-radius: 20px 20px 20px 5px;
    padding: 12px 18px;
    max-width: 80%;
    margin-bottom: 10px;
    font-size: 15px;
    line-height: 1.55;
    box-shadow: 0 1px 5px rgba(0,0,0,0.08);
    word-break: break-word;
}
/* chat container - border=True 컨테이너 스타일 */
[data-testid="stVerticalBlockBorderWrapper"]:has(.chat-bubble-user),
[data-testid="stVerticalBlockBorderWrapper"]:has(.chat-bubble-bot) {
    background: #f9f9fb !important;
    border-radius: 18px !important;
    min-height: 340px !important;
    border: 1px solid rgba(0,0,0,0.06) !important;
}

/* 공지 카드 */
.notice-card {
    background: white;
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 12px;
    box-shadow: 0 1px 5px rgba(0,0,0,0.06);
    border: 1px solid rgba(0,0,0,0.05);
    transition: box-shadow 0.15s ease;
}
.notice-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.10); }
.notice-tag {
    display: inline-block;
    background: rgba(10,132,255,0.1);
    color: #0a84ff;
    border-radius: 7px;
    padding: 2px 9px;
    font-size: 11px;
    font-weight: 700;
    margin-right: 8px;
    letter-spacing: 0.02em;
}
.notice-title {
    font-size: 15px;
    font-weight: 600;
    color: #1d1d1f;
    margin: 6px 0 2px 0;
    line-height: 1.4;
}
.notice-date { font-size: 12px; color: #86868b; }
.notice-summary {
    font-size: 13px;
    color: #3c3c43;
    margin-top: 8px;
    line-height: 1.55;
}

/* 사이드바 섹션 레이블 */
.sb-label {
    font-size: 10px;
    font-weight: 700;
    color: rgba(0,0,0,0.35);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 18px 0 6px 2px;
}
.sb-info-row {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 5px 0;
}
.sb-info-key {
    font-size: 12px;
    color: rgba(0,0,0,0.4);
    min-width: 52px;
}
.sb-info-val {
    font-size: 13px;
    font-weight: 500;
    color: #1d1d1f;
    line-height: 1.4;
}

/* 온보딩 카드 — column 컨테이너에 직접 적용 */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: white !important;
    border-radius: 20px !important;
    padding: 32px 36px !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08) !important;
    border: none !important;
}

/* 구분선 */
hr { border: none; border-top: 1px solid rgba(0,0,0,0.08) !important; margin: 14px 0 !important; }

/* 숨기기 */
#MainMenu, footer, header { visibility: hidden; }
</style>
"""


# ============================================================
# 온보딩 화면
# ============================================================

def render_onboarding():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    # 전체 화면 중앙 정렬
    _, center, _ = st.columns([1, 2.2, 1])
    with center:
        st.markdown("<div style='height:60px'></div>", unsafe_allow_html=True)

        # 로고
        st.markdown("""
<div style="text-align:center; margin-bottom:32px;">
  <div style="display:inline-flex; align-items:center; justify-content:center;
       width:64px; height:64px; background:linear-gradient(135deg,#0a84ff,#34aadc);
       border-radius:18px; font-size:30px; margin-bottom:14px; box-shadow:0 4px 16px rgba(10,132,255,0.35);">
    🔍
  </div>
  <div style="font-size:26px; font-weight:700; color:#1d1d1f; letter-spacing:-0.03em;">상상파인더</div>
  <div style="font-size:14px; color:#86868b; margin-top:4px;">한성대 공지 스마트 검색</div>
</div>
""", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("""
<div style="font-size:18px; font-weight:700; color:#1d1d1f; margin-bottom:4px;">반갑습니다 👋</div>
<div style="font-size:14px; color:#86868b; margin-bottom:8px;">먼저 기본 정보를 알려주세요.</div>
""", unsafe_allow_html=True)

            name = st.text_input("이름", placeholder="홍길동", key="ob_name")

            college = st.selectbox(
                "단과대",
                list(COLLEGE_MAP.keys()),
                key="ob_college"
            )

            track_options = COLLEGE_MAP.get(college, ["기타"])
            track = st.selectbox("트랙 / 학과", track_options, key="ob_track")

            grade = st.selectbox("학년", ["1학년", "2학년", "3학년", "4학년"], key="ob_grade")

            interests = st.multiselect(
                "관심사 (복수 선택 가능)",
                CATEGORIES + ["교환학생"],
                placeholder="관심 카테고리를 선택하세요",
                key="ob_interests"
            )

            extracurricular = st.radio(
                "비교과 점수를 채웠나요?",
                ["예, 채웠어요", "아직 안 채웠어요"],
                horizontal=True,
                key="ob_extra"
            )

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            if st.button("시작하기 →", use_container_width=True):
                if not name.strip():
                    st.markdown('<div style="background:#e8f1ff;color:#0a84ff;border-radius:10px;padding:10px 16px;font-size:14px;font-weight:500;border:1px solid #b3d1ff;">✏️ 이름을 입력해 주세요.</div>', unsafe_allow_html=True)
                else:
                    st.session_state.profile = {
                        "name": name.strip(),
                        "college": college,
                        "track": track,
                        "grade": grade,
                        "interests": interests,
                        "extracurricular": extracurricular == "예, 채웠어요",
                    }
                    st.session_state.onboarded = True
                    st.rerun()


# ============================================================
# 사이드바
# ============================================================

def render_sidebar(profile: dict):
    with st.sidebar:
        # 로고
        st.markdown("""
<div style="display:flex; align-items:center; gap:10px; padding-bottom:18px;">
  <div style="width:38px; height:38px; background:linear-gradient(135deg,#0a84ff,#34aadc);
       border-radius:11px; display:flex; align-items:center; justify-content:center;
       font-size:20px; flex-shrink:0; box-shadow:0 2px 8px rgba(10,132,255,0.4);">🔍</div>
  <div>
    <div style="font-size:16px; font-weight:700; color:#1d1d1f; letter-spacing:-0.02em;">상상파인더</div>
    <div style="font-size:11px; color:#86868b; margin-top:1px;">Hansung Notice Finder</div>
  </div>
</div>
<hr/>
""", unsafe_allow_html=True)

        # 사용자 정보
        st.markdown('<div class="sb-label">내 정보</div>', unsafe_allow_html=True)

        rows = [
            ("이름", profile.get("name", "")),
            ("단과대", profile.get("college", "")),
            ("학과", profile.get("track", "")),
            ("학년", profile.get("grade", "")),
        ]
        html = ""
        for key, val in rows:
            html += f"""
<div class="sb-info-row">
  <span class="sb-info-key">{key}</span>
  <span class="sb-info-val">{val}</span>
</div>"""
        st.markdown(html, unsafe_allow_html=True)

        st.markdown("<hr/>", unsafe_allow_html=True)

        # 바로가기
        st.markdown('<div class="sb-label">바로가기</div>', unsafe_allow_html=True)
        links = [
            ("🏫", "한성대학교", "https://www.hansung.ac.kr/hansung/index.do"),
            ("💻", "한성 e-class", "https://learn.hansung.ac.kr/"),
            ("📋", "종합정보시스템", "https://info.hansung.ac.kr/"),
            ("📊", "스마트자기관리시스템", "https://hsportal.hansung.ac.kr/"),
            ("📚", "학술정보관", "https://hsel.hansung.ac.kr/"),
        ]
        link_html = ""
        for icon, label, url in links:
            link_html += f"""
<a href="{url}" target="_blank" style="
    display:flex; align-items:center; gap:9px;
    padding:8px 10px; border-radius:9px;
    text-decoration:none; color:#1d1d1f;
    font-size:13px; font-weight:500;
    transition:background 0.15s;
    margin-bottom:2px;
" onmouseover="this.style.background='rgba(0,0,0,0.05)'"
  onmouseout="this.style.background='transparent'">
  <span style="font-size:15px;">{icon}</span>
  <span>{label}</span>
  <span style="margin-left:auto; font-size:11px; color:#aeaeb2;">↗</span>
</a>"""
        st.markdown(link_html, unsafe_allow_html=True)




# ============================================================
# 메인 화면 — 챗봇
# ============================================================

def render_chatbot(profile: dict):
    top_k = 5
    alpha = 0.7

    # 채팅 영역 — st.container로 감싸서 CSS 적용
    with st.container(border=True):
        if not st.session_state.chat_history:
            name = profile.get("name", "")
            st.markdown(f"""
<div style="text-align:center; padding:48px 0 36px; color:#86868b;">
  <div style="font-size:38px; margin-bottom:14px;">🔍</div>
  <div style="font-size:17px; font-weight:600; color:#1d1d1f; margin-bottom:6px;">
    안녕하세요, {name}님!
  </div>
  <div style="font-size:13px; line-height:1.6;">
    우리 대학의 공지를 검색해보세요.<br/>
    <span style="color:#aeaeb2;">"장학금 신청 기간 알려줘" &nbsp;·&nbsp; "취업박람회 언제야?" &nbsp;·&nbsp; "비교과 프로그램 추천해줘"</span>
  </div>
</div>
""", unsafe_allow_html=True)
        else:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(
                        f'<div class="chat-bubble-user">{msg["content"]}</div>',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<div class="chat-bubble-bot">{msg["content"]}</div>',
                        unsafe_allow_html=True)
                    if msg.get("results"):
                        for r in msg["results"]:
                            body_map = {n["url"]: n.get("body", "")
                                        for n in (st.session_state.notices or load_notices_cache())}
                            body    = body_map.get(r["url"], "")
                            summary = summarize_notice(r["title"], body) if body else ""
                            st.markdown(f"""
<div class="notice-card">
  <span class="notice-tag">{r.get('category','기타')}</span>
  <span class="notice-date">{r['date']}</span>
  <div class="notice-title">{r['title']}</div>
  {"<div class='notice-summary'>" + summary + "</div>" if summary else ""}
  <div style="margin-top:10px;">
    <a href="{r['url']}" target="_blank"
       style="font-size:12px;color:#0a84ff;text-decoration:none;font-weight:600;">
      공지 바로가기 →
    </a>
  </div>
</div>
""", unsafe_allow_html=True)

    # 입력 폼: 카테고리 | 입력창 | 전송
    with st.form("chat_form", clear_on_submit=True):
        c0, c1, c2 = st.columns([1.2, 4, 0.8])
        with c0:
            cat_filter = st.selectbox("카테고리", ["전체"] + CATEGORIES,
                                      label_visibility="collapsed", key="chat_cat")
        with c1:
            user_input = st.text_input(
                "메시지", placeholder="무엇이 궁금하세요?",
                label_visibility="collapsed")
        with c2:
            submitted = st.form_submit_button("전송", use_container_width=True)

    if submitted and user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        cat_filter = st.session_state.get("chat_cat", "전체")
        results = hybrid_search(
            user_input, top_k=top_k, alpha=alpha,
            category_filter=cat_filter if cat_filter != "전체" else None,
        )
        reply = f"{len(results)}개의 관련 공지를 찾았습니다." if results \
            else "관련 공지를 찾지 못했습니다. 캐시를 먼저 불러와 주세요."
        st.session_state.chat_history.append(
            {"role": "bot", "content": reply, "results": results})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("대화 초기화", use_container_width=False):
            st.session_state.chat_history = []
            st.rerun()


# ============================================================
# 메인 화면 — 추천 게시물
# ============================================================

def render_recommend(profile: dict):
    st.markdown(f"""
<div style="background:white; border-radius:14px; padding:16px 20px; margin-bottom:20px;
     box-shadow:0 1px 5px rgba(0,0,0,0.06); display:flex; align-items:center; gap:16px;">
  <div style="font-size:30px;">🎓</div>
  <div>
    <div style="font-size:15px; font-weight:700; color:#1d1d1f;">
      {profile.get('college','')} &nbsp;·&nbsp; {profile.get('track','')} &nbsp;·&nbsp; {profile.get('grade','')}
    </div>
    <div style="font-size:13px; color:#86868b; margin-top:3px;">
      관심사: {', '.join(profile.get('interests',[])) or '없음'}
      &nbsp;|&nbsp; 비교과: {'✅ 완료' if profile.get('extracurricular') else '⏳ 미완료'}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([2, 1.5, 2])
    with col_c:
        btn_rec = st.button("맞춤 공지 추천받기", type="primary", use_container_width=True)
    if btn_rec:
        with st.spinner("추천 중..."):
            recs = recommend_notices(profile, top_k=5)
        if not recs:
            st.info("추천 결과가 없습니다. 먼저 캐시를 불러와 주세요.")
        else:
            body_map = {n["url"]: n.get("body", "")
                        for n in (st.session_state.notices or load_notices_cache())}
            for r in recs:
                body    = body_map.get(r["url"], "")
                summary = summarize_notice(r["title"], body) if body else ""
                st.markdown(f"""
<div class="notice-card">
  <span class="notice-tag">{r.get('category','기타')}</span>
  <span class="notice-date">{r['date']}</span>
  <div class="notice-title">{r['title']}</div>
  {"<div class='notice-summary'>" + summary + "</div>" if summary else ""}
  <div style="margin-top:10px;">
    <a href="{r['url']}" target="_blank"
       style="font-size:12px;color:#0a84ff;text-decoration:none;font-weight:600;">
      공지 바로가기 →
    </a>
  </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 엔트리포인트
# ============================================================

def main():
    st.set_page_config(
        page_title="상상파인더",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 세션 초기화
    if "onboarded"    not in st.session_state: st.session_state.onboarded    = False
    if "profile"      not in st.session_state: st.session_state.profile      = {}
    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    if "notices"      not in st.session_state: st.session_state.notices      = []

    # ── 온보딩 전 ─────────────────────────────────────────────
    if not st.session_state.onboarded:
        render_onboarding()
        return

    # ── 온보딩 후 ─────────────────────────────────────────────
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    profile = st.session_state.profile

    # 캐시 자동 로드 (최초 1회)
    if not st.session_state.notices:
        notices = load_notices_cache()
        if notices:
            st.session_state.notices = notices
            # ChromaDB에 이미 데이터가 있으면 index_notices 생략 (속도 최적화)
            if get_chroma().count() == 0:
                index_notices(notices)
                _build_bm25_index.clear()

    render_sidebar(profile)

    # 메인 탭
    tab_chat, tab_rec = st.tabs(["  💬 챗봇 검색  ", "  ✨ 추천 게시물  "])

    with tab_chat:
        render_chatbot(profile)

    with tab_rec:
        render_recommend(profile)


if __name__ == "__main__":
    main()
