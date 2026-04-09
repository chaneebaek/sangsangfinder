# ============================================================
# [파일 2] app.py — 로컬 Streamlit 실행용
#
# 사전 준비:
#   1. train_colab.py 실행 후 saved_models/ 폴더를 이 파일과 같은 디렉토리에 복사
#
# 디렉토리 구조:
#   project/
#   ├── app.py
#   ├── models/              ← saved_models/ 안의 내용을 복사
#   │   ├── embed_finetuned/
#   │   └── classify_finetuned/
#   └── data/
#       └── notices_cache.json
#
# 실행:
#   pip install chromadb sentence-transformers requests \
#               beautifulsoup4 rank_bm25 streamlit transformers torch
#   streamlit run app.py
# ============================================================

import os, re, time, json, hashlib, warnings
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urlencode, parse_qs, urlunparse
from datetime import datetime

import streamlit as st

warnings.filterwarnings("ignore")

# ── 설정 ──────────────────────────────────────────────────────────────
EMBED_MODEL_PATH    = "/Users/dohyun/Desktop/캡스톤/qa_dataset_generation/models/embed_finetuned"
SUMMARY_MODEL_PATH  = "/Users/dohyun/Desktop/캡스톤/qa_dataset_generation/models/summary_finetuned"
CLASSIFY_MODEL_PATH = "/Users/dohyun/Desktop/캡스톤/qa_dataset_generation/models/classify_finetuned"
BASE_MODEL_EMBED    = "jhgan/ko-sroberta-multitask"
CHROMA_DB_PATH      = "/Users/dohyun/Desktop/캡스톤/qa_dataset_generation/chroma_db"
NOTICES_CACHE_PATH  = "/Users/dohyun/Desktop/캡스톤/qa_dataset_generation/data/notices_cache.json"

BOARD_LIST_URL = "https://www.hansung.ac.kr/bbs/hansung/2127/artclList.do"
HEADERS        = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
TARGET_YEAR    = str(datetime.now().year)              # ✅ fix #4: 연도 자동

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

os.makedirs("/Users/dohyun/Desktop/캡스톤/qa_dataset_generation/data", exist_ok=True)
os.makedirs(CHROMA_DB_PATH, exist_ok=True)


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
# 크롤러 — 본문 추출 (텍스트 / 이미지 OCR / PDF)
#
# pip install pdfplumber pytesseract Pillow
# macOS:  brew install tesseract tesseract-lang
# Ubuntu: sudo apt-get install tesseract-ocr tesseract-ocr-kor
# ============================================================

def _ocr_image(img_url: str) -> str:
    """이미지 URL → OCR 텍스트 (한국어)
    설치:
      macOS:   brew install tesseract tesseract-lang
      Ubuntu:  sudo apt-get install tesseract-ocr tesseract-ocr-kor
      Windows: Tesseract 설치 후 경로 자동 설정됨
    """
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
    """PDF URL → 텍스트 (pdfplumber)"""
    try:
        import pdfplumber
        from io import BytesIO
        res = requests.get(pdf_url, headers=HEADERS, timeout=15)
        res.raise_for_status()
        with pdfplumber.open(BytesIO(res.content)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages[:10]]  # 최대 10페이지
        return " ".join(pages).strip()
    except Exception as e:
        print(f"    ⚠️ PDF 추출 실패 ({pdf_url[:50]}): {e}")
        return ""


def get_post_content(url: str) -> str:
    """
    공지 본문을 최대한 추출:
      1) .txt 텍스트
      2) .txt 안 이미지 → OCR
      3) 첨부 PDF → pdfplumber
    """
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        div  = soup.select_one(".txt")

        parts = []

        if div:
            # 1) 텍스트 본문
            text = div.get_text(" ", strip=True)
            if text:
                parts.append(text)

            # 2) 이미지 본문 → OCR
            for img in div.find_all("img"):
                src = img.get("src", "")
                if not src:
                    continue
                if src.startswith("/"):
                    src = "https://www.hansung.ac.kr" + src
                ocr_text = _ocr_image(src)
                if ocr_text:
                    parts.append(f"[이미지 OCR] {ocr_text}")

        # 3) 첨부파일 PDF 추출
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
        res.raise_for_status()                             # ✅ fix #8
        soup  = BeautifulSoup(res.text, "html.parser")
        items = []
        for tr in soup.find_all("tr"):
            if not tr.find_all("td"):
                continue
            # ✅ fix #6: class 목록 중 "notice" 있는 경우만 고정공지로 처리
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
    print(f"📋 {TARGET_YEAR}년 공지 수집 시작...")
    while True:
        items, done = get_list_page(page)
        if items:
            all_items.extend(items)
        if done or not items:
            break
        page += 1
        time.sleep(0.3)

    for i, item in enumerate(all_items):
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
# 모델 로더 — st.cache_resource로 캐싱 (재실행해도 재로드 없음)
# ============================================================

@st.cache_resource
def get_embed_model():
    path = EMBED_MODEL_PATH if os.path.exists(EMBED_MODEL_PATH) else BASE_MODEL_EMBED
    from sentence_transformers import SentenceTransformer
    # GPU 없는 로컬 환경 — CPU로 실행 (검색/추천에는 GPU 불필요)
    return SentenceTransformer(path, device="cpu")


@st.cache_resource
def get_summary_pipeline():
    if not os.path.exists(SUMMARY_MODEL_PATH):
        return None
    from transformers import pipeline
    return pipeline("summarization", model=SUMMARY_MODEL_PATH,
                    tokenizer=SUMMARY_MODEL_PATH, max_new_tokens=128,
                    device=-1)  # CPU 명시


@st.cache_resource
def get_classifier():
    if not os.path.exists(CLASSIFY_MODEL_PATH):
        return None, None
    from transformers import pipeline
    clf = pipeline("text-classification", model=CLASSIFY_MODEL_PATH,
                   tokenizer=CLASSIFY_MODEL_PATH,
                   device=-1)  # CPU 명시
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
        doc_id   = hashlib.md5(item["url"].encode()).hexdigest()
        body     = item.get("body", "")
        category = classify_notice(item["title"], body)
        text     = f"제목: {item['title']}\n\n{body}"
        embedding = model.encode(text).tolist()

        existing = collection.get(ids=[doc_id])["ids"]
        if existing:
            # ✅ fix #3: add 대신 upsert → 본문/제목 변경 시 업데이트
            collection.update(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[{
                    "title":    item["title"],
                    "url":      item["url"],
                    "date":     item["date"],
                    "category": category,
                }]
            )
            update_count += 1
        else:
            collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[{
                    "title":    item["title"],
                    "url":      item["url"],
                    "date":     item["date"],
                    "category": category,
                }]
            )
            new_count += 1

    print(f"임베딩 완료 — 신규: {new_count}건 / 업데이트: {update_count}건 / DB 총: {collection.count()}건")


# ============================================================
# 하이브리드 검색
# ✅ fix #5: BM25 인덱스를 st.cache_data로 캐싱
# ============================================================

@st.cache_data(ttl=600, show_spinner=False)
def _build_bm25_index(category_filter: str):
    """카테고리별 BM25 인덱스를 캐싱 (10분 TTL)"""
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

    cat_key = category_filter if category_filter and category_filter != "전체" else "전체"
    where   = {"category": category_filter} \
              if category_filter and category_filter != "전체" else None

    # ✅ fix #5: 캐싱된 BM25 인덱스 사용
    bm25, ids, documents, metadatas = _build_bm25_index(cat_key)
    if bm25 is None:
        return []

    # 벡터 검색
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

    # BM25 검색
    bm25_raw  = bm25.get_scores(tokenize_ko(query))
    bm25_max  = max(bm25_raw) if max(bm25_raw) > 0 else 1
    bm25_scores = {did: s / bm25_max for did, s in zip(ids, bm25_raw)}

    # 합산
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
        f"{user_profile.get('department', '')} "
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
# Streamlit UI
# ============================================================


def main():
    st.set_page_config(
        page_title="상상파인더",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Apple Finder 스타일 CSS ────────────────────────────────
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600&display=swap');

/* 사이드바 항상 펼침 강제 */
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebarCollapsedControl"] { display: none !important; }
.css-1d391kg { display: flex !important; }
section[data-testid="stSidebar"] {
    width: 280px !important;
    min-width: 280px !important;
    transform: translateX(0px) !important;
    visibility: visible !important;
    display: flex !important;
}
section[data-testid="stSidebar"] > div:first-child {
    width: 280px !important;
}

/* 전체 배경 */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #f5f5f7 !important;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display",
                 "Noto Sans KR", sans-serif !important;
}

/* 사이드바 — Finder 좌측 패널 */
[data-testid="stSidebar"] {
    background: rgba(248,248,250,0.97) !important;
    border-right: 1px solid rgba(0,0,0,0.08) !important;
}
[data-testid="stSidebar"] * {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text",
                 "Noto Sans KR", sans-serif !important;
}

/* 메인 영역 */
[data-testid="stMain"] {
    background-color: #f5f5f7 !important;
}

/* 버튼 */
.stButton > button {
    background: #0071e3 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 8px 18px !important;
    transition: all 0.15s ease !important;
    font-family: -apple-system, "SF Pro Text", "Noto Sans KR", sans-serif !important;
}
.stButton > button:hover {
    background: #0077ed !important;
    transform: scale(1.01) !important;
}

/* 인풋 박스 */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: white !important;
    border: 1px solid rgba(0,0,0,0.12) !important;
    border-radius: 10px !important;
    font-size: 15px !important;
    padding: 10px 14px !important;
    font-family: -apple-system, "SF Pro Text", "Noto Sans KR", sans-serif !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
}
.stTextInput > div > div > input:focus {
    border-color: #0071e3 !important;
    box-shadow: 0 0 0 3px rgba(0,113,227,0.15) !important;
}

/* 셀렉트박스 */
.stSelectbox > div > div {
    background: white !important;
    border-radius: 8px !important;
    border: 1px solid rgba(0,0,0,0.12) !important;
}

/* 탭 */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(120,120,128,0.12) !important;
    border-radius: 10px !important;
    padding: 3px !important;
    gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #3c3c43 !important;
    padding: 6px 16px !important;
    font-family: -apple-system, "SF Pro Text", "Noto Sans KR", sans-serif !important;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    color: #1d1d1f !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.12) !important;
}

/* 채팅 메시지 */
.chat-bubble-user {
    background: #0071e3;
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 10px 16px;
    max-width: 70%;
    margin-left: auto;
    margin-bottom: 8px;
    font-size: 15px;
    line-height: 1.5;
    word-break: break-word;
}
.chat-bubble-bot {
    background: white;
    color: #1d1d1f;
    border-radius: 18px 18px 18px 4px;
    padding: 10px 16px;
    max-width: 80%;
    margin-bottom: 8px;
    font-size: 15px;
    line-height: 1.5;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    word-break: break-word;
}
.chat-area {
    background: white;
    border-radius: 16px;
    padding: 20px;
    min-height: 300px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
    margin-bottom: 16px;
}

/* 공지 카드 */
.notice-card {
    background: white;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    border: 1px solid rgba(0,0,0,0.06);
    transition: box-shadow 0.15s ease;
}
.notice-card:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.10);
}
.notice-tag {
    display: inline-block;
    background: rgba(0,113,227,0.1);
    color: #0071e3;
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: 600;
    margin-right: 8px;
}
.notice-title {
    font-size: 15px;
    font-weight: 600;
    color: #1d1d1f;
    margin: 4px 0;
}
.notice-date {
    font-size: 12px;
    color: #86868b;
}
.notice-summary {
    font-size: 13px;
    color: #3c3c43;
    margin-top: 6px;
    line-height: 1.5;
}

/* 사이드바 섹션 헤더 */
.sidebar-section {
    font-size: 11px;
    font-weight: 600;
    color: #86868b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 12px 0 6px 0;
}

/* 상태 배지 */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 12px;
    color: #3c3c43;
    padding: 4px 10px;
    background: rgba(120,120,128,0.1);
    border-radius: 20px;
    margin-bottom: 4px;
}
.status-dot-green { color: #34c759; }
.status-dot-yellow { color: #ff9f0a; }

/* 로고 영역 */
.finder-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 0 16px 0;
}
.logo-icon {
    width: 36px;
    height: 36px;
    background: linear-gradient(135deg, #0071e3, #34aadc);
    border-radius: 9px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
}
.logo-text {
    font-size: 17px;
    font-weight: 600;
    color: #1d1d1f;
    letter-spacing: -0.02em;
}
.logo-sub {
    font-size: 11px;
    color: #86868b;
    font-weight: 400;
}

/* 구분선 */
hr { border: none; border-top: 1px solid rgba(0,0,0,0.08) !important; margin: 12px 0 !important; }

/* 숨기기 */
#MainMenu, footer, header { visibility: hidden; }


</style>
""", unsafe_allow_html=True)

    # ── 세션 초기화 ────────────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "profile" not in st.session_state:
        st.session_state.profile = {}
    if "notices" not in st.session_state:
        st.session_state.notices = []

    # ── 전체 레이아웃: 좌측 패널 + 메인 ──────────────────────
    col_left, col_main_area = st.columns([1, 3.5], gap="small")

    # ── 좌측 패널 (Finder 사이드바 역할) ──────────────────────
    with col_left:
        st.markdown("""
<div style="background:rgba(248,248,250,0.97); border-radius:16px;
     border:1px solid rgba(0,0,0,0.08); padding:20px 16px;
     min-height:90vh; position:sticky; top:0;">

<div class="finder-logo">
  <div class="logo-icon">🔍</div>
  <div>
    <div class="logo-text">상상파인더</div>
    <div class="logo-sub">Hansung Notice Finder</div>
  </div>
</div>
<hr style="margin:12px 0;">
</div>
""", unsafe_allow_html=True)

        # 모델 상태
        st.markdown('<div class="sidebar-section">시스템 상태</div>', unsafe_allow_html=True)
        embed_ready    = os.path.exists(EMBED_MODEL_PATH)
        classify_ready = os.path.exists(CLASSIFY_MODEL_PATH)
        cache_ready    = os.path.exists(NOTICES_CACHE_PATH)
        db_count       = get_chroma().count()

        def badge(label, ok, ok_text="정상", fail_text="미설치"):
            dot = "🟢" if ok else "🟡"
            txt = ok_text if ok else fail_text
            st.markdown(f'<div class="status-badge">{dot} {label} — {txt}</div>',
                       unsafe_allow_html=True)

        badge("임베딩 모델", embed_ready, "파인튜닝됨", "베이스 모델")
        badge("분류 모델",   classify_ready, "파인튜닝됨", "키워드 방식")
        badge("공지 캐시",   cache_ready, "있음", "없음")
        st.markdown(f'<div class="status-badge">🔵 DB — {db_count}건</div>',
                   unsafe_allow_html=True)

        st.markdown("<hr style='margin:12px 0;'>", unsafe_allow_html=True)

        # 프로필
        st.markdown('<div class="sidebar-section">내 프로필</div>', unsafe_allow_html=True)
        dept = st.selectbox("학과", [
            "컴퓨터공학부", "AI응용학과", "기계시스템공학과",
            "전자공학부", "경영학부", "영어영문학과", "기타"
        ])
        grade     = st.selectbox("학년", ["1학년","2학년","3학년","4학년"])
        interests = st.multiselect("관심사", CATEGORIES + ["교환학생"],
                                   placeholder="카테고리 선택")
        if st.button("프로필 저장", use_container_width=True):
            st.session_state.profile = {
                "department": dept, "grade": grade, "interests": interests
            }
            st.success("저장됐습니다.")

        st.markdown("<hr style='margin:12px 0;'>", unsafe_allow_html=True)

        # 데이터 관리
        st.markdown('<div class="sidebar-section">데이터</div>', unsafe_allow_html=True)
        if st.button("📥 공지 크롤링 & 저장", use_container_width=True):
            with st.spinner("크롤링 중..."):
                notices = crawl_all()
            with st.spinner("임베딩 저장 중..."):
                index_notices(notices)
            st.session_state.notices = notices
            _build_bm25_index.clear()
            st.success(f"{len(notices)}건 완료")

        if st.button("📂 캐시 불러오기 & 저장", use_container_width=True):
            notices = load_notices_cache()
            if notices:
                with st.spinner("임베딩 저장 중..."):
                    index_notices(notices)
                st.session_state.notices = notices
                _build_bm25_index.clear()
                st.success(f"{len(notices)}건 완료")
            else:
                st.warning("캐시 없음. data/notices_cache.json을 확인하세요.")

    # ── 메인 영역 ──────────────────────────────────────────────
    with col_main_area:
      tab1, tab2, tab3 = st.tabs(["  검색  ", "  추천  ", "  전체 공지  "])

      # ══════════════════════════════════════════════════════
      # 탭 1 — 챗봇 검색
      # ══════════════════════════════════════════════════════
      with tab1:
          col_main, col_opt = st.columns([3, 1])
          with col_opt:
              cat_filter = st.selectbox("카테고리", ["전체"] + CATEGORIES,
                                       label_visibility="collapsed")
              top_k = st.slider("결과 수", 1, 10, 5, label_visibility="collapsed")
              alpha = st.slider("벡터/BM25", 0.0, 1.0, 0.7, 0.1,
                               label_visibility="collapsed",
                               help="1.0=순수 벡터, 0.0=순수 BM25")

          with col_main:
              # 채팅 히스토리
              chat_container = st.container()
              with chat_container:
                  st.markdown('<div class="chat-area">', unsafe_allow_html=True)
                  if not st.session_state.chat_history:
                      st.markdown("""
<div style="text-align:center; padding: 40px 0; color: #86868b;">
    <div style="font-size:36px; margin-bottom:12px;">🔍</div>
    <div style="font-size:16px; font-weight:500; color:#1d1d1f; margin-bottom:6px;">상상파인더</div>
    <div style="font-size:13px;">한성대 공지를 자연어로 검색하세요</div>
    <div style="font-size:12px; margin-top:16px; color:#aeaeb2;">
      "장학금 신청 기간 알려줘" · "취업박람회 언제야?" · "비교과 프로그램 추천해줘"
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
                                      body_map = {n["url"]: n.get("body","")
                                                 for n in (st.session_state.notices or load_notices_cache())}
                                      body    = body_map.get(r["url"], "")
                                      summary = summarize_notice(r["title"], body) if body else ""
                                      st.markdown(f"""
<div class="notice-card">
    <span class="notice-tag">{r.get('category','기타')}</span>
    <span class="notice-date">{r['date']}</span>
    <div class="notice-title">{r['title']}</div>
    {"<div class='notice-summary'>" + summary + "</div>" if summary else ""}
    <div style="margin-top:8px;">
      <a href="{r['url']}" target="_blank"
         style="font-size:12px;color:#0071e3;text-decoration:none;font-weight:500;">
        공지 바로가기 →
      </a>
    </div>
</div>
""", unsafe_allow_html=True)
                  st.markdown('</div>', unsafe_allow_html=True)

              # 입력창
              with st.form("chat_form", clear_on_submit=True):
                  c1, c2 = st.columns([5, 1])
                  with c1:
                      user_input = st.text_input(
                          "메시지",
                          placeholder="무엇이 궁금하세요?",
                          label_visibility="collapsed"
                      )
                  with c2:
                      submitted = st.form_submit_button("전송", use_container_width=True)

              if submitted and user_input:
                  st.session_state.chat_history.append(
                      {"role": "user", "content": user_input}
                  )
                  results = hybrid_search(
                      user_input, top_k=top_k, alpha=alpha,
                      category_filter=cat_filter if cat_filter != "전체" else None,
                  )
                  if results:
                      reply = f"{len(results)}개의 관련 공지를 찾았습니다."
                  else:
                      reply = "관련 공지를 찾지 못했습니다. 캐시를 먼저 불러와 주세요."
                  st.session_state.chat_history.append(
                      {"role": "bot", "content": reply, "results": results}
                  )
                  st.rerun()

              # 히스토리 초기화
              if st.session_state.chat_history:
                  if st.button("대화 초기화", use_container_width=False):
                      st.session_state.chat_history = []
                      st.rerun()

      # ══════════════════════════════════════════════════════
      # 탭 2 — 맞춤 추천
      # ══════════════════════════════════════════════════════
      with tab2:
          profile = st.session_state.profile
          if not profile:
              st.markdown("""
<div style="text-align:center; padding:60px 0; color:#86868b;">
    <div style="font-size:32px; margin-bottom:12px;">👤</div>
    <div style="font-size:15px; color:#1d1d1f; font-weight:500;">프로필을 먼저 설정하세요</div>
    <div style="font-size:13px; margin-top:6px;">사이드바에서 학과, 학년, 관심사를 입력해주세요</div>
</div>
""", unsafe_allow_html=True)
          else:
              st.markdown(f"""
<div style="background:white; border-radius:12px; padding:14px 18px; margin-bottom:16px;
       box-shadow:0 1px 4px rgba(0,0,0,0.06); display:flex; align-items:center; gap:16px;">
    <div style="font-size:28px;">🎓</div>
    <div>
      <div style="font-size:15px; font-weight:600; color:#1d1d1f;">
        {profile.get('department','')} · {profile.get('grade','')}
      </div>
      <div style="font-size:13px; color:#86868b; margin-top:2px;">
        관심사: {', '.join(profile.get('interests',[])) or '없음'}
      </div>
    </div>
</div>
""", unsafe_allow_html=True)
              if st.button("맞춤 공지 추천받기", type="primary"):
                  with st.spinner("추천 중..."):
                      recs = recommend_notices(profile, top_k=5)
                  if not recs:
                      st.info("추천 결과가 없습니다. 먼저 캐시를 불러와 주세요.")
                  else:
                      body_map = {n["url"]: n.get("body","")
                                 for n in (st.session_state.notices or load_notices_cache())}
                      for r in recs:
                          body    = body_map.get(r["url"], "")
                          summary = summarize_notice(r["title"], body) if body else ""
                          st.markdown(f"""
<div class="notice-card">
    <span class="notice-tag">{r.get('category','기타')}</span>
    <span class="notice-date">{r['date']}</span>
    <span style="font-size:11px;color:#86868b;margin-left:6px;">
      유사도 {r['score']}
    </span>
    <div class="notice-title">{r['title']}</div>
    {"<div class='notice-summary'>" + summary + "</div>" if summary else ""}
    <div style="margin-top:8px;">
      <a href="{r['url']}" target="_blank"
         style="font-size:12px;color:#0071e3;text-decoration:none;font-weight:500;">
        공지 바로가기 →
      </a>
    </div>
</div>
""", unsafe_allow_html=True)

      # ══════════════════════════════════════════════════════
      # 탭 3 — 전체 공지
      # ══════════════════════════════════════════════════════
      with tab3:
          notices = st.session_state.notices or load_notices_cache()
          if not notices:
              st.markdown("""
<div style="text-align:center; padding:60px 0; color:#86868b;">
    <div style="font-size:32px; margin-bottom:12px;">📋</div>
    <div style="font-size:15px; color:#1d1d1f; font-weight:500;">공지가 없습니다</div>
    <div style="font-size:13px; margin-top:6px;">사이드바에서 캐시를 불러와 주세요</div>
</div>
""", unsafe_allow_html=True)
          else:
              c1, c2 = st.columns([2, 1])
              with c1:
                  search_keyword = st.text_input("공지 검색", placeholder="제목으로 검색...",
                                                label_visibility="collapsed")
              with c2:
                  cat_filter3 = st.selectbox("카테고리", ["전체"] + CATEGORIES,
                                            key="tab3_cat", label_visibility="collapsed")

              filtered = notices
              if cat_filter3 != "전체":
                  filtered = [n for n in filtered if n.get("category") == cat_filter3]
              if search_keyword:
                  filtered = [n for n in filtered if search_keyword.lower() in n["title"].lower()]

              st.markdown(
                  f'<div style="font-size:13px;color:#86868b;margin-bottom:12px;">'
                  f'총 {len(filtered)}건</div>',
                  unsafe_allow_html=True)

              for n in filtered[:50]:  # 최대 50건 표시
                  st.markdown(f"""
<div class="notice-card">
    <span class="notice-tag">{n.get('category','기타')}</span>
    <span class="notice-date">{n['date']}</span>
    <div class="notice-title">{n['title']}</div>
    <div style="margin-top:6px;">
      <a href="{n['url']}" target="_blank"
         style="font-size:12px;color:#0071e3;text-decoration:none;font-weight:500;">
        공지 바로가기 →
      </a>
    </div>
</div>
""", unsafe_allow_html=True)

              if len(filtered) > 50:
                  st.markdown(
                      f'<div style="text-align:center;font-size:13px;color:#86868b;padding:12px;">'
                      f'상위 50건만 표시됩니다</div>',
                      unsafe_allow_html=True)


# ============================================================
# 엔트리포인트
# ============================================================

if __name__ == "__main__":
    main()
