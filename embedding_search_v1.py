# pip install chromadb sentence-transformers feedparser requests beautifulsoup4 rank_bm25

import feedparser
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urlencode, parse_qs, urlunparse
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import hashlib
import time
import re

# ── 설정 ──────────────────────────────────────────────────────────────
MODEL_NAME = "jhgan/ko-sroberta-multitask"
BOARD_LIST_URL = "https://www.hansung.ac.kr/bbs/hansung/2127/artclList.do"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
TARGET_YEAR = "2026"

# ── 초기화 ────────────────────────────────────────────────────────────
model = SentenceTransformer(MODEL_NAME)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="hansung_notices",
    metadata={"hnsw:space": "cosine"}
)


# ── URL 정제 ──────────────────────────────────────────────────────────
def clean_url(url):
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    params.pop("layout", None)
    new_query = urlencode({k: v[0] for k, v in params.items()})
    return urlunparse(parsed._replace(query=new_query))


# ── 본문 크롤링 ───────────────────────────────────────────────────────
def get_post_content(url):
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        content_div = soup.select_one(".txt")
        if not content_div:
            return ""
        return content_div.get_text(strip=True)
    except Exception as e:
        print(f"  ⚠️ 크롤링 실패: {e}")
        return ""


# ── 게시판 목록 페이지 파싱 ───────────────────────────────────────────
def get_list_page(page: int):
    try:
        res = requests.get(
            BOARD_LIST_URL,
            params={"page": page},
            headers=HEADERS,
            timeout=10
        )
        soup = BeautifulSoup(res.text, "html.parser")

        items = []
        for tr in soup.find_all("tr"):
            # 고정공지(tr.notice) 스킵, class 없는 일반 게시글만
            if tr.get("class"):
                continue

            date_el = tr.select_one(".td-date")
            link_el = tr.select_one(".td-title a")
            if not date_el or not link_el:
                continue

            date_text = date_el.get_text(strip=True)  # ex) "2026.04.02"

            # 2026년 아닌 글 나오면 중단
            if not date_text.startswith(TARGET_YEAR):
                return items, True  # done=True

            href = link_el.get("href", "")
            if href.startswith("/"):
                href = "https://www.hansung.ac.kr" + href
            url = clean_url(href)

            # 제목에서 카테고리 prefix 제거 (ex. "국제2026..." → "2026...")
            title = link_el.get_text(strip=True)
            title = re.sub(r"^(국제|학사|비교과|장학|취업|진로|창업|기타|현장실습)\s*", "", title)

            items.append({"title": title, "url": url, "date": date_text})

        return items, False

    except Exception as e:
        print(f"  ⚠️ 목록 파싱 실패 (page={page}): {e}")
        return [], False


# ── 전체 2026년 공지 수집 ─────────────────────────────────────────────
def crawl_all_2026():
    """2026년 공지 전부 수집해서 반환"""
    all_items = []
    page = 1

    print(f"📋 2026년 공지 전체 수집 시작...")
    while True:
        items, done = get_list_page(page)
        if items:
            all_items.extend(items)
            print(f"  페이지 {page}: {len(items)}건 수집 (누적 {len(all_items)}건)")
        if done or not items:
            break
        page += 1
        time.sleep(0.3)  # 서버 부담 줄이기

    print(f"  → 총 {len(all_items)}건 수집 완료\n")
    return all_items


# ── 임베딩 & 저장 ─────────────────────────────────────────────────────
def index_notices(items: list):
    new_count = 0
    skip_count = 0

    for item in items:
        doc_id = hashlib.md5(item["url"].encode()).hexdigest()

        # 중복 체크
        if collection.get(ids=[doc_id])["ids"]:
            skip_count += 1
            continue

        body = get_post_content(item["url"])
        text_to_embed = f"제목: {item['title']}\n\n{body}"
        embedding = model.encode(text_to_embed).tolist()

        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text_to_embed],
            metadatas=[{
                "title": item["title"],
                "url": item["url"],
                "date": item["date"],
            }]
        )
        new_count += 1
        print(f"  ✅ [{item['date']}] {item['title'][:45]}")
        time.sleep(0.2)

    print(f"\n임베딩 완료 — 신규: {new_count}건 / 중복 스킵: {skip_count}건")
    print(f"DB 총 보유: {collection.count()}건\n")


# ── 하이브리드 검색 ───────────────────────────────────────────────────
def tokenize(text: str) -> list[str]:
    """간단한 한국어 토크나이저 (공백 + 특수문자 분리)"""
    return re.findall(r"[\w가-힣]+", text.lower())


def hybrid_search(query: str, top_k: int = 5, alpha: float = 0.7):
    """
    alpha: 벡터 검색 가중치 (1-alpha = BM25 가중치)
    기본값 0.7 → 벡터 70%, BM25 30%
    """
    # 전체 문서 가져오기 (BM25용)
    all_docs = collection.get(include=["documents", "metadatas"])
    documents = all_docs["documents"]
    metadatas = all_docs["metadatas"]
    ids = all_docs["ids"]

    if not documents:
        print("저장된 공지가 없습니다.")
        return

    # ── 벡터 검색 ──
    query_embedding = model.encode(query).tolist()
    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k * 2, len(documents)),  # 넉넉히 뽑아서 재랭킹
        include=["metadatas", "distances", "documents"]
    )

    # distance → similarity 점수로 변환 후 정규화
    vector_scores = {}
    raw_distances = vector_results["distances"][0]
    if raw_distances:
        max_sim = 1 - min(raw_distances)
        min_sim = 1 - max(raw_distances)
        for vid, dist in zip(vector_results["ids"][0], raw_distances):
            sim = 1 - dist
            # 0~1 정규화
            norm = (sim - min_sim) / (max_sim - min_sim + 1e-9)
            vector_scores[vid] = norm

    # ── BM25 검색 ──
    tokenized_docs = [tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = tokenize(query)
    bm25_raw = bm25.get_scores(tokenized_query)

    # BM25 점수 정규화
    bm25_max = max(bm25_raw) if max(bm25_raw) > 0 else 1
    bm25_scores = {doc_id: score / bm25_max for doc_id, score in zip(ids, bm25_raw)}

    # ── 점수 합산 ──
    all_ids = set(vector_scores.keys()) | set(bm25_scores.keys())
    final_scores = {}
    for doc_id in all_ids:
        v_score = vector_scores.get(doc_id, 0)
        b_score = bm25_scores.get(doc_id, 0)
        final_scores[doc_id] = alpha * v_score + (1 - alpha) * b_score

    # 상위 top_k 추출
    top_ids = sorted(final_scores, key=lambda x: final_scores[x], reverse=True)[:top_k]

    # 메타데이터 매핑
    meta_map = {doc_id: meta for doc_id, meta in zip(ids, metadatas)}

    print(f"\n🔍 '{query}' 검색 결과 (벡터 {int(alpha*100)}% + BM25 {int((1-alpha)*100)}%)")
    print("=" * 55)
    for i, doc_id in enumerate(top_ids):
        meta = meta_map[doc_id]
        score = final_scores[doc_id]
        print(f"{i+1}. [{score:.3f}] {meta['title']}")
        print(f"   {meta['date']} | {meta['url']}\n")


# ── 실행 ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. 2026년 전체 공지 수집
    items = crawl_all_2026()

    # 2. 임베딩 & 저장
    if items:
        index_notices(items)
    else:
        # 게시판 구조가 달라서 파싱 실패한 경우
        print("⚠️ 목록 파싱 실패. 아래 '게시판 구조 디버그' 섹션 참고")

    # 3. 검색 테스트
    hybrid_search("장학금 신청")
    hybrid_search("취업박람회")
    hybrid_search("어학연수 영어")