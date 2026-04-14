# ============================================================
# crawl_2025.py — 2025년 한성대 공지 전체 크롤링 (제목 + 본문 텍스트)
#
# 실행:
#   python crawl_2025.py
#
# 출력:
#   qa_dataset_generation/data/2025_notice.json
#   (기존 파일 있으면 처리된 URL 건너뜀 — resume 지원)
# ============================================================

import json
import time
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

HEADERS  = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
BASE_URL = "https://www.hansung.ac.kr"
BOARD_ID = "2127"
LIST_URL = f"{BASE_URL}/bbs/hansung/{BOARD_ID}/artclList.do"

YEAR       = "2025"
OUT_PATH   = Path("qa_dataset_generation/data/2025_notice.json")
DELAY      = 0.3   # 요청 간 딜레이(초)
SAVE_EVERY = 50    # N건마다 중간 저장

# ── 카테고리 분류 ──────────────────────────────────────────────
# 1단계: 제목 prefix → 카테고리 직매핑
CATEGORY_PREFIX = {
    "채용정보":    "취업/채용",
    "강소기업채용": "취업/채용",
    "인턴쉽":     "인턴십",
    "교외장학금":  "장학금",
    "국가장학금":  "장학금",
    "학자금대출":  "학자금/근로장학",
    "국가근로":    "학자금/근로장학",
    "면학근로":    "학자금/근로장학",
    "공모전":     "공모전/경진대회",
}

# 2단계: 제목 → 본문 순 키워드 매칭
CATEGORY_KEYWORDS = {
    "ROTC":           ["ROTC", "학군사관", "학군단", "현역병 모집", "현역병모집", "예비군", "전문사관",
                       "재병역판정검사"],
    "기숙사/생활관":   ["기숙사", "생활관", "상상빌리지", "우촌학사", "임대기숙사", "사감",
                       "입사생 선발", "대학생주택", "학사관"],
    "비교과":         ["비교과", "동아리", "D-School", "포럼", "대동제", "영상제", "입학식",
                       "HS CREW", "상상파크", "라이프 디자인", "문화탐방", "만우절", "오찬 소통",
                       "Lunch with", "천원의 아침밥", "ESG", "진로집단상담", "리더십 탐험",
                       "학생축제", "문화제", "페스티벌", "소모임",
                       "디즈니 프로그램", "디즈니프로그램", "FSU-Disney", "Disney",
                       "새내기 새로배움터", "새로배움터", "총학생회",
                       "사진전", "영상·사진전", "진로 설명회", "트랙 진로",
                       "학생자치기구", "마음건강", "건강검진", "건강 검진", "대사증후군",
                       "건강관리실", "아웃리치", "동선이음", "청년의 날", "청년문화예술패스",
                       "청년정책 네트워크", "청년정책네트워크", "온통청년", "소셜다이닝",
                       "통일체험", "영테크", "리빙랩"],
    "취업/채용":      ["채용", "신입", "공채", "취업", "채용박람회", "취업박람회", "모집공고",
                       "직무", "채용연계", "일반채용", "추천채용",
                       "초빙 공고", "조교 모집", "수습직원"],
    "인턴십":         ["인턴", "인턴십", "일경험", "체험형", "현장실습", "IPP"],
    "장학금":         ["장학", "장학생", "장학재단", "장학금", "기부장학", "장학사업", "스칼라십", "장학지원"],
    "학자금/근로장학": ["학자금대출", "학자금", "이자지원", "국가근로", "면학근로", "근로장학", "대출이자",
                       "등록금 납부", "등록금 분할", "학기초과자 등록금",
                       "사회첫출발", "청년월세지원", "이사비 지원", "중개보수"],
    "학사행정":       ["수강신청", "수강정정", "졸업", "휴학", "복학", "학점", "트랙변경", "성적",
                       "폐강", "복수전공", "부전공", "휴복학", "재입학", "연계전공",
                       "Micro Degree", "MD과정", "교양영어", "이수신청", "트랙선택",
                       "계절학기", "수업평가", "학위취득유예", "수강포기", "서면신청", "서면 신청",
                       "교차전부", "교차 전부", "편입생", "전부(과)", "학위수여식", "학사학위취득",
                       "오리엔테이션", "반편성고사", "합격자 공고", "합격자공고", "합격자 발표", "합격자발표",
                       "선발 결과", "선발결과", "이수 면제", "이수면제", "수업운영 안내", "출결",
                       "중간고사", "기말고사", "전공과목 변경", "전공변경", "다전공 신청",
                       "학석사연계", "합격자 공지", "합격자공지",
                       "학사경고", "자기설계전공", "교양필수", "상상력이노베이터",
                       "TOPIK", "한국어능력시험", "학술정보관", "예방접종", "정전",
                       "소방 시설", "소방시설", "교체공사", "보수 공사", "공사 안내",
                       "우편취급국", "국제우편", "전산서비스", "종합정보시스템",
                       "유무선인터넷", "유.무선", "퇴직교원", "노동조합", "개방이사",
                       "케이키친", "헤이영", "스쿨버스", "영업시간",
                       "학생단체", "대학발전계획", "중대본", "코딩 라운지"],
    "창업":           ["창업", "창업동아리", "창업지원", "창업멘토링", "스타트업", "아이디어톤",
                       "입주기업", "예비창업", "학생 CEO", "CEO 발굴", "메이커"],
    "국제교류":       ["교환학생", "어학연수", "파견", "글로벌버디", "국제교류", "해외", "어학", "글로컬",
                       "글로벌 튜터링", "글로벌 Conversation", "글로벌 컨버세이션", "글로벌 문화 소통",
                       "단기연수", "단기 연수", "K-Move", "WEST 연수", "한·미대학생",
                       "글로벌 동행", "글로벌동행"],
    "교육/특강":      ["특강", "교육생", "아카데미", "KDT", "K-디지털", "강좌", "교육과정", "역량강화",
                       "평생교육", "RISE", "마이크로디그리", "TOPCIT", "연구방법론",
                       "초청강연", "특별강연", "핵심역량진단", "HS-CESA", "K-CESA", "UICA", "K-NSSE",
                       "폭력예방교육", "필수교육", "전문과정", "SW마에스트로", "코딩 캠프", "코딩캠프",
                       "신규 교과목", "재정데이터", "직업흥미검사", "심리증진",
                       "연구윤리", "워크숍", "진로지도시스템", "진로 캠프", "진로캠프",
                       "심폐소생술", "저작권", "청년인생설계", "기초역량 가이드", "미디어 클래스",
                       "과학살롱", "기초학문", "고시반", "인문학 캠프", "블록체인 밋업",
                       "진로탐색", "GPU 활용"],
    "공모전/경진대회": ["공모전", "경진대회", "챌린지", "해커톤", "대회", "공모", "문학상"],
    "봉사/서포터즈":  ["서포터즈", "서포터스", "봉사", "멘토", "봉사자", "기자단", "자원활동", "멘토단",
                       "멘토링", "자원봉사", "홍보대사", "하랑", "소통-e", "앰버서더",
                       "방송국 HBS", "홍보단", "수습기자", "운영자문위원", "모니터링단", "자문단",
                       "바로알림단", "기획단", "체험단", "발굴단", "순찰대", "제작단",
                       "자원지도자", "볼런톤", "청백리포터", "Friends of Korea",
                       "안전요원", "포모플로깅", "넷제로프렌즈", "KEPCO프렌즈"],
}


def get_post_content(url: str) -> str:
    """공지 본문 텍스트만 추출 (OCR·PDF 제외)."""
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        div  = soup.select_one(".txt")
        if div:
            return div.get_text(" ", strip=True)
    except Exception as e:
        print(f"  ⚠️ 본문 크롤링 실패: {e}")
    return ""


def infer_category(title: str, body: str = "") -> str:
    """제목 prefix → 제목 키워드 → 본문 키워드 순으로 카테고리 판별."""
    for prefix, cat in CATEGORY_PREFIX.items():
        if title.startswith(prefix):
            return cat
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in title for kw in keywords):
            return cat
    if body:
        for cat, keywords in CATEGORY_KEYWORDS.items():
            if any(kw in body for kw in keywords):
                return cat
    return "기타"


# ── 목록 페이지 파싱 ──────────────────────────────────────────

def fetch_list_page(page: int) -> list[dict]:
    """목록 한 페이지 → [{title, url, date}] 반환. 빈 페이지면 []."""
    try:
        res = requests.get(LIST_URL, params={"page": page}, headers=HEADERS, timeout=10)
        res.raise_for_status()
    except Exception as e:
        print(f"  ⚠️ 페이지 {page} 요청 실패: {e}")
        return []

    soup  = BeautifulSoup(res.text, "html.parser")
    rows  = soup.select("table.board-table tbody tr")
    items = []

    for row in rows:
        if "notice" in row.get("class", []):  # 고정 공지 건너뜀
            continue

        title_td = row.select_one("td.td-title a")
        date_td  = row.select_one("td.td-date")
        if not title_td or not date_td:
            continue

        title = title_td.get_text(strip=True)
        date  = date_td.get_text(strip=True)
        href  = title_td.get("href", "")
        url   = BASE_URL + href if href.startswith("/") else href

        items.append({"title": title, "url": url, "date": date})

    return items


# ── 메인 ─────────────────────────────────────────────────────

def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 기존 파일에서 처리된 URL 로드 (resume용)
    results: list[dict] = []
    seen_urls: set[str] = set()
    if OUT_PATH.exists():
        with open(OUT_PATH, encoding="utf-8") as f:
            results = json.load(f)
        seen_urls = {item["url"] for item in results}
        print(f"기존 {len(results)}건 로드 — 처리된 URL {len(seen_urls)}건 건너뜀")

    page      = 1
    new_count = 0
    skip_count = 0
    empty_count = 0
    stop      = False
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"  {YEAR}년 공지 크롤링 시작  [{datetime.now().strftime('%H:%M:%S')}]")
    print(f"{'='*60}\n")

    while not stop:
        print(f"── 페이지 {page} 요청 중...", flush=True)
        items = fetch_list_page(page)
        if not items:
            print(f"  페이지 {page}: 결과 없음 — 종료")
            break

        page_new = 0
        for item in items:
            year = item["date"][:4]

            if year > YEAR:
                continue  # 더 최신 연도 — 건너뜀

            if year < YEAR:
                print(f"  {year}년 데이터 도달 — 수집 종료")
                stop = True
                break

            # year == YEAR
            if item["url"] in seen_urls:
                skip_count += 1
                continue

            # 본문 크롤링
            t0   = time.time()
            body = get_post_content(item["url"])
            elapsed = time.time() - t0

            item["body"]     = body
            item["category"] = infer_category(item["title"], body)

            flag = " ⚠️ 본문없음" if not body else ""
            if not body:
                empty_count += 1

            total_elapsed = time.time() - start_time
            print(
                f"  [{new_count + 1:4d}] {item['date']}  [{item['category']:<12}]"
                f"  {item['title'][:40]:<40}"
                f"  {len(body):>5}자  {elapsed:.1f}s{flag}",
                flush=True,
            )

            results.append(item)
            seen_urls.add(item["url"])
            new_count += 1
            page_new  += 1

            # 중간 저장
            if new_count % SAVE_EVERY == 0:
                with open(OUT_PATH, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                mins, secs = divmod(int(total_elapsed), 60)
                print(
                    f"\n  ── 중간 저장  {new_count}건 신규 / 총 {len(results)}건"
                    f"  (경과 {mins}분 {secs}초) ──\n",
                    flush=True,
                )

            time.sleep(DELAY)

        print(f"  → 페이지 {page} 완료: {page_new}건 수집 / 건너뜀 {skip_count}건 누적\n")

        if not stop:
            page += 1
            time.sleep(DELAY)

    # 최종 저장
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    total_elapsed = time.time() - start_time
    mins, secs = divmod(int(total_elapsed), 60)
    print(f"\n{'='*60}")
    print(f"  완료  [{datetime.now().strftime('%H:%M:%S')}]  총 {mins}분 {secs}초")
    print(f"  신규 {new_count}건 추가 / 총 {len(results)}건")
    if empty_count:
        print(f"  본문 없음: {empty_count}건")
    print(f"  저장 위치: {OUT_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
