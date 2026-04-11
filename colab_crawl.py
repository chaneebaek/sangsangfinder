# ============================================================
# colab_crawl.py — Google Colab용 공지 재크롤링 스크립트
#
# 실행 순서:
#   셀 1: 패키지 설치
#   셀 2: Google Drive 마운트
#   셀 3: 이 파일 실행
#
# 또는 셀마다 아래 구분선(# %% [셀 N])을 기준으로 잘라서 붙여넣기
# ============================================================

# %% [셀 1] 패키지 설치 ─────────────────────────────────────
# !pip install -q opencv-python-headless pymupdf pdfplumber requests beautifulsoup4

# %% [셀 2] Google Drive 마운트 ─────────────────────────────
# from google.colab import drive
# drive.mount('/content/drive')

# %% [셀 3] 크롤링 실행 ─────────────────────────────────────

import json
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import requests
from bs4 import BeautifulSoup

# ── 경로 설정 ────────────────────────────────────────────────
SOURCE_PATH = '/content/drive/MyDrive/sangsangfinder/2026_notice.json'
SAVE_PATH   = '/content/drive/MyDrive/sangsangfinder/2026_notice.json'

# ── Clova OCR 설정 ───────────────────────────────────────────
# .env를 Drive에 올려뒀다면 아래 주석 해제:
# from dotenv import load_dotenv
# load_dotenv('/content/drive/MyDrive/sangsangfinder/.env')
#
# 또는 직접 입력:
# import os; os.environ["CLOVA_OCR_API_URL"] = "..."
#             os.environ["CLOVA_OCR_SECRET_KEY"] = "..."

_CLOVA_API_URL    = os.getenv("CLOVA_OCR_API_URL")
_CLOVA_SECRET_KEY = os.getenv("CLOVA_OCR_SECRET_KEY")

# ── 상수 ─────────────────────────────────────────────────────
HEADERS  = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
BASE_URL = "https://www.hansung.ac.kr"

_SKIP_IMG_EXTS = {".gif", ".svg", ".ico", ".webp", ".bmp"}
_SKIP_IMG_RE   = re.compile(
    r"(logo|banner|icon|btn_|bullet|bg_|spacer|arrow|line_|border)", re.IGNORECASE
)
_NOISE_RE      = re.compile(r"[^\w\s가-힣.,!?%\-:/()\[\]@#&*+]")
_SLICE_HEIGHT  = 1500
_SLICE_OVERLAP = 100

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
    "정보":       "공모전/경진대회",
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
                      "사진전", "영상·사진전", "진로 설명회", "트랙 진로"],
    "취업/채용":      ["채용", "신입", "공채", "취업", "채용박람회", "취업박람회", "모집공고", "직무", "채용연계", "일반채용", "추천채용"],
    "인턴십":         ["인턴", "인턴십", "일경험", "체험형", "현장실습", "IPP"],
    "장학금":         ["장학", "장학생", "장학재단", "장학금", "기부장학", "장학사업", "스칼라십", "장학지원"],
    "학자금/근로장학": ["학자금대출", "학자금", "이자지원", "국가근로", "면학근로", "근로장학", "대출이자",
                      "등록금 납부", "등록금 분할", "학기초과자 등록금"],
    "학사행정":       ["수강신청", "수강정정", "졸업", "휴학", "복학", "학점", "트랙변경", "성적", "폐강", "복수전공", "부전공", "휴복학",
                      "재입학", "연계전공", "Micro Degree", "MD과정", "교양영어", "이수신청",
                      "트랙선택", "계절학기", "수업평가", "학위취득유예", "수강포기", "서면신청", "서면 신청",
                      "교차전부", "교차 전부", "편입생", "전부(과)", "학위수여식", "학사학위취득",
                      "오리엔테이션", "반편성고사", "합격자 공고", "합격자공고", "합격자 발표", "합격자발표",
                      "선발 결과", "선발결과", "이수 면제", "이수면제", "수업운영 안내", "출결",
                      "중간고사", "기말고사", "전공과목 변경", "전공변경", "다전공 신청",
                      "학석사연계", "합격자 공지", "합격자공지",
                      "학사경고", "자기설계전공", "교양필수", "상상력이노베이터"],
    "창업":           ["창업", "창업동아리", "창업지원", "창업멘토링", "스타트업", "아이디어톤", "입주기업", "예비창업",
                      "학생 CEO", "CEO 발굴"],
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
                      "과학살롱", "기초학문"],
    "공모전/경진대회": ["공모전", "경진대회", "챌린지", "해커톤", "대회", "공모", "문학상"],
    "봉사/서포터즈":  ["서포터즈", "서포터스", "봉사", "멘토", "봉사자", "기자단", "자원활동", "멘토단", "멘토링", "자원봉사",
                      "홍보대사", "하랑", "소통-e", "앰버서더", "방송국 HBS", "홍보단",
                      "수습기자", "운영자문위원", "모니터링단", "자문단",
                      "바로알림단", "기획단", "체험단", "발굴단", "순찰대", "제작단",
                      "자원지도자", "볼런톤", "청백리포터", "Friends of Korea"],
}


# ── 유틸 ─────────────────────────────────────────────────────

def _is_text_image(src: str) -> bool:
    ext = os.path.splitext(src.split("?")[0])[1].lower()
    if ext in _SKIP_IMG_EXTS:
        return False
    if _SKIP_IMG_RE.search(src):
        return False
    return True


def _clean_ocr_text(text: str) -> str:
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = _NOISE_RE.sub("", text)
    return text.strip()


def _img_ext_from_url(url: str) -> str:
    ext = os.path.splitext(url.split("?")[0])[1].lower().lstrip(".")
    return ext if ext in {"jpg", "jpeg", "png", "tif", "tiff"} else "jpg"


# ── 전처리 ───────────────────────────────────────────────────

def _preprocess(img):
    """OCR 전처리: 업스케일 → CLAHE → Otsu/Adaptive 이진화 → 언샤프 마스킹."""
    import cv2, numpy as np

    h, w = img.shape[:2]
    if min(h, w) < 800:
        scale = 800 / min(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_ratio = np.sum(otsu == 255) / otsu.size

    if 0.01 < white_ratio < 0.99:
        binary = otsu
    else:
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

    img  = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    img  = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    return img


# ── Clova OCR ────────────────────────────────────────────────

def _clova_ocr_bytes(img_bytes: bytes, fmt: str = "jpg") -> list:
    """이미지 bytes → Clova OCR API → 텍스트 리스트."""
    if not _CLOVA_API_URL or not _CLOVA_SECRET_KEY:
        raise RuntimeError("CLOVA_OCR_API_URL 또는 CLOVA_OCR_SECRET_KEY가 설정되지 않았습니다.")

    request_json = {
        "images": [{"format": fmt, "name": "ocr"}],
        "requestId": str(uuid.uuid4()),
        "version": "V2",
        "timestamp": int(round(time.time() * 1000)),
    }
    response = requests.post(
        _CLOVA_API_URL,
        headers={"X-OCR-SECRET": _CLOVA_SECRET_KEY},
        data={"message": json.dumps(request_json).encode("UTF-8")},
        files=[("file", img_bytes)],
        timeout=30,
    )
    response.raise_for_status()

    fields = response.json()["images"][0].get("fields", [])
    return [f["inferText"] for f in fields if f.get("inferText", "").strip()]


def _ocr_img_array(img, fmt: str = "jpg") -> list:
    """ndarray → Clova OCR (긴 이미지 슬라이스 분할)."""
    import cv2

    def _ocr_slice(crop) -> list:
        _, buf = cv2.imencode(f".{fmt}", crop)
        return _clova_ocr_bytes(buf.tobytes(), fmt)

    h = img.shape[0]
    if h <= _SLICE_HEIGHT:
        return _ocr_slice(img)

    texts = []
    y = 0
    while y < h:
        y_end = min(y + _SLICE_HEIGHT, h)
        texts.extend(_ocr_slice(img[y:y_end, :]))
        if y_end == h:
            break
        y = y_end - _SLICE_OVERLAP
    return texts


def _ocr_image(img_url: str) -> str:
    """이미지 URL → OCR 텍스트 (Clova OCR)."""
    if not _is_text_image(img_url):
        return ""
    try:
        import cv2, numpy as np
        res = requests.get(img_url, headers=HEADERS, timeout=10)
        res.raise_for_status()
        img = cv2.imdecode(np.frombuffer(res.content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return ""
        h, w = img.shape[:2]
        if h < 30 or w < 100:
            return ""
        fmt   = _img_ext_from_url(img_url)
        img   = _preprocess(img)
        lines = _ocr_img_array(img, fmt)
        return _clean_ocr_text(" ".join(lines))
    except Exception as e:
        print(f"    ⚠️ OCR 실패 ({img_url[:50]}): {e}")
        return ""


def _extract_pdf(pdf_url: str) -> str:
    """PDF → 텍스트 (디지털: pdfplumber / 스캔: PyMuPDF + Clova OCR)."""
    try:
        res = requests.get(pdf_url, headers=HEADERS, timeout=15)
        res.raise_for_status()
        pdf_bytes = res.content

        import pdfplumber
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages[:10]]
        text = " ".join(pages).strip()
        if text:
            return text

        import fitz, cv2, numpy as np
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        ocr_parts = []
        for page_num in range(min(len(doc), 10)):
            pix = doc[page_num].get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            img   = _preprocess(img)
            lines = _ocr_img_array(img, "jpg")
            if lines:
                ocr_parts.append(" ".join(lines))
        return _clean_ocr_text("\n".join(ocr_parts))

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

            img_srcs = []
            for img in div.find_all("img"):
                src = img.get("src", "")
                if not src:
                    continue
                if src.startswith("/"):
                    src = BASE_URL + src
                if _is_text_image(src):
                    img_srcs.append(src)

            if img_srcs:
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = {executor.submit(_ocr_image, src): src for src in img_srcs}
                    for future in as_completed(futures):
                        ocr_text = future.result()
                        if ocr_text:
                            parts.append(f"[이미지 OCR] {ocr_text}")

        for a in soup.find_all("a", href=True):
            href     = a["href"]
            filename = a.get_text(strip=True).lower()
            if "download.do" in href and filename.endswith(".pdf"):
                pdf_url  = BASE_URL + href if href.startswith("/") else href
                pdf_text = _extract_pdf(pdf_url)
                if pdf_text:
                    parts.append(f"[첨부PDF] {pdf_text}")

        return " ".join(parts)
    except Exception as e:
        print(f"  ⚠️ 본문 크롤링 실패: {e}")
        return ""


def infer_category(title: str, body: str) -> str:
    # 1단계: prefix 직매핑
    for prefix, cat in CATEGORY_PREFIX.items():
        if title.startswith(prefix):
            return cat
    # 2단계: 제목 키워드 우선
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in title for kw in keywords):
            return cat
    # 3단계: 본문 키워드 fallback
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in body for kw in keywords):
            return cat
    return "기타"


# ── 메인 크롤링 루프 ──────────────────────────────────────────

if __name__ == "__main__":
    with open(SOURCE_PATH, encoding="utf-8") as f:
        notices = json.load(f)

    total = len(notices)
    print(f"\n총 {total}건 재크롤링 시작...\n")

    for i, item in enumerate(notices):
        body             = get_post_content(item["url"])
        item["body"]     = body
        item["category"] = infer_category(item["title"], body)
        print(f"  [{i+1}/{total}] [{item['category']}] {item['title'][:50]}")
        time.sleep(0.3)

        if (i + 1) % 50 == 0:
            with open(SAVE_PATH, "w", encoding="utf-8") as f:
                json.dump(notices, f, ensure_ascii=False, indent=2)
            print(f"  ── 중간 저장 완료 ({i+1}/{total}건) ──")

    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(notices, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 완료: {SAVE_PATH} ({total}건 업데이트)")
