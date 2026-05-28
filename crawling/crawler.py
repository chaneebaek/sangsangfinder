# ============================================================
# crawler.py — 공통 크롤러 모듈
#
# 설치:
#   pip install opencv-python-headless pdfplumber pymupdf requests beautifulsoup4 python-dotenv
# ============================================================

import json
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

HEADERS  = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
BASE_URL = "https://www.hansung.ac.kr"

# ── Clova OCR 설정 ─────────────────────────────────────────────
_CLOVA_API_URL    = os.getenv("CLOVA_OCR_API_URL")
_CLOVA_SECRET_KEY = os.getenv("CLOVA_OCR_SECRET_KEY")

# ── 이미지 필터 설정 ──────────────────────────────────────────
_SKIP_IMG_EXTS = {".gif", ".svg", ".ico", ".webp", ".bmp"}
_SLICE_HEIGHT  = 1500
_SLICE_OVERLAP = 100

_SKIP_IMG_RE = re.compile(
    r"(logo|banner|icon|btn_|bullet|bg_|spacer|arrow|line_|border)",
    re.IGNORECASE,
)
_NOISE_RE = re.compile(r"[^\w\s가-힣.,!?%\-:/()\[\]@#&*+]")
_PDF_TEXT_SUFFICIENT_CHARS = 120
_PDF_TEXT_SPARSE_CHARS = 40


# ============================================================
# 내부 유틸
# ============================================================

def _is_text_image(src: str) -> bool:
    """OCR 시도할 가치가 있는 이미지인지 판단."""
    path = src.split("?")[0]
    ext  = os.path.splitext(path)[1].lower()
    if ext in _SKIP_IMG_EXTS:
        return False
    if _SKIP_IMG_RE.search(src):
        return False
    return True


def _clean_ocr_text(text: str) -> str:
    """OCR 결과 후처리: 과도한 개행·공백 정리, 노이즈 문자 제거."""
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = _NOISE_RE.sub("", text)
    return text.strip()


def _img_ext_from_url(url: str) -> str:
    """URL에서 이미지 확장자 추출 (Clova OCR format 파라미터용)."""
    ext = os.path.splitext(url.split("?")[0])[1].lower().lstrip(".")
    return ext if ext in {"jpg", "jpeg", "png", "tif", "tiff", "pdf"} else "jpg"


# ============================================================
# 이미지 전처리
# ============================================================

def _preprocess(img):
    """OCR 전처리: 저대비·저해상도 이미지 보정.

    1) 업스케일: 짧은 쪽이 800px 미만이면 확대
    2) CLAHE: 국소 대비 강화
    3) 이진화 자동 선택:
       - Otsu: 히스토그램이 이봉분포(배경/텍스트 명확히 분리)일 때 더 깔끔
       - Adaptive: Otsu 결과가 한쪽으로 쏠릴 때(배경 불균일) 폴백
    4) 언샤프 마스킹: 텍스트 경계 선명화
    """
    import cv2
    import numpy as np

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
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

    img  = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    img  = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    return img


# ============================================================
# Clova OCR
# ============================================================

def _clova_ocr_bytes(img_bytes: bytes, fmt: str = "jpg") -> list[str]:
    """이미지 bytes → Clova OCR API → 텍스트 리스트."""
    if not _CLOVA_API_URL or not _CLOVA_SECRET_KEY:
        raise RuntimeError("CLOVA_OCR_API_URL 또는 CLOVA_OCR_SECRET_KEY가 .env에 없습니다.")

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


def _ocr_img_array(img, fmt: str = "jpg") -> list[str]:
    """ndarray 이미지 → Clova OCR 텍스트 리스트 (긴 이미지 슬라이스 분할)."""
    import cv2

    def _ocr_slice(crop) -> list[str]:
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
        import cv2
        import numpy as np

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


# ============================================================
# PDF 추출
# ============================================================

def _extract_pdf(pdf_url: str) -> str:
    """PDF URL → 텍스트.
    - pdfplumber로 텍스트 레이어 추출
    - 텍스트가 부족하거나 이미지 중심인 페이지만 Clova OCR 추가 수행
    """
    try:
        res = requests.get(pdf_url, headers=HEADERS, timeout=15)
        res.raise_for_status()
        pdf_bytes = res.content

        parts = []
        pages_to_ocr: set[int] | None = set()
        try:
            import pdfplumber
            with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
                for page_num, page in enumerate(pdf.pages[:10]):
                    page_text = (page.extract_text() or "").strip()
                    if page_text:
                        parts.append(page_text)

                    signal_len = len(re.sub(r"\s+", "", page_text))
                    has_images = bool(getattr(page, "images", None))
                    text_is_sparse = signal_len < _PDF_TEXT_SPARSE_CHARS
                    image_heavy = has_images and signal_len < _PDF_TEXT_SUFFICIENT_CHARS
                    if text_is_sparse or image_heavy:
                        pages_to_ocr.add(page_num)
        except ImportError:
            print("    ℹ️ pdfplumber 미설치 — PDF 텍스트 레이어 추출 스킵")
            pages_to_ocr = None

        # 비용 절감을 위해 텍스트가 충분한 페이지는 OCR하지 않는다.
        try:
            import cv2
            import fitz
            import numpy as np

            doc       = fitz.open(stream=pdf_bytes, filetype="pdf")
            ocr_parts = []

            target_pages = (
                range(min(len(doc), 10))
                if pages_to_ocr is None
                else sorted(page for page in pages_to_ocr if page < min(len(doc), 10))
            )
            for page_num in target_pages:
                pix = doc[page_num].get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                if pix.n == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

                img   = _preprocess(img)
                lines = _ocr_img_array(img, "jpg")
                if lines:
                    ocr_parts.append(" ".join(lines))

            ocr_text = _clean_ocr_text("\n".join(ocr_parts))
            if ocr_text:
                parts.append(f"[PDF OCR] {ocr_text}")

        except ImportError:
            print("    ℹ️ pymupdf 미설치 — 스캔 PDF OCR 스킵 (pip install pymupdf)")

        return " ".join(parts)

    except Exception as e:
        print(f"    ⚠️ PDF 추출 실패 ({pdf_url[:50]}): {e}")
        return ""


# ============================================================
# 본문 크롤링
# ============================================================

def get_post_content(url: str) -> str:
    """공지 본문 추출: 텍스트 + 이미지 OCR (병렬) + PDF."""
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
