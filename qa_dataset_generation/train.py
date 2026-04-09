import os, re, time, json, warnings
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urlencode, parse_qs, urlunparse
from datetime import datetime

warnings.filterwarnings("ignore")

# ── 설정 ──────────────────────────────────────────────────────────────
BOARD_LIST_URL     = "https://www.hansung.ac.kr/bbs/hansung/2127/artclList.do"
HEADERS            = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
TARGET_YEAR        = str(datetime.now().year)          # ✅ fix #4: 연도 자동

BASE_MODEL_EMBED   = "jhgan/ko-sroberta-multitask"
BASE_MODEL_CLS     = "klue/bert-base"

EMBED_MODEL_PATH    = "./models/embed_finetuned"
CLASSIFY_MODEL_PATH = "./models/classify_finetuned"

NOTICES_CACHE_PATH  = "./data/notices_cache.json"
SYNTHETIC_QA_PATH   = "./data/synthetic_qa.json"
DRIVE_SAVE_PATH     = "./saved_models"  # ✅ 로컬/클라우드 범용 경로

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

os.makedirs("./models", exist_ok=True)
os.makedirs("./data",   exist_ok=True)


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
# pip install pdfplumber pytesseract Pillow requests
# Colab: !apt-get install -q tesseract-ocr tesseract-ocr-kor
# ============================================================

def _ocr_image(img_url: str) -> str:
    """이미지 URL → OCR 텍스트 (한국어)
    설치:
      Linux:   sudo apt-get install tesseract-ocr tesseract-ocr-kor
      Windows: Tesseract 설치 후 아래 경로 설정
               pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    """
    try:
        import pytesseract
        from PIL import Image
        from io import BytesIO
        import platform
        # Windows 환경 자동 감지 → 기본 경로 설정
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
    """PDF URL → 텍스트 (pdfplumber, 실패 시 빈 문자열)"""
    try:
        import pdfplumber
        from io import BytesIO
        res = requests.get(pdf_url, headers=HEADERS, timeout=15)
        res.raise_for_status()
        with pdfplumber.open(BytesIO(res.content)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages[:10]]  # 최대 10페이지 제한
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
            # download.do 링크 중 PDF인 것
            if "download.do" in href and filename.endswith(".pdf"):
                pdf_url  = BASE + href if href.startswith("/") else href
                pdf_text = _extract_pdf(pdf_url)
                if pdf_text:
                    parts.append(f"[첨부PDF] {pdf_text[:1000]}")  # 너무 길면 앞 1000자

        return " ".join(parts)

    except Exception as e:
        print(f"  ⚠️ 본문 크롤링 실패: {e}")
        return ""


def get_list_page(page: int):
    try:
        res = requests.get(BOARD_LIST_URL, params={"page": page},
                           headers=HEADERS, timeout=10)
        res.raise_for_status()                              # ✅ fix #8
        soup  = BeautifulSoup(res.text, "html.parser")
        items = []
        for tr in soup.find_all("tr"):
            if not tr.find_all("td"):
                continue
            # ✅ fix #6: class="notice" 인 경우만 고정공지로 처리
            tr_classes = tr.get("class") or []
            if "notice" in tr_classes:
                continue
            date_el = tr.select_one(".td-date")
            link_el = tr.select_one(".td-title a")
            if not date_el or not link_el:
                continue
            date_text = date_el.get_text(strip=True)
            if not date_text.startswith(TARGET_YEAR):
                # 연초 엣지케이스: 작년 글이 섞여도 바로 종료하지 않고
                # page 5 이상에서만 종료 (연초 1~2월 대비)
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


def load_qwen_model():
    """
    Qwen2.5-7B-Instruct 로컬 모델 로드.
    한국어 QA 생성용. T4 GPU(15GB) 기준 약 14GB 사용.
    최초 1회 로드 후 재사용 권장.

    사용법:
        qwen = load_qwen_model()
        crawl_all(qwen_pipe=qwen)
    """
    from transformers import pipeline
    import torch

    print("  🤖 Qwen2.5-7B-Instruct 로드 중... (최초 1회, 수 분 소요)")
    pipe = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-7B-Instruct",
        torch_dtype=torch.bfloat16,  # A5000: bfloat16이 더 안정적
        device_map="auto",
    )
    print("  ✅ Qwen 모델 로드 완료")
    return pipe


def _qa_from_notice_local(qwen_pipe, title: str, body: str) -> list:
    """공지 1건에 대해 Qwen2.5-7B로 고유 질문 3개 생성"""
    messages = [
        {
            "role": "system",
            "content": (
                "너는 한국어 질문 생성 전문가야. "
                "주어진 공지사항을 읽고, 오직 이 공지에서만 답할 수 있는 구체적인 질문 3개를 만들어. "
                "날짜, 금액, 신청 대상, 장소, 방법, 기간 등 공지의 구체적인 정보를 질문에 포함시켜. "
                "질문만 줄바꿈으로 구분해서 출력해. 번호나 기호 없이."
            )
        },
        {
            "role": "user",
            "content": f"공지 제목: {title}\n공지 내용: {body[:400]}"
        }
    ]
    try:
        result = qwen_pipe(
            messages,
            max_new_tokens=200,  # 한국어 질문 3개가 잘리지 않도록 여유 확보
        )
        output = result[0]["generated_text"][-1]["content"]
        questions = [q.strip() for q in output.strip().split("\n") if len(q.strip()) > 5]
        generated = questions[:3]
        if len(generated) < 3:
            print(f"    ⚠️ 질문 {len(generated)}개만 생성됨 (목표: 3개)")
        return generated
    except Exception:
        return []


def crawl_all(qwen_pipe=None) -> list:
    """
    공지 크롤링 + QA 동시 생성.
    qwen_pipe: load_qwen_model() 로 로드한 Qwen2.5-7B 파이프라인
               있으면 크롤링과 동시에 고품질 QA 생성
               없으면 크롤링만 하고 QA는 finetune_embedding() 시점에 폴백 생성

    사용법:
        notices = crawl_all()                    # QA 없이 크롤링만
        qwen = load_qwen_model()
        notices = crawl_all(qwen_pipe=qwen)      # 크롤링 + QA 동시 생성 (권장)
    """
    all_items, page = [], 1
    qa_pairs = []

    if qwen_pipe:
        print(f"📋 {TARGET_YEAR}년 공지 수집 + QA 생성 시작 (Qwen2.5-7B 사용)...")
    else:
        print(f"📋 {TARGET_YEAR}년 공지 수집 시작 (QA는 파인튜닝 시점에 생성)...")

    # 목록 수집
    while True:
        items, done = get_list_page(page)
        if items:
            all_items.extend(items)
            print(f"  페이지 {page}: {len(items)}건 (누적 {len(all_items)}건)")
        if done or not items:
            break
        page += 1
        time.sleep(0.3)

    # 본문 크롤링 + QA 생성
    print(f"  → 본문 크롤링 시작...")
    for i, item in enumerate(all_items):
        item["body"]     = get_post_content(item["url"])
        item["category"] = infer_category(item["title"], item["body"])
        print(f"  [{i+1}/{len(all_items)}] [{item['category']}] {item['title'][:40]}", end="")

        # Qwen 있으면 바로 QA 생성
        if qwen_pipe and item["body"]:
            text = f"제목: {item['title']}\n{item['body'][:500]}"
            questions = _qa_from_notice_local(qwen_pipe, item["title"], item["body"])
            if questions:
                for q in questions:
                    qa_pairs.append({"query": q, "positive": text})
                print(f" → QA {len(questions)}개", end="")
            else:
                # API 실패 시 제목 기반 폴백
                qa_pairs.append({"query": f"{item['title']} 알려줘", "positive": text})
                print(f" → 폴백", end="")
        print()
        time.sleep(0.3)  # 크롤링 + API 딜레이 합산

    # 캐시 저장
    with open(NOTICES_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 공지 캐시 저장: {NOTICES_CACHE_PATH} ({len(all_items)}건)")

    # QA 저장 (생성된 경우만)
    if qa_pairs:
        with open(SYNTHETIC_QA_PATH, "w", encoding="utf-8") as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
        print(f"✅ QA 저장: {SYNTHETIC_QA_PATH} ({len(qa_pairs)}쌍)\n")
    else:
        print(f"  → QA는 finetune_embedding() 시점에 생성됩니다.\n")

    return all_items


def load_notices_cache() -> list:
    if os.path.exists(NOTICES_CACHE_PATH):
        with open(NOTICES_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


# ============================================================
# 파인튜닝 1 — 임베딩 모델 (Contrastive Learning)
# ============================================================

def generate_qa_with_qwen(notices: list, qwen_pipe) -> list:
    """
    Qwen2.5-7B-Instruct로 각 공지마다 고유한 질문 3개 생성.
    완전 무료, 로컬 실행, 한국어 품질 우수.

    사용법:
        qwen = load_qwen_model()
        pairs = generate_qa_with_qwen(notices, qwen_pipe=qwen)
    """
    pairs  = []
    failed = 0

    print(f"  🤖 Qwen2.5-7B로 QA 생성 중... (총 {len(notices)}건)")
    for i, notice in enumerate(notices):
        title = notice["title"]
        body  = notice.get("body", "")
        text  = f"제목: {title}\n{body[:500]}"

        if not body or len(body) < 20:
            pairs.append({"query": f"{title} 알려줘", "positive": text})
            continue

        questions = _qa_from_notice_local(qwen_pipe, title, body)
        if questions:
            for q in questions:
                pairs.append({"query": q, "positive": text})
        else:
            failed += 1
            pairs.append({"query": f"{title} 알려줘", "positive": text})

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(notices)}건 완료 (누적 {len(pairs)}쌍)")

    print(f"  ✅ QA 생성 완료 — {len(pairs)}쌍 / 실패: {failed}건")
    return pairs


def generate_synthetic_qa_fallback(notices: list) -> list:
    """
    API 없이 쓸 수 있는 폴백용 템플릿 QA 생성.
    generate_qa_with_api() 실패 시 또는 API 키 없을 때 사용.
    품질은 낮지만 파인튜닝 자체는 가능.

    공지당 서로 다른 질문 유형만 추가하여 positive가 동일한 중복 쌍을 최소화.
    (MultipleNegativesRankingLoss in-batch false negative 방지)
    """
    pairs = []
    for notice in notices:
        title = notice["title"]
        body  = notice.get("body", "")
        text  = f"제목: {title}\n{body[:300]}"

        # 공지당 질문 유형을 다양하게 — positive 동일 중복을 피하기 위해
        # 질문마다 본문 청크를 달리하여 positive도 구분
        body_chunks = [body[:300], body[100:400], body[200:500]]

        queries = [f"{title} 알려줘"]

        # 날짜 추출 → 기간/마감 질문
        dates = re.findall(r"\d{4}[.\-/]\d{1,2}[.\-/]\d{1,2}", body)
        if dates:
            queries.append(f"{title} 신청 기간 언제야?")

        # 금액 추출 → 금액 질문
        if re.search(r"\d+만원|\d+원", body):
            queries.append(f"{title} 금액 얼마야?")

        # 대상 추출 → 대상 질문
        if any(kw in body for kw in ["학년", "전공", "학과", "재학생", "대학원생"]):
            queries.append(f"{title} 신청 대상 알려줘")

        # 질문마다 다른 본문 청크를 positive로 사용해 중복 방지
        for i, q in enumerate(queries):
            chunk = body_chunks[i % len(body_chunks)]
            positive = f"제목: {title}\n{chunk}"
            pairs.append({"query": q, "positive": positive})

    return pairs


def finetune_embedding(notices: list, qwen_pipe=None):
    """
    qwen_pipe: load_qwen_model() 로 로드한 Qwen2.5-7B 파이프라인
               있으면 고품질 QA 생성 (crawl_all에서 이미 생성했으면 자동 재사용)
               없으면 폴백 방식 사용

    사용법:
        qwen = load_qwen_model()
        finetune_embedding(notices, qwen_pipe=qwen)
        # 또는 crawl_all(qwen_pipe=qwen) 으로 이미 QA 생성했으면
        finetune_embedding(notices)  ← 자동 재사용
    """
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from torch.utils.data import DataLoader

    print("=" * 55)
    print("🔧 [1/2] 임베딩 모델 파인튜닝 시작")
    print("=" * 55)

    # ── QA 로드 or 생성 ───────────────────────────────────────
    # crawl_all(qwen_pipe=...) 로 이미 생성된 QA가 있으면 재사용
    # 없으면 Qwen 또는 폴백으로 새로 생성
    existing_qa = load_synthetic_qa()
    if existing_qa:
        print(f"  ♻️  기존 QA 재사용 — {len(existing_qa)}쌍 (crawl_all 시 생성된 것)")
        pairs = existing_qa
    elif qwen_pipe:
        print("  🤖 Qwen2.5-7B 사용 — 공지별 고유 QA 생성")
        pairs = generate_qa_with_qwen(notices, qwen_pipe=qwen_pipe)
    else:
        print("  ⚠️ Qwen 모델 없음 — 폴백 방식 사용")
        print("     고품질 학습 원하면: qwen=load_qwen_model() 후 재실행")
        pairs = generate_synthetic_qa_fallback(notices)

    if len(pairs) < 10:
        print(f"⚠️ 학습 데이터 부족 ({len(pairs)}쌍) — 중단")
        return

    # QA 저장 (새로 생성한 경우만)
    if not existing_qa:
        with open(SYNTHETIC_QA_PATH, "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)
    print(f"  Synthetic QA {len(pairs)}쌍 준비 완료")

    # ── 학습/평가 셋 분리 (데이터 리케이지 방지) ─────────────────────
    import random
    random.seed(42)
    shuffled = pairs[:]
    random.shuffle(shuffled)
    eval_size   = max(50, int(len(shuffled) * 0.1))  # 10% 또는 최소 50쌍
    eval_pairs  = shuffled[:eval_size]
    train_pairs = shuffled[eval_size:]
    print(f"  학습 {len(train_pairs)}쌍 / 평가 {len(eval_pairs)}쌍 분리 완료")

    # ── In-batch false negative 방지 샘플러 ──────────────────────────
    # 같은 positive를 가진 쌍이 같은 배치에 들어가면 false negative 발생.
    # positive 기준으로 그룹화한 뒤 배치 내 중복이 최소화되도록 인터리브.
    from collections import defaultdict
    pos_groups: dict = defaultdict(list)
    for p in train_pairs:
        pos_groups[p["positive"]].append(p)

    # 그룹별로 1개씩 순환 추출 → 같은 positive가 연속 배치에 들어가지 않음
    interleaved = []
    groups = list(pos_groups.values())
    max_len = max(len(g) for g in groups)
    for i in range(max_len):
        for g in groups:
            if i < len(g):
                interleaved.append(g[i])

    train_examples = [InputExample(texts=[p["query"], p["positive"]]) for p in interleaved]
    # ✅ A5000: batch_size 64, num_workers 4, pin_memory / shuffle=False (순서 유지)
    train_dataloader = DataLoader(
        train_examples, shuffle=False, batch_size=64,
        num_workers=4, pin_memory=True,
    )

    model = SentenceTransformer(BASE_MODEL_EMBED)
    loss  = losses.MultipleNegativesRankingLoss(model)

    # InformationRetrievalEvaluator: 학습과 분리된 eval_pairs 사용
    unique_ep       = list(dict.fromkeys(p["positive"] for p in eval_pairs))
    ep_corpus_to_id = {text: str(idx) for idx, text in enumerate(unique_ep)}
    ir_queries  = {str(i): p["query"]                 for i, p in enumerate(eval_pairs)}
    ir_corpus   = {str(idx): text                     for idx, text in enumerate(unique_ep)}
    ir_relevant = {str(i): {ep_corpus_to_id[p["positive"]]} for i, p in enumerate(eval_pairs)}
    evaluator   = InformationRetrievalEvaluator(
        ir_queries, ir_corpus, ir_relevant, name="hansung_qa"
    )

    model.fit(
        train_objectives=[(train_dataloader, loss)],
        epochs=15,
        warmup_steps=100,
        use_amp=True,          # ✅ A5000: 자동 혼합 정밀도
        evaluator=evaluator,
        evaluation_steps=200,
        output_path=EMBED_MODEL_PATH,
        show_progress_bar=True,
    )
    print(f"✅ 임베딩 모델 저장: {EMBED_MODEL_PATH}\n")

    # ✅ 메모리 해제 — 다음 파인튜닝 전 RAM/GPU 확보
    del model, loss, train_dataloader, train_examples
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()
    print("  🧹 임베딩 메모리 해제 완료")


# ============================================================
# 파인튜닝 2 — 분류 모델 (KLUE-BERT)
# ============================================================

def build_classify_dataset(notices: list) -> list:
    # ✅ fix #9 주석: weak label 한계 명시
    # 현재 라벨은 키워드 규칙 기반 → 발표 시 "규칙 기반 silver label" 로 설명 권장
    return [
        {
            "text":  f"{n['title']} {n.get('body', '')[:256]}",
            "label": CATEGORIES.index(n.get("category", "기타")),
        }
        for n in notices
        if n.get("category") in CATEGORIES
    ]


def finetune_classify(notices: list):
    import numpy as np
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                               Trainer, TrainingArguments)
    from datasets import Dataset
    from sklearn.metrics import accuracy_score

    print("=" * 55)
    print("🔧 [2/2] 분류 모델 파인튜닝 시작")
    print("=" * 55)

    raw_data = build_classify_dataset(notices)
    if len(raw_data) < 10:                                 # ✅ fix #2: 데이터 가드
        print(f"⚠️ 분류 학습 데이터 부족 ({len(raw_data)}건) — 중단")
        return
    print(f"  학습 데이터 {len(raw_data)}건")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_CLS)
    model     = AutoModelForSequenceClassification.from_pretrained(
                    BASE_MODEL_CLS, num_labels=len(CATEGORIES))

    def preprocess(examples):
        return tokenizer(examples["text"], max_length=256,
                         truncation=True, padding="max_length")

    dataset = Dataset.from_list(raw_data)
    # stratify_by_column: 라벨 불균형 시 특정 라벨 누락 방지
    try:
        split = dataset.train_test_split(test_size=0.1, seed=42,
                                         stratify_by_column="label")
    except Exception:
        split = dataset.train_test_split(test_size=0.1, seed=42)
    tokenized = split.map(preprocess, batched=True, remove_columns=["text"])

    def compute_metrics(pred):
        preds = np.argmax(pred.predictions, axis=1)
        return {"accuracy": accuracy_score(pred.label_ids, preds)}

    # ✅ fix #12: eval_strategy 버전 호환
    try:
        args = TrainingArguments(
            output_dir=CLASSIFY_MODEL_PATH,
            num_train_epochs=4,
            per_device_train_batch_size=32,  # ✅ A5000: 4 → 32
            per_device_eval_batch_size=32,
            warmup_steps=30,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            bf16=True,                       # ✅ A5000: bfloat16 (fp16보다 안정적)
            dataloader_num_workers=4,        # ✅ A5000: CPU 32코어 활용
            dataloader_pin_memory=True,
        )
    except TypeError:
        args = TrainingArguments(
            output_dir=CLASSIFY_MODEL_PATH,
            num_train_epochs=4,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            warmup_steps=30,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            bf16=True,
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
        )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        processing_class=tokenizer,   # ✅ fix: tokenizer → processing_class
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(CLASSIFY_MODEL_PATH)
    tokenizer.save_pretrained(CLASSIFY_MODEL_PATH)

    with open(f"{CLASSIFY_MODEL_PATH}/label_map.json", "w") as f:
        json.dump({str(i): c for i, c in enumerate(CATEGORIES)}, f)
    print(f"✅ 분류 모델 저장: {CLASSIFY_MODEL_PATH}\n")

    # ✅ 메모리 해제
    del model, tokenizer, trainer, tokenized, dataset, split
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()
    print("  🧹 분류 메모리 해제 완료")


# ============================================================
# 파인튜닝 전후 임베딩 품질 정량 평가 (Recall@K, MRR)
# ============================================================

def evaluate_embedding(notices: list, k_list: list = None):
    """
    베이스 모델 vs 파인튜닝 모델의 Recall@K, MRR 비교
    발표 자료의 정량 지표로 사용 가능
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np

    if k_list is None:
        k_list = [1, 3, 5, 10]

    pairs = load_synthetic_qa()
    if not pairs:
        pairs = generate_synthetic_qa_fallback(notices)
    if len(pairs) < 5:
        print("⚠️ 평가 데이터 부족")
        return

    # 코퍼스 중복 제거 — 같은 공지에 대해 3개 질문이 있어도 corpus는 1개만
    unique_corpus = list(dict.fromkeys(p["positive"] for p in pairs))
    corpus_to_id  = {text: idx for idx, text in enumerate(unique_corpus)}

    corpus       = unique_corpus
    queries      = [p["query"] for p in pairs]
    ground_truth = [corpus_to_id[p["positive"]] for p in pairs]

    results = {}
    models_to_eval = {"베이스 모델": BASE_MODEL_EMBED}
    if os.path.exists(EMBED_MODEL_PATH):
        models_to_eval["파인튜닝 모델"] = EMBED_MODEL_PATH

    for model_name, model_path in models_to_eval.items():
        print(f"\n📊 [{model_name}] 평가 중...")
        model         = SentenceTransformer(model_path)
        corpus_embs   = model.encode(corpus,  convert_to_numpy=True, show_progress_bar=True)
        query_embs    = model.encode(queries, convert_to_numpy=True, show_progress_bar=True)

        # 코사인 유사도 계산
        corpus_norm = corpus_embs / (np.linalg.norm(corpus_embs, axis=1, keepdims=True) + 1e-9)
        query_norm  = query_embs  / (np.linalg.norm(query_embs,  axis=1, keepdims=True) + 1e-9)
        sim_matrix  = query_norm @ corpus_norm.T   # (Q, C)

        recall_at_k = {k: 0.0 for k in k_list}
        mrr          = 0.0

        for q_idx, sims in enumerate(sim_matrix):
            ranked = np.argsort(-sims)             # 내림차순 정렬
            gt     = ground_truth[q_idx]

            # Recall@K
            for k in k_list:
                if gt in ranked[:k]:
                    recall_at_k[k] += 1

            # MRR
            rank = np.where(ranked == gt)[0]
            if len(rank) > 0:
                mrr += 1.0 / (rank[0] + 1)

        n = len(queries)
        model_result = {f"Recall@{k}": round(recall_at_k[k] / n, 4) for k in k_list}
        model_result["MRR"] = round(mrr / n, 4)
        results[model_name] = model_result

        print(f"  결과: {model_result}")

    # 비교 출력
    print("\n" + "=" * 55)
    print("📈 파인튜닝 전후 비교")
    print("=" * 55)
    header = f"{'지표':<15}" + "".join(f"{name:<20}" for name in results)
    print(header)
    print("-" * 55)
    all_metrics = [f"Recall@{k}" for k in k_list] + ["MRR"]
    for metric in all_metrics:
        row = f"{metric:<15}"
        for name, vals in results.items():
            row += f"{vals.get(metric, '-'):<20}"
        print(row)

    return results


def load_synthetic_qa() -> list:
    if os.path.exists(SYNTHETIC_QA_PATH):
        with open(SYNTHETIC_QA_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


# ============================================================
# 전체 파인튜닝 래퍼 — RAM 부족 방지용
# ============================================================

def finetune_all(notices: list, qwen_pipe=None):
    """
    임베딩 + 분류 모델을 순서대로 파인튜닝.
    각 단계 후 모델/데이터 메모리를 명시적으로 해제해서
    연속 실행 시 RAM 부족 세션 다운 방지.

    qwen_pipe: load_qwen_model() 로 로드한 Qwen2.5-7B 파이프라인
               crawl_all(qwen_pipe=...) 로 이미 QA 생성했으면 None으로 호출해도 됨
    사용법:
        # QA가 이미 생성된 경우 (crawl_all(qwen_pipe=qwen) 실행 후)
        finetune_all(notices)

        # QA 없이 finetune_all 실행하는 경우
        qwen = load_qwen_model()
        finetune_all(notices, qwen_pipe=qwen)
    """
    import gc, torch, json

    print("🚀 전체 파인튜닝 시작 (메모리 절약 모드)")
    print(f"  공지 {len(notices)}건 로드됨")

    # 🚨 수정: QA 생성을 Qwen 해제 전에 먼저 실행 → 저장 후 해제
    if qwen_pipe is not None:
        if not load_synthetic_qa():  # 기존 캐시 없을 때만 생성
            print("  🤖 VRAM 해제 전 Qwen2.5-7B로 QA 데이터 우선 생성 중...")
            pairs = generate_qa_with_qwen(notices, qwen_pipe=qwen_pipe)
            with open(SYNTHETIC_QA_PATH, "w", encoding="utf-8") as f:
                json.dump(pairs, f, ensure_ascii=False, indent=2)
            print(f"  ✅ QA {len(pairs)}쌍 생성 및 저장 완료")
        else:
            print("  ♻️  기존 QA 재사용")

        print("  🧹 학습 전 Qwen 파이프라인 메모리 해제 중...")
        del qwen_pipe
        qwen_pipe = None
        gc.collect()
        torch.cuda.empty_cache()
        print("  ✅ Qwen 메모리 해제 완료\n")
    elif load_synthetic_qa():
        print("  ♻️  기존 QA 재사용\n")
    else:
        print("  ⚠️ Qwen 없음 — 폴백 방식으로 QA 생성\n")

    # ── 1. 임베딩 ──────────────────────────────────────────
    # Qwen은 이미 해제되었으므로 None 전달 (저장된 QA 자동 재사용)
    finetune_embedding(notices, qwen_pipe=None)
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  RAM 여유 확보 후 분류 모델 시작\n")

    # ── 2. 분류 ────────────────────────────────────────────
    classify_notices = [
        {"title": n["title"], "body": n.get("body", ""), "category": n.get("category", "기타")}
        for n in notices
    ]
    finetune_classify(classify_notices)
    del classify_notices
    gc.collect()
    torch.cuda.empty_cache()

    print("\n🎉 전체 파인튜닝 완료!")
    print("  → save_models() 로 Drive에 저장하세요.")


# ============================================================
# Google Drive 저장
# ============================================================

def save_models():
    import shutil
    os.makedirs(DRIVE_SAVE_PATH, exist_ok=True)

    for src, name in [
        (EMBED_MODEL_PATH,    "embed_finetuned"),
        (CLASSIFY_MODEL_PATH, "classify_finetuned"),
        ("./data",            "data"),
    ]:
        dst = os.path.join(DRIVE_SAVE_PATH, name)
        if os.path.exists(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"  ✅ {src} → {dst}")
        else:
            print(f"  ⚠️ {src} 없음 (파인튜닝 먼저 실행)")

    print(f"\n🎉 저장 완료: {DRIVE_SAVE_PATH}")
    print("→ saved_models/ 폴더를 app.py와 같은 디렉토리에 두세요.")


# ============================================================
# 엔트리포인트
# ============================================================

if __name__ == "__main__":
    print("train_colab.py 로드 완료.")
    print(f"  대상 연도: {TARGET_YEAR}")
    print("사용 가능한 함수:")
    print("  notices = crawl_all()")
    print("  notices = load_notices_cache()  ← 캐시 재사용")
    print("  qwen = load_qwen_model()                      ← Qwen2.5-7B 로드")
    print("  notices = crawl_all(qwen_pipe=qwen)           ← 크롤링 + QA 동시 생성")
    print("  finetune_all(notices)                         ← QA 자동 재사용")
    print("  finetune_embedding(notices, qwen_pipe=qwen)   ← 임베딩만")
    print("  finetune_classify(notices)                    ← 분류만")
    print("  evaluate_embedding(notices)      ← 파인튜닝 전후 Recall@K, MRR 비교")
    print("  save_models()")
