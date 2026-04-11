import os, re, sys, time, json, warnings
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urlencode, parse_qs, urlunparse
from datetime import datetime

# crawler.py는 프로젝트 루트에 위치 — 하위 디렉토리에서 실행 시 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crawler import get_post_content  # noqa: E402  (중복 구현 제거)

warnings.filterwarnings("ignore")

# ── 설정 ──────────────────────────────────────────────────────────────
BOARD_LIST_URL     = "https://www.hansung.ac.kr/bbs/hansung/2127/artclList.do"
HEADERS            = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
TARGET_YEAR        = str(datetime.now().year)          # ✅ fix #4: 연도 자동

BASE_MODEL_EMBED   = "jhgan/ko-sroberta-multitask"
BASE_MODEL_CLS     = "klue/bert-base"

EMBED_MODEL_PATH    = "./models/embed_finetuned"
CLASSIFY_MODEL_PATH = "./models/classify_finetuned"

NOTICES_CACHE_PATH  = "./data/2026_notice.json"
SYNTHETIC_QA_PATH   = "./data/synthetic_qa.json"
DRIVE_SAVE_PATH     = "./saved_models"  # ✅ 로컬/클라우드 범용 경로

CATEGORIES = ["취업/채용", "인턴십", "장학금", "학자금/근로장학", "학사행정",
              "창업", "국제교류", "교육/특강", "비교과", "공모전/경진대회",
              "봉사/서포터즈", "기타"]

# 1단계: 제목 prefix → 카테고리 직매핑
CATEGORY_PREFIX = {
    "채용정보":   "취업/채용",
    "강소기업채용": "취업/채용",
    "인턴쉽":    "인턴십",
    "교외장학금":  "장학금",
    "국가장학금":  "장학금",
    "학자금대출":  "학자금/근로장학",
    "국가근로":   "학자금/근로장학",
    "면학근로":   "학자금/근로장학",
    "공모전":    "공모전/경진대회",
    "정보":      "공모전/경진대회",
}

# 2단계: 제목 → 본문 순 키워드 매칭
CATEGORY_KEYWORDS = {
    "비교과":       ["비교과"],
    "취업/채용":    ["채용", "신입", "공채", "취업", "채용박람회", "취업박람회", "모집공고", "직무", "채용연계", "일반채용", "추천채용"],
    "인턴십":       ["인턴", "인턴십", "일경험", "체험형", "현장실습", "IPP"],
    "장학금":       ["장학", "장학생", "장학재단", "장학금", "기부장학", "장학사업", "스칼라십", "장학지원"],
    "학자금/근로장학": ["학자금대출", "학자금", "이자지원", "국가근로", "면학근로", "근로장학", "대출이자"],
    "학사행정":     ["수강신청", "수강정정", "졸업", "휴학", "복학", "학점", "트랙변경", "성적", "폐강", "복수전공", "부전공", "휴복학"],
    "창업":         ["창업", "창업동아리", "창업지원", "창업멘토링", "스타트업", "아이디어톤", "입주기업", "예비창업"],
    "국제교류":     ["교환학생", "어학연수", "파견", "글로벌버디", "국제교류", "해외", "어학", "글로컬"],
    "교육/특강":    ["특강", "교육생", "아카데미", "KDT", "K-디지털", "강좌", "교육과정", "역량강화"],
    "공모전/경진대회": ["공모전", "경진대회", "챌린지", "해커톤", "대회", "공모"],
    "봉사/서포터즈": ["서포터즈", "봉사", "멘토", "봉사자", "기자단", "자원활동", "멘토단", "멘토링", "자원봉사"],
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


def tokenize_ko(text: str) -> list:
    return re.findall(r"[\w가-힣]+", text.lower())


# ============================================================
# 크롤러 — 본문 추출 (텍스트 / 이미지 OCR / PDF)
#
# pip install easyocr opencv-python-headless pdfplumber pymupdf
# _ocr_image, _extract_pdf, get_post_content → crawler.py 참고
# ============================================================


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


def crawl_all() -> list:
    """공지 크롤링. 사용법: notices = crawl_all()"""
    all_items, page = [], 1

    print(f"📋 {TARGET_YEAR}년 공지 수집 시작...")

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

    # 본문 크롤링
    print(f"  → 본문 크롤링 시작...")
    for i, item in enumerate(all_items):
        item["body"]     = get_post_content(item["url"])
        item["category"] = infer_category(item["title"], item["body"])
        print(f"  [{i+1}/{len(all_items)}] [{item['category']}] {item['title'][:40]}")
        time.sleep(0.3)

    with open(NOTICES_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 공지 캐시 저장: {NOTICES_CACHE_PATH} ({len(all_items)}건)")

    return all_items


def load_notices_cache() -> list:
    if os.path.exists(NOTICES_CACHE_PATH):
        with open(NOTICES_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


# ============================================================
# 파인튜닝 1 — 임베딩 모델 (Contrastive Learning)
# ============================================================

def finetune_embedding(notices: list):
    """synthetic_qa.json 로드 후 임베딩 모델 파인튜닝."""
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from torch.utils.data import DataLoader

    print("=" * 55)
    print("🔧 [1/2] 임베딩 모델 파인튜닝 시작")
    print("=" * 55)

    pairs = load_synthetic_qa()
    if not pairs:
        print(f"⚠️ {SYNTHETIC_QA_PATH} 없음 — QA 생성 후 재실행")
        return

    if len(pairs) < 10:
        print(f"⚠️ 학습 데이터 부족 ({len(pairs)}쌍) — 중단")
        return

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

def finetune_all(notices: list):
    """
    임베딩 + 분류 모델을 순서대로 파인튜닝.
    사용법: finetune_all(notices)  ← synthetic_qa.json 미리 생성 필요
    """
    import gc, torch

    print("🚀 전체 파인튜닝 시작 (메모리 절약 모드)")
    print(f"  공지 {len(notices)}건 로드됨")

    if load_synthetic_qa():
        print("  ♻️  기존 QA 재사용\n")
    else:
        print("  ⚠️ synthetic_qa.json 없음 — QA 생성 후 재실행\n")
        return

    # ── 1. 임베딩 ──────────────────────────────────────────
    finetune_embedding(notices)
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
    print("  finetune_all(notices)           ← synthetic_qa.json 미리 생성 필요")
    print("  finetune_embedding(notices)     ← 임베딩만")
    print("  finetune_classify(notices)      ← 분류만")
    print("  evaluate_embedding(notices)     ← 파인튜닝 전후 Recall@K, MRR 비교")
    print("  save_models()")
