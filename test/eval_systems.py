"""
6개 검색 시스템 비교 평가 스크립트 (구 모델 vs 신 모델)
  System A: jhgan/ko-sroberta (OLD) + dense only
  System B: jhgan/ko-sroberta (OLD) + BM25 hybrid  (α=0.5)
  System C: BM-K/KoSimCSE    (NEW) + dense only
  System D: BM-K/KoSimCSE    (NEW) + BM25 hybrid   (α=0.5)
  System E: 파인튜닝 임베딩   (NEW) + BM25 hybrid   (α=0.5)  <- 현재 시스템
  System F: System E + cross-encoder reranker

Metrics (TEST split): Recall@5, MRR, NDCG@5
Corpus : qa_dataset_generation/data/test_notices_2025.json  (100 notices)
QA     : qa_dataset_generation/data/qa_test_2025.jsonl      (TEST split)
"""

import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# ── 경로 ─────────────────────────────────────────────────────────────────
ROOT            = Path(__file__).parent.parent  # 프로젝트 루트
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.core.models import SimCSEEmbedder  # noqa: E402

QA_DATA_DIR     = ROOT / "qa_dataset_generation" / "data"
CORPUS_PATH     = QA_DATA_DIR / "test_notices_2025.json"
QA_PATH         = QA_DATA_DIR / "qa_test_2025.jsonl"
OLD_BASE_MODEL  = "jhgan/ko-sroberta-multitask"         # 구 베이스 모델
NEW_BASE_MODEL  = "BM-K/KoSimCSE-roberta-multitask"     # 신 베이스 모델
BASE_MODEL      = NEW_BASE_MODEL                        # 하위 호환 alias
FINETUNED_MODEL = str(ROOT / "models" / "embed_finetuned")

# cross-encoder 모델: 한국어 지원 모델로 교체 권장
#   - BAAI/bge-reranker-v2-m3                      (다국어, 고성능)
#   - cross-encoder/mmarco-mMiniLMv2-L12-H384-v1   (다국어, 경량)
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

K            = 5    # Recall@K, NDCG@K
RERANK_TOPN  = 20   # reranker 후보 수
ALPHA        = 0.5  # hybrid: dense 가중치


def load_embedder(model_path: str):
    if model_path == NEW_BASE_MODEL or Path(model_path).name == "embed_finetuned":
        print("  파이프라인: SimCSE CLS pooling")
        return SimCSEEmbedder(model_path, device="cpu")

    from sentence_transformers import SentenceTransformer
    print("  파이프라인: sentence-transformers 기본 encode")
    return SentenceTransformer(model_path, device="cpu")


# ── 텍스트 포맷 (app.py 의 index_notices 와 동일) ────────────────────────
def format_doc(notice: dict) -> str:
    return f"제목: {notice['title']}\n\n{notice.get('body', '')}"


def tokenize_ko(text: str) -> list:
    return re.findall(r"[\w가-힣]+", text.lower())


# ── 평가 지표 ────────────────────────────────────────────────────────────
def recall_at_k(ranked: list, gt: int, k: int) -> float:
    return 1.0 if gt in ranked[:k] else 0.0


def mrr_score(ranked: list, gt: int) -> float:
    for i, r in enumerate(ranked):
        if r == gt:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked: list, gt: int, k: int) -> float:
    # 단일 정답 문서: IDCG = 1/log2(2) = 1.0
    for i, r in enumerate(ranked[:k]):
        if r == gt:
            return 1.0 / math.log2(i + 2)
    return 0.0


def compute_scores(all_ranked: list[list[int]], all_gt: list[int]) -> dict:
    # TEST split 출처 (CLAUDE.md: 평가 시 split 명시)
    n = len(all_ranked)
    r5   = sum(recall_at_k(r, g, K) for r, g in zip(all_ranked, all_gt)) / n
    mrr_ = sum(mrr_score(r, g)      for r, g in zip(all_ranked, all_gt)) / n
    ndcg = sum(ndcg_at_k(r, g, K)  for r, g in zip(all_ranked, all_gt)) / n
    return {
        f"Recall@{K}": round(r5,   4),
        "MRR":          round(mrr_, 4),
        f"NDCG@{K}":   round(ndcg, 4),
    }


# ── 검색 시스템 ──────────────────────────────────────────────────────────
class DenseRetriever:
    """System A: 단일 bi-encoder, dense 유사도만 사용"""

    def __init__(self, model_path: str, docs: list[str]):
        print(f"  모델 로딩: {model_path}")
        self.model = load_embedder(model_path)
        print("  문서 인코딩 중...", flush=True)
        embs = self.model.encode(docs, show_progress_bar=True)
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
        self.doc_embs = embs / norms

    def search(self, query: str, k: int) -> list[int]:
        q = self.model.encode([query])
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
        sims = (q @ self.doc_embs.T)[0]
        return np.argsort(-sims)[:k].tolist()


class HybridRetriever:
    """System B/C: dense(α) + BM25(1-α) 하이브리드"""

    def __init__(self, model_path: str, docs: list[str], alpha: float = ALPHA):
        self.alpha = alpha
        print(f"  모델 로딩: {model_path}")
        self.model = load_embedder(model_path)
        print("  문서 인코딩 중...", flush=True)
        embs = self.model.encode(docs, show_progress_bar=True)
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
        self.doc_embs = embs / norms
        print("  BM25 인덱스 구축 중...", flush=True)
        self.bm25 = BM25Okapi([tokenize_ko(d) for d in docs])

    def _scores(self, query: str) -> np.ndarray:
        q = self.model.encode([query])
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
        dense = (q @ self.doc_embs.T)[0]
        d_min, d_max = dense.min(), dense.max()
        dense_n = (dense - d_min) / (d_max - d_min + 1e-9)

        bm25 = np.array(self.bm25.get_scores(tokenize_ko(query)))
        b_max = bm25.max()
        bm25_n = bm25 / (b_max + 1e-9)

        return self.alpha * dense_n + (1 - self.alpha) * bm25_n

    def search(self, query: str, k: int) -> list[int]:
        return np.argsort(-self._scores(query))[:k].tolist()


class RerankRetriever:
    """System D: HybridRetriever 후보를 cross-encoder로 재정렬"""

    def __init__(self, hybrid: HybridRetriever, docs: list[str],
                 ce_model: str, rerank_topn: int = RERANK_TOPN):
        self.hybrid      = hybrid
        self.docs        = docs
        self.rerank_topn = rerank_topn
        print(f"  Cross-encoder 로딩: {ce_model}")
        self.ce = CrossEncoder(ce_model)

    def search(self, query: str, k: int) -> list[int]:
        candidates = self.hybrid.search(query, self.rerank_topn)
        pairs      = [(query, self.docs[i]) for i in candidates]
        scores     = self.ce.predict(pairs)
        reranked   = sorted(zip(candidates, scores), key=lambda x: -x[1])
        return [idx for idx, _ in reranked[:k]]


# ── 메인 ─────────────────────────────────────────────────────────────────
def compare_systems():
    corpus  = json.load(open(CORPUS_PATH, encoding="utf-8"))
    qa_list = [json.loads(l) for l in open(QA_PATH, encoding="utf-8") if l.strip()]

    # TEST split 출처 로그 (CLAUDE.md 규칙 준수)
    print("=" * 65)
    print(f"[TEST split] QA 파일 : {QA_PATH.name}")
    print(f"[TEST split] 코퍼스  : {CORPUS_PATH.name}  ({len(corpus)}개 공지)")
    print(f"  OLD 베이스 모델: {OLD_BASE_MODEL}")
    print(f"  NEW 베이스 모델: {NEW_BASE_MODEL}")
    print("=" * 65)

    docs         = [format_doc(n) for n in corpus]
    title_to_idx = {n["title"]: i for i, n in enumerate(corpus)}

    queries, gt_indices = [], []
    skipped = 0
    for qa in qa_list:
        idx = title_to_idx.get(qa["notice_title"])
        if idx is None:
            skipped += 1
            continue
        queries.append(qa["question"])
        gt_indices.append(idx)

    if skipped:
        print(f"⚠️  코퍼스 미매칭 QA {skipped}개 제외")
    print(f"평가 QA: {len(queries)}개 (고유 공지: {len(set(gt_indices))}개)\n")

    results = {}

    # ── System A : OLD 베이스 + dense ────────────────────────────────────
    print("─" * 65)
    print(f"System A: {OLD_BASE_MODEL} (OLD) + dense only")
    sys_a    = DenseRetriever(OLD_BASE_MODEL, docs)
    ranked_a = [sys_a.search(q, K) for q in queries]
    results["A (old+dense)"] = compute_scores(ranked_a, gt_indices)
    print(f"  결과: {results['A (old+dense)']}\n")

    # ── System B : OLD 베이스 + hybrid ───────────────────────────────────
    print("─" * 65)
    print(f"System B: {OLD_BASE_MODEL} (OLD) + BM25 hybrid")
    sys_b    = HybridRetriever(OLD_BASE_MODEL, docs)
    ranked_b = [sys_b.search(q, K) for q in queries]
    results["B (old+hybrid)"] = compute_scores(ranked_b, gt_indices)
    print(f"  결과: {results['B (old+hybrid)']}\n")

    # ── System C : NEW 베이스 + dense ────────────────────────────────────
    print("─" * 65)
    print(f"System C: {NEW_BASE_MODEL} (NEW) + dense only")
    sys_c    = DenseRetriever(NEW_BASE_MODEL, docs)
    ranked_c = [sys_c.search(q, K) for q in queries]
    results["C (new+dense)"] = compute_scores(ranked_c, gt_indices)
    print(f"  결과: {results['C (new+dense)']}\n")

    # ── System D : NEW 베이스 + hybrid ───────────────────────────────────
    print("─" * 65)
    print(f"System D: {NEW_BASE_MODEL} (NEW) + BM25 hybrid")
    sys_d    = HybridRetriever(NEW_BASE_MODEL, docs)
    ranked_d = [sys_d.search(q, K) for q in queries]
    results["D (new+hybrid)"] = compute_scores(ranked_d, gt_indices)
    print(f"  결과: {results['D (new+hybrid)']}\n")

    # ── System E : NEW 파인튜닝 + hybrid ─────────────────────────────────
    print("─" * 65)
    print("System E: NEW 파인튜닝 임베딩 + BM25 hybrid  ← 현재 시스템")
    if os.path.exists(FINETUNED_MODEL):
        sys_e = HybridRetriever(FINETUNED_MODEL, docs)
        print(f"  파인튜닝 모델 사용: {FINETUNED_MODEL}")
    else:
        print(f"  ⚠️  파인튜닝 모델 없음 ({FINETUNED_MODEL})")
        print("  → NEW 베이스 모델로 대체 (System D 와 동일 결과 예상)")
        sys_e = sys_d
    ranked_e = [sys_e.search(q, K) for q in queries]
    results["E (finetuned+hybrid)"] = compute_scores(ranked_e, gt_indices)
    print(f"  결과: {results['E (finetuned+hybrid)']}\n")

    # ── System F : System E + cross-encoder reranker ──────────────────────
    print("─" * 65)
    print("System F: System E + cross-encoder reranker")
    sys_f    = RerankRetriever(sys_e, docs, CROSS_ENCODER_MODEL)
    ranked_f = [sys_f.search(q, K) for q in queries]
    results["F (E+reranker)"] = compute_scores(ranked_f, gt_indices)
    print(f"  결과: {results['F (E+reranker)']}\n")

    # ── 최종 비교 테이블 ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"📊 시스템 비교  [TEST split — {QA_PATH.name}]")
    print("=" * 70)
    print(f"{'시스템':<35} {f'Recall@{K}':>10} {'MRR':>10} {f'NDCG@{K}':>10}")
    print("-" * 70)
    separators = {"C (new+dense)": "── NEW 모델 ─────────────────────────────────────────────────────"}
    for name, m in results.items():
        if name in separators:
            print(separators[name])
        print(f"{name:<35} {m[f'Recall@{K}']:>10.4f} {m['MRR']:>10.4f} {m[f'NDCG@{K}']:>10.4f}")
    print("=" * 70)
    print(f"평가 QA: {len(queries)}개 | 코퍼스: {len(corpus)}개 | K={K} | α={ALPHA}")
    print(f"OLD: {OLD_BASE_MODEL}")
    print(f"NEW: {NEW_BASE_MODEL}")
    print(f"Corpus split: TEST (qa_test_2025.jsonl, 2025년 공지 기반 독립 생성)")


if __name__ == "__main__":
    compare_systems()
