# CLAUDE.md — sangsangfinder

## Project Overview
Hansung University notice search & recommendation system using RAG pipeline.
- Embedding: `jhgan/ko-sroberta-multitask`
- Hybrid search: BM25 (0.3) + dense vector (0.7)
- Vector DB: ChromaDB → Pinecone 마이그레이션 예정
- UI: Streamlit
- Deployment: Streamlit Community Cloud 예정 (변경 가능)

---

## Quick Start

```bash
# 앱 실행
streamlit run app.py

# 크롤링 (로컬)
python crawler.py

# 크롤링 (Colab용)
# colab_crawl.py 를 Colab에 업로드 후 실행

# 임베딩 학습 (Colab)
python train.py

# 테스트
pytest tests/ -v --tb=short
```

---

## Environment

```bash
# 필수 환경변수 (.env 파일에 저장)
GEMINI_API_KEY=...
```

- Python 3.10+ 권장
- `pip install -r requirements.txt`
- GPU 환경(Colab T4)에서 `train.py` 실행 권장

---

## Key Files

| 파일 | 역할 |
|------|------|
| `app.py` | Streamlit UI + RAG 파이프라인 진입점 |
| `crawler.py` | 한성대 공지사항 크롤러 |
| `colab_crawl.py` | Colab 환경용 크롤러 |
| `train.py` | 임베딩/요약/분류 모델 파인튜닝 (Colab) |
| `crawl_2025_titles.py` | 2025년 공지 제목 수집 스크립트 |
| `embedding_search_v1.py` | 임베딩 검색 초기 구현 |
| `qa_dataset_generation/` | QA 데이터셋 생성 파이프라인 |

---

## Claude Behavior (Token Efficiency)

- 행동 전 먼저 생각하라. 코드 작성 전 기존 파일을 읽어라.
- 출력은 간결하게, 추론은 철저하게.
- 파일 전체 재작성보다 부분 편집(Edit)을 우선하라.
- 이미 읽은 파일은 변경 가능성이 없으면 다시 읽지 마라.
- 완료 선언 전 코드를 테스트하라.
- 서두 인사말("물론이죠!", "좋은 질문입니다!") 및 마무리 멘트("도움이 됐으면 좋겠습니다") 금지.
- 해결책은 단순하고 직접적으로. 불필요한 추상화 금지.
- 사용자 지시는 이 파일보다 항상 우선한다.

---

## Core Principles

### 1. Logical Consistency Check (논리적 정합성)
모든 코드 변경 전 반드시 확인:
- 데이터 흐름이 앞 단계 → 뒤 단계로 일관되게 이어지는가?
- 전처리 → 임베딩 → 인덱싱 → 검색 → 생성 각 단계의 입출력 타입이 맞는가?
- 새 모듈이 기존 인터페이스를 깨지 않는가?

### 2. Evaluation Data Integrity (평가 데이터 격리)
**절대 규칙: test/val 셋은 학습·파인튜닝·파라미터 튜닝에 절대 사용하지 않는다.**
- 평가 전 반드시 split 출처를 명시적으로 주석으로 표기
```python
  # TRAIN split — used for tuning BM25 k1/b parameters
  # TEST split  — held out, touched only in final eval
```
- 평가 스크립트는 항상 split 출처를 로그에 출력
- 지표(Recall@K, MRR)를 보고할 때 어떤 split인지 반드시 명시

### 3. TDD (Test-Driven Development)
새 기능 구현 순서:
1. `tests/` 에 실패하는 테스트 먼저 작성
2. 테스트를 통과하는 최소 구현
3. 리팩토링
```
tests/
test_chunker.py       # 청킹 로직
test_retriever.py     # hybrid search 결과 검증
test_reranker.py      # reranker context compression
test_eval_pipeline.py # split 격리 검증 포함
```

테스트 실행:
```bash
pytest tests/ -v --tb=short
```

> **TODO**: `tests/` 디렉토리 미생성. 신규 기능 추가 시 아래 구조로 생성:
> `test_chunker.py`, `test_retriever.py`, `test_reranker.py`, `test_eval_pipeline.py`

---

## Workflow Rules

- **새 컴포넌트 추가 시**: 테스트 → 구현 → 문서 순서 준수
- **평가 지표 보고 시**: split 명시 없이는 숫자만 말하지 않기
- **파이프라인 변경 시**: 상·하류 단계 인터페이스 영향 먼저 확인
- **10건 이상의 대규모 작업 시**: 10개 이하로 실행해서 파이프라인이 정상 동작하는지 확인하고 대규모 작업 수행
- **출력 파일(JSONL 등) 작성 시**: 항상 `"a"` 모드로 열어라. 토큰 부족·중단 등으로 재실행 시 기존 결과가 사라지지 않도록 한다.
  - 재실행 시 중복 방지: 기존 파일에서 처리된 항목(e.g. `notice_title`)을 set으로 읽어 건너뜀
  - JSON 캐시 파일(`notices_cache.json` 등 전체를 덮어쓰는 파일)은 예외적으로 `"w"` 유지

---

## Security

- **`.env` 파일은 절대 열람·출력·커밋하지 않는다.** API 키 등 민감 정보가 포함되어 있으므로, 어떤 이유로도 내용을 읽거나 로그에 출력하지 않는다.

---

## Known Pitfalls
- [ ] 과거 사례: test/val 분리 없이 전체 데이터로 Recall@K 계산 → 지표 신뢰 불가
      → 방지책: `test_eval_pipeline.py`에서 split 격리 assertion 필수
- [ ] `app.py`에 `/Users/dohyun/...` 절대경로 하드코딩됨
      → 실행 환경에 맞게 상단 경로 상수를 반드시 수정해야 함