# Sangsangfinder

한성대학교 공지사항 검색 및 추천 시스템 — RAG(Retrieval-Augmented Generation) 파이프라인 기반

## 주요 기능

- 한성대 공지사항 RSS 크롤링 및 ChromaDB 적재 (주기적 크롤링: 우선 x)
- Hybrid Search: BM25(0.3) + Dense Vector(0.7) 결합 검색
- QA 생성-Judge 구조로 QA 데이터셋 자동 생성
- Streamlit UI를 통한 공지사항 검색 인터페이스

## 기술 스택

| 구분 | 내용 |
|------|------|
| 임베딩 모델 | `jhgan/ko-sroberta-multitask` |
| 검색 방식 | Hybrid (BM25 0.3 + Dense 0.7) |
| Vector DB | ChromaDB (Pinecone 마이그레이션 예정) |
| UI | Streamlit |
| QA 생성 | Claude API (Opus / Haiku) |
| 배포 | Streamlit Community Cloud (예정) |

## 디렉토리 구조

```
sangsangfinder/
├── app.py                      # Streamlit 검색 UI
├── rss_monitor.py              # RSS 크롤링 및 ChromaDB 적재
├── embedding_search_v1.py      # Hybrid search 구현
├── requirements.txt
├── state/
│   └── seen_items.json         # 처리된 RSS 항목 상태
├── chroma_db/                  # 임베딩 벡터 저장소
└── qa_dataset_generation/
    ├── run_qa_pipeline.py      # QA 데이터셋 생성 파이프라인
    ├── train.py                # 파인튜닝 스크립트
    ├── train_guide.ipynb       # 학습 가이드 노트북
    └── data/
        └── qa_dataset.jsonl    # 생성된 QA 데이터셋
```

## 로컬 실행

### 1. Python 3.11 가상환경 생성

```bash
python3.11 -m venv .venv311
```

> Python 3.12에서는 `torch==2.1.0` 호환 문제로 **Python 3.11 권장**

### 2. 의존성 설치

```bash
.venv311/bin/pip install -r requirements.txt
```

### 3. 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성하고 필요한 API 키를 설정합니다.

### 4. RSS 모니터 실행

```bash
.venv311/bin/python -u rss_monitor.py
```

- 첫 실행: RSS 최근 50건을 기준 상태로만 저장하고 종료
- 이후 실행: 새 공지가 있을 때만 본문 크롤링 및 임베딩 저장

### 5. Streamlit UI 실행

```bash
.venv311/bin/streamlit run app.py
```

## QA 데이터셋 생성

Teacher-Student-Judge 구조로 공지사항 기반 QA 쌍을 자동 생성합니다.

```bash
.venv311/bin/python qa_dataset_generation/run_qa_pipeline.py
```

| 역할 | 모델 |
|------|------|
| Teacher (문제 생성) | claude-opus-4-6 |
| Student (답변 생성) | claude-haiku-4-5 |
| Judge (품질 평가) | claude-haiku-4-5 |

생성 결과는 `qa_dataset_generation/data/qa_dataset.jsonl`에 저장됩니다.

## 파인튜닝

```bash
.venv311/bin/python qa_dataset_generation/train.py
```

자세한 내용은 [train_guide.ipynb](qa_dataset_generation/train_guide.ipynb) 참고

## GitHub Actions

- 워크플로우: [.github/workflows/rss-monitor.yml](.github/workflows/rss-monitor.yml)
- 10분마다 RSS를 확인하고, 상태 파일과 DB 변경분을 자동 커밋

## 상태 파일

- `state/seen_items.json` — 이미 처리한 RSS 항목 목록
- `chroma_db/` — 임베딩된 공지 벡터 데이터
- `.cache/` — 임베딩 모델 캐시 (첫 실행 시 자동 다운로드)

## 주의사항

- `.env` 파일은 절대 커밋하지 않습니다 (API 키 포함)
- 평가 시 train/test split을 반드시 명시합니다
- 임베딩 모델 최초 다운로드 시 시간이 소요될 수 있습니다
