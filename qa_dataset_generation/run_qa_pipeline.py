"""
QA Dataset Generation Pipeline (Teacher-Student-Judge)
notices_cache.json에서 공지 3개를 로드해 QA 데이터셋 생성
"""

import json
import os
import time
from pathlib import Path
from anthropic import Anthropic

# 프로젝트 루트의 .env 로드 (python-dotenv 없이 직접 파싱)
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

client = Anthropic()

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────
TEACHER_MODEL = "claude-opus-4-6"
STUDENT_MODEL = "claude-haiku-4-5"
JUDGE_MODEL   = "claude-haiku-4-5"

SEED_PER_NOTICE    = 3
STUDENT_PER_NOTICE = 10
QUALITY_THRESHOLD  = 3

NOTICES_CACHE_PATH = "./data/notices_cache.json"
OUTPUT_PATH        = "./data/qa_dataset.jsonl"

# ─────────────────────────────────────────
# Step 1. Teacher로 seed QA 생성
# ─────────────────────────────────────────
TEACHER_SYSTEM = """당신은 대학교 공지사항 기반 QA 데이터셋을 생성하는 전문가입니다.
공지사항을 분석해 학생들이 실제로 물어볼 법한 질문과 정확한 답변을 생성하세요.

반드시 JSON 배열만 출력하세요. 다른 텍스트는 절대 포함하지 마세요."""

TEACHER_PROMPT = """다음 공지사항을 읽고 QA 쌍 {n}개를 생성하세요.

질문 유형 (골고루 포함):
- factual: 날짜, 장소, 대상, 금액 등 사실 확인
- procedural: 신청 방법, 절차, 단계
- conditional: ~한 경우 해당되는지, 자격 조건

공지사항:
{notice}

출력 형식 (JSON 배열):
[
  {{
    "question": "질문",
    "answer": "공지사항에 근거한 정확한 답변",
    "type": "factual|procedural|conditional",
    "source_span": "답변 근거가 되는 원문 구절 (20자 이내)"
  }}
]"""


def _parse_json_response(text: str) -> list:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def generate_seed_qa(notice: str, n: int = SEED_PER_NOTICE) -> list[dict]:
    """Teacher 모델로 고품질 seed QA 생성 (streaming)"""
    with client.messages.stream(
        model=TEACHER_MODEL,
        max_tokens=2000,
        thinking={"type": "adaptive"},
        system=TEACHER_SYSTEM,
        messages=[{"role": "user", "content": TEACHER_PROMPT.format(n=n, notice=notice)}],
    ) as stream:
        final = stream.get_final_message()
    text = next(b.text for b in final.content if b.type == "text")
    return _parse_json_response(text)


# ─────────────────────────────────────────
# Step 2. Student로 대량 생성 (few-shot)
# ─────────────────────────────────────────
STUDENT_SYSTEM = """당신은 대학교 공지사항 기반 QA 데이터셋 생성 어시스턴트입니다.
반드시 JSON 배열만 출력하세요."""

STUDENT_PROMPT = """아래 예시를 참고해 동일한 품질과 형식으로 QA를 생성하세요.
예시와 중복되지 않는 새로운 질문을 만드세요.

[예시]
{seed_examples}

[공지사항]
{notice}

위 공지사항을 기반으로 QA {n}개를 JSON 배열로 생성:"""


def generate_bulk_qa(notice: str, seed_qa: list[dict], n: int = STUDENT_PER_NOTICE) -> list[dict]:
    """Student 모델로 대량 QA 생성 (few-shot)"""
    seed_str = json.dumps(seed_qa[:3], ensure_ascii=False, indent=2)
    with client.messages.stream(
        model=STUDENT_MODEL,
        max_tokens=3000,
        system=STUDENT_SYSTEM,
        messages=[{"role": "user", "content": STUDENT_PROMPT.format(
            seed_examples=seed_str, notice=notice, n=n
        )}],
    ) as stream:
        final = stream.get_final_message()
    text = next(b.text for b in final.content if b.type == "text")
    return _parse_json_response(text)


# ─────────────────────────────────────────
# Step 3. LLM-as-Judge 품질 검증 (2단계)
#   3-A. Hallucination 탐지 — answer가 공지 범위를 벗어나는지
#   3-B. 품질 점수 — 근거 명확성 + 답변 완성도
# ─────────────────────────────────────────
JUDGE_SYSTEM = """당신은 QA 데이터셋 품질 평가 전문가입니다.
반드시 JSON만 출력하세요."""

JUDGE_HALLUCINATION_PROMPT = """공지사항과 아래 QA 쌍을 비교하여 hallucination 여부를 판별하세요.

공지사항:
{notice}

Q: {question}
A: {answer}

판별 기준:
- answer에 공지사항에 없는 사실·수치·날짜·조건이 포함되어 있으면 hallucination
- question이 공지사항 범위를 완전히 벗어난 내용이면 hallucination
- 공지사항에 명시되지 않은 내용을 answer가 추론·가정으로 채우면 hallucination

JSON으로만 출력:
{{
  "hallucination": true|false,
  "hallucinated_span": "공지에 없는 부분 (없으면 null)",
  "reason": "한 줄 이유"
}}"""

JUDGE_QUALITY_PROMPT = """다음 QA 쌍이 공지사항에 얼마나 충실한지 점수를 매기세요.
(hallucination은 이미 별도 검증됐으므로 여기서는 근거 명확성과 답변 완성도만 평가)

공지사항:
{notice}

Q: {question}
A: {answer}

평가 기준:
- 5점: 공지사항 근거 명확, 답변 완전하고 정확
- 4점: 근거 있음, 답변 대체로 정확
- 3점: 근거 있으나 답변 불완전하거나 모호
- 2점: 근거 불명확하거나 답변 부정확
- 1점: 공지사항과 무관하거나 오답

JSON으로만 출력:
{{"score": 1~5, "reason": "한 줄 이유"}}"""


def judge_qa(notice: str, qa: dict) -> dict:
    """LLM-as-Judge 2단계: hallucination 탐지 → 품질 점수"""
    # 3-A: hallucination 탐지
    with client.messages.stream(
        model=JUDGE_MODEL,
        max_tokens=300,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": JUDGE_HALLUCINATION_PROMPT.format(
            notice=notice,
            question=qa["question"],
            answer=qa["answer"],
        )}],
    ) as stream:
        final = stream.get_final_message()
    hall_text = next(b.text for b in final.content if b.type == "text")
    hall_result = _parse_json_response(hall_text)

    # hallucination 확정이면 score=0으로 즉시 반환 (품질 평가 생략)
    if hall_result.get("hallucination"):
        return {
            **qa,
            "hallucination": True,
            "hallucinated_span": hall_result.get("hallucinated_span"),
            "hall_reason": hall_result.get("reason"),
            "judge_score": 0,
            "judge_reason": f"[HALLUCINATION] {hall_result.get('reason')}",
        }

    # 3-B: 품질 점수
    with client.messages.stream(
        model=JUDGE_MODEL,
        max_tokens=200,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": JUDGE_QUALITY_PROMPT.format(
            notice=notice,
            question=qa["question"],
            answer=qa["answer"],
        )}],
    ) as stream:
        final = stream.get_final_message()
    qual_text = next(b.text for b in final.content if b.type == "text")
    qual_result = _parse_json_response(qual_text)

    return {
        **qa,
        "hallucination": False,
        "hallucinated_span": None,
        "hall_reason": None,
        "judge_score": qual_result["score"],
        "judge_reason": qual_result["reason"],
    }


# ─────────────────────────────────────────
# 메인 파이프라인
# ─────────────────────────────────────────
def format_notice(n: dict) -> str:
    return f"제목: {n['title']}\n날짜: {n['date']}\n카테고리: {n['category']}\n\n{n['body']}"


def run_pipeline(notices: list[dict], output_path: str = OUTPUT_PATH):
    all_qa = []
    stats = {"total": 0, "passed": 0, "hallucinated": 0, "low_quality": 0}

    for i, notice_obj in enumerate(notices):
        notice_str = format_notice(notice_obj)
        print(f"\n[{i+1}/{len(notices)}] {notice_obj['title'][:50]}")

        # Step 1: seed 생성
        print("  → Teacher(Opus 4.6): seed QA 생성 중...")
        try:
            seed_qa = generate_seed_qa(notice_str)
            print(f"     seed {len(seed_qa)}개 생성 완료")
        except Exception as e:
            print(f"     ⚠️ seed 생성 실패: {e}")
            continue
        time.sleep(0.5)

        # Step 2: 대량 생성
        print("  → Student(Haiku 4.5): 대량 생성 중...")
        try:
            bulk_qa = generate_bulk_qa(notice_str, seed_qa)
            print(f"     bulk {len(bulk_qa)}개 생성 완료")
        except Exception as e:
            print(f"     ⚠️ bulk 생성 실패: {e}")
            bulk_qa = []
        time.sleep(0.5)

        all_generated = seed_qa + bulk_qa

        # Step 3: Judge 2단계 검증
        print("  → Judge(Haiku 4.5): hallucination 탐지 + 품질 검증 중...")
        for qa in all_generated:
            qa["notice_id"] = i
            qa["notice_title"] = notice_obj["title"]
            try:
                judged = judge_qa(notice_str, qa)
            except Exception as e:
                print(f"     ⚠️ judge 실패: {e}")
                continue

            stats["total"] += 1
            if judged["hallucination"]:
                stats["hallucinated"] += 1
                print(f"     [HALL] {qa['question'][:40]} → {judged['hallucinated_span']}")
            elif judged["judge_score"] >= QUALITY_THRESHOLD:
                all_qa.append(judged)
                stats["passed"] += 1
            else:
                stats["low_quality"] += 1
            time.sleep(0.3)

        print(f"     통과: {stats['passed']} / hallucination: {stats['hallucinated']} / 저품질: {stats['low_quality']} / 전체: {stats['total']}")

    # 저장
    out = Path(output_path)
    with out.open("w", encoding="utf-8") as f:
        for qa in all_qa:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    print(f"\n{'='*50}")
    print(f"완료! 총 {stats['passed']}개 QA 저장 → {output_path}")
    if stats["total"] > 0:
        print(f"통과율:        {stats['passed']}/{stats['total']} ({stats['passed']/stats['total']*100:.1f}%)")
        print(f"Hallucination: {stats['hallucinated']}/{stats['total']} ({stats['hallucinated']/stats['total']*100:.1f}%)")
        print(f"저품질 탈락:   {stats['low_quality']}/{stats['total']} ({stats['low_quality']/stats['total']*100:.1f}%)")
    return all_qa


if __name__ == "__main__":
    data = json.load(open(NOTICES_CACHE_PATH, encoding="utf-8"))
    # body가 충분한 공지 3개 선택
    notices = [n for n in data if len(n.get("body", "")) > 300][:3]
    print(f"선택된 공지 {len(notices)}건:")
    for n in notices:
        print(f"  - {n['title'][:60]}")
    run_pipeline(notices)
