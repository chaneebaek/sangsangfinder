"""
QA Dataset Generation Pipeline (Teacher-Student)
sangsangfinder 캡스톤 프로젝트용
"""

import json
import time
from pathlib import Path
from anthropic import Anthropic

client = Anthropic()

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────
TEACHER_MODEL = "claude-sonnet-4-5"      # 고품질 seed 생성
STUDENT_MODEL = "claude-haiku-4-5-20251001"  # 대량 생성
JUDGE_MODEL   = "claude-haiku-4-5-20251001"  # 자동 검증

SEED_PER_NOTICE   = 3    # Teacher: 공지사항 1개당 QA 생성 수
STUDENT_PER_NOTICE = 10  # Student: 공지사항 1개당 QA 생성 수
QUALITY_THRESHOLD  = 3   # Judge 점수 기준 (1~5)

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


def generate_seed_qa(notice: str, n: int = SEED_PER_NOTICE) -> list[dict]:
    """Teacher 모델로 고품질 seed QA 생성"""
    resp = client.messages.create(
        model=TEACHER_MODEL,
        max_tokens=2000,
        system=TEACHER_SYSTEM,
        messages=[{
            "role": "user",
            "content": TEACHER_PROMPT.format(n=n, notice=notice)
        }]
    )
    text = resp.content[0].text.strip()
    # 코드블록 제거
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


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
    resp = client.messages.create(
        model=STUDENT_MODEL,
        max_tokens=3000,
        system=STUDENT_SYSTEM,
        messages=[{
            "role": "user",
            "content": STUDENT_PROMPT.format(
                seed_examples=seed_str,
                notice=notice,
                n=n
            )
        }]
    )
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


# ─────────────────────────────────────────
# Step 3. LLM-as-Judge 품질 검증
# ─────────────────────────────────────────
JUDGE_SYSTEM = """당신은 QA 데이터셋 품질 평가 전문가입니다.
반드시 JSON만 출력하세요."""

JUDGE_PROMPT = """다음 QA 쌍이 공지사항 내용에 충실한지 평가하세요.

공지사항:
{notice}

Q: {question}
A: {answer}

평가 기준:
- 5점: 공지사항 근거 명확, 답변 완전하고 정확
- 4점: 근거 있음, 답변 대체로 정확
- 3점: 근거 있으나 답변 불완전
- 2점: 근거 불명확하거나 답변 부정확
- 1점: 공지사항과 무관하거나 오답

JSON으로만 출력:
{{"score": 1~5, "reason": "한 줄 이유"}}"""


def judge_qa(notice: str, qa: dict) -> dict:
    """LLM-as-Judge로 QA 품질 점수 부여"""
    resp = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=200,
        system=JUDGE_SYSTEM,
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(
                notice=notice,
                question=qa["question"],
                answer=qa["answer"]
            )
        }]
    )
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    result = json.loads(text.strip())
    return {**qa, "judge_score": result["score"], "judge_reason": result["reason"]}


# ─────────────────────────────────────────
# 메인 파이프라인
# ─────────────────────────────────────────
def run_pipeline(notices: list[str], output_path: str = "qa_dataset.jsonl"):
    """전체 파이프라인 실행"""
    all_qa = []
    stats = {"total": 0, "passed": 0, "failed": 0}

    for i, notice in enumerate(notices):
        print(f"\n[{i+1}/{len(notices)}] 공지사항 처리 중...")

        # Step 1: seed 생성
        print("  → Teacher: seed QA 생성 중...")
        seed_qa = generate_seed_qa(notice)
        print(f"     seed {len(seed_qa)}개 생성 완료")
        time.sleep(0.5)

        # Step 2: 대량 생성
        print("  → Student: 대량 생성 중...")
        bulk_qa = generate_bulk_qa(notice, seed_qa)
        print(f"     bulk {len(bulk_qa)}개 생성 완료")
        time.sleep(0.5)

        # 전체 QA (seed + bulk)
        all_generated = seed_qa + bulk_qa

        # Step 3: Judge 검증
        print("  → Judge: 품질 검증 중...")
        for qa in all_generated:
            qa["notice_id"] = i
            judged = judge_qa(notice, qa)
            stats["total"] += 1

            if judged["judge_score"] >= QUALITY_THRESHOLD:
                all_qa.append(judged)
                stats["passed"] += 1
            else:
                stats["failed"] += 1
            time.sleep(0.2)

        print(f"     통과: {stats['passed']} / 전체: {stats['total']}")

    # 저장
    output = Path(output_path)
    with output.open("w", encoding="utf-8") as f:
        for qa in all_qa:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    print(f"\n{'='*40}")
    print(f"완료! 총 {stats['passed']}개 QA 저장 → {output_path}")
    print(f"통과율: {stats['passed']}/{stats['total']} ({stats['passed']/stats['total']*100:.1f}%)")
    return all_qa
