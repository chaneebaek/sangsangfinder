"""
QA Dataset Generation Pipeline (Teacher-Student-Judge)
2026_notice.json에서 공지 3개를 로드해 QA 데이터셋 생성
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
TEACHER_MODEL = "claude-sonnet-4-6"
JUDGE_MODEL   = "claude-haiku-4-5"

QUALITY_THRESHOLD = 3


def get_qa_count(body: str) -> tuple[int, int]:
    """본문 길이에 따라 (well-formed 수, 유저형 수) 반환"""
    if len(body) < 300:
        return 2, 1  # 총 3개
    return 3, 2      # 총 5개

NOTICES_CACHE_PATH = "./data/2026_notice.json"
OUTPUT_PATH        = "./data/qa_dataset.jsonl"

# ─────────────────────────────────────────
# Step 1. Teacher로 seed QA 생성
# ─────────────────────────────────────────
TEACHER_SYSTEM = """당신은 대학교 공지사항 기반 QA 데이터셋을 생성하는 전문가입니다.
공지사항을 분석해 학생들이 실제로 물어볼 법한 질문과 정확한 답변을 생성하세요.

반드시 JSON 배열만 출력하세요. 다른 텍스트는 절대 포함하지 마세요."""

TEACHER_PROMPT = """다음 공지사항을 읽고 QA 쌍을 생성하세요.

well-formed {n_well}개 (type: factual / procedural / conditional):
- 주어·목적어·서술어를 갖춘 완전한 문장 형식
- 예: "장학금 신청 기간은 언제까지인가요?"

user-style {n_user}개 (type: user_short):
- 실제 학생이 챗봇에 입력하는 짧고 구어체적인 질문
- 주어 생략, 반말, 핵심 키워드 위주
- 예: "장학금 언제까지야?", "장학금 신청 마감", "신청 방법 알려줘"

공지사항:
{notice}

출력 형식 (JSON 배열, 총 {n_total}개):
[
  {{
    "question": "질문",
    "answer": "공지사항에 근거한 정확한 답변",
    "type": "factual|procedural|conditional|user_short",
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


def generate_seed_qa(notice: str, n_well: int, n_user: int) -> list[dict]:
    """Teacher 모델로 고품질 seed QA 생성 (streaming)"""
    with client.messages.stream(
        model=TEACHER_MODEL,
        max_tokens=2000,
        system=TEACHER_SYSTEM,
        messages=[{"role": "user", "content": TEACHER_PROMPT.format(
            n_well=n_well, n_user=n_user, n_total=n_well + n_user, notice=notice,
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


def load_processed_titles(output_path: str) -> set[str]:
    """이미 처리된 공지 제목 집합 반환 (resume용)"""
    processed = set()
    p = Path(output_path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "notice_title" in obj:
                    processed.add(obj["notice_title"])
            except json.JSONDecodeError:
                pass
    return processed


def run_pipeline(notices: list[dict], output_path: str = OUTPUT_PATH):
    # 출력 파일이 이미 존재하면 append로 열어 기존 결과를 보존
    processed_titles = load_processed_titles(output_path)
    if processed_titles:
        print(f"[Resume] 기존 처리 공지 {len(processed_titles)}건 감지 → 이어서 실행합니다.")

    all_qa = []
    stats = {"passed": 0, "total": 0, "hallucinated": 0, "low_quality": 0}
    out_file = open(Path(output_path), "a", encoding="utf-8")

    try:
        for i, notice_obj in enumerate(notices):
            if notice_obj["title"] in processed_titles:
                print(f"\n[{i+1}/{len(notices)}] 건너뜀 (이미 처리됨): {notice_obj['title'][:50]}", flush=True)
                continue
            notice_str = format_notice(notice_obj)
            print(f"\n[{i+1}/{len(notices)}] {notice_obj['title'][:50]}", flush=True)

            # Step 1: seed 생성
            n_well, n_user = get_qa_count(notice_obj.get("body", ""))
            print(f"  → Teacher(Sonnet 4.6): seed QA 생성 중... (well-formed {n_well} + 유저형 {n_user})", flush=True)
            try:
                seed_qa = generate_seed_qa(notice_str, n_well, n_user)
                print(f"     seed {len(seed_qa)}개 생성 완료", flush=True)
            except Exception as e:
                print(f"     ⚠️ seed 생성 실패: {e}", flush=True)
                continue
            time.sleep(0.5)

            all_generated = seed_qa

            # Step 3: Judge 2단계 검증 (비활성화)
            # print("  → Judge(Haiku 4.5): hallucination 탐지 + 품질 검증 중...", flush=True)
            # for qa in all_generated:
            #     qa["notice_id"] = i
            #     qa["notice_title"] = notice_obj["title"]
            #     try:
            #         judged = judge_qa(notice_str, qa)
            #     except Exception as e:
            #         print(f"     ⚠️ judge 실패: {e}", flush=True)
            #         continue
            #     stats["total"] += 1
            #     if judged["hallucination"]:
            #         stats["hallucinated"] += 1
            #         print(f"     [HALL] {qa['question'][:40]} → {judged['hallucinated_span']}", flush=True)
            #     elif judged["judge_score"] >= QUALITY_THRESHOLD:
            #         all_qa.append(judged)
            #         stats["passed"] += 1
            #         out_file.write(json.dumps(judged, ensure_ascii=False) + "\n")
            #         out_file.flush()
            #     else:
            #         stats["low_quality"] += 1
            #     time.sleep(0.3)
            # print(f"     통과: {stats['passed']} / hallucination: {stats['hallucinated']} / 저품질: {stats['low_quality']} / 전체: {stats['total']}", flush=True)

            for qa in all_generated:
                qa["notice_id"] = i
                qa["notice_title"] = notice_obj["title"]
                all_qa.append(qa)
                out_file.write(json.dumps(qa, ensure_ascii=False) + "\n")
                out_file.flush()
                stats["passed"] += 1

            print(f"     누적 저장: {stats['passed']}개", flush=True)
    finally:
        out_file.close()

    print(f"\n{'='*50}")
    print(f"완료! 총 {stats['passed']}개 QA 저장 → {output_path}")
    # if stats["total"] > 0:
    #     print(f"통과율:        {stats['passed']}/{stats['total']} ({stats['passed']/stats['total']*100:.1f}%)")
    #     print(f"Hallucination: {stats['hallucinated']}/{stats['total']} ({stats['hallucinated']/stats['total']*100:.1f}%)")
    #     print(f"저품질 탈락:   {stats['low_quality']}/{stats['total']} ({stats['low_quality']/stats['total']*100:.1f}%)")
    return all_qa


if __name__ == "__main__":
    data = json.load(open(NOTICES_CACHE_PATH, encoding="utf-8"))
    # body가 충분한 공지 3개 선택
    notices = [n for n in data if len(n.get("body", "")) > 300][:3]
    print(f"선택된 공지 {len(notices)}건:")
    for n in notices:
        print(f"  - {n['title'][:60]}")
    run_pipeline(notices)  # 출력 파일 존재 시 자동 resume
