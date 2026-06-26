"""
qa_dataset_all.jsonl 전체에 대해 Gemini 2.5 Flash Judge 실행
결과: data/qa_judged_all_gemini.jsonl  (append + resume 지원)
"""

import json
import os
import re
import time
from pathlib import Path
from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig

# .env 로드
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

judge_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
JUDGE_MODEL  = "gemini-2.5-flash"

INFERRED_RATIO_WARN = 0.2

# ── 프롬프트 ──────────────────────────────────────
JUDGE_SYSTEM = """당신은 QA 데이터셋 품질 평가 전문가입니다.
반드시 JSON만 출력하세요."""

JUDGE_CLAIM_EXTRACT_PROMPT = """아래 답변에서 공지사항과 대조해야 할 사실적 주장을 모두 추출하세요.
날짜, 금액, 인원, 조건, 장소, URL 등 구체적 수치·정보 위주로 추출합니다.
일반적 설명문("신청하실 수 있습니다" 등)은 제외합니다.

A: {answer}
근거 원문 구절(Teacher 제공, 참고용): {source_span}

JSON 배열로만 출력 (주장이 없으면 빈 배열):
["주장1", "주장2", ...]"""

JUDGE_VERIFY_CLAIMS_PROMPT = """공지사항에 아래 사실적 주장들이 각각 명시되어 있는지 확인하세요.

공지사항:
{notice}

Q: {question}
A: {answer}

확인할 주장:
{claims_numbered}

각 주장을 아래 기준으로 분류하세요:
- "verified": 공지사항에 명확히 있음
- "not_found": 공지사항에 없음 → hallucination
- "inferred": 공지사항에서 합리적으로 추론 가능하나 명시 안 됨 → hallucination 아님

JSON으로만 출력:
{{
  "results": [
    {{"claim": "주장", "status": "verified|not_found|inferred", "evidence": "공지 원문 근거 (없으면 null)"}}
  ],
  "hallucination": true|false,
  "hallucinated_claims": ["not_found 판정된 주장들"],
  "reason": "한 줄 요약"
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


# ── 유틸 ──────────────────────────────────────────
def _parse_json_response(text: str):
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        last_bracket = text.rfind("]")
        last_brace = text.rfind("}")
        cutoff = max(last_bracket, last_brace)
        if cutoff == -1:
            raise
        closer = "]" if last_bracket > last_brace else "}"
        parsed = json.loads(text[:cutoff + 1] + (closer if text[cutoff] != closer else ""))
    if isinstance(parsed, dict) and "score" in parsed:
        parsed["score"] = int(str(parsed["score"]).strip())
    return parsed


def _call_judge(prompt: str, max_tokens: int) -> dict:
    for attempt in range(5):
        try:
            response = judge_client.models.generate_content(
                model=JUDGE_MODEL,
                contents=f"{JUDGE_SYSTEM}\n\n{prompt}",
                config=GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    thinking_config=ThinkingConfig(thinking_budget=0),
                ),
            )
            return _parse_json_response(response.text)
        except Exception as e:
            msg = str(e)
            if "429" in msg:
                m = re.search(r"retryDelay.*?(\d+)s", msg)
                wait = int(m.group(1)) + 2 if m else 60
                print(f"     ⏳ Rate limit — {wait}초 대기 ({attempt+1}/5)", flush=True)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Judge 호출 5회 재시도 실패")


def judge_qa(notice: str, qa: dict) -> dict:
    question, answer = qa["question"], qa["answer"]
    source_span = qa.get("source_span", "")

    # 3-A: claim 추출
    claims = _call_judge(
        JUDGE_CLAIM_EXTRACT_PROMPT.format(answer=answer, source_span=source_span),
        max_tokens=300,
    )
    if not isinstance(claims, list):
        claims = []

    inferred_claims: list = []
    needs_review: bool = False
    unchecked: bool = False

    if claims:
        claims_numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
        verify_result = _call_judge(
            JUDGE_VERIFY_CLAIMS_PROMPT.format(
                notice=notice,
                question=question,
                answer=answer,
                claims_numbered=claims_numbered,
            ),
            max_tokens=1500,
        )
        is_hallucinated = verify_result.get("hallucination", False)
        hallucinated_claims = verify_result.get("hallucinated_claims", [])
        hall_reason = verify_result.get("reason", "")

        results = verify_result.get("results", [])
        inferred_claims = [r["claim"] for r in results if r.get("status") == "inferred"]
        if results:
            inferred_ratio = len(inferred_claims) / len(results)
            if inferred_ratio > INFERRED_RATIO_WARN:
                needs_review = True
    else:
        is_hallucinated = False
        hallucinated_claims = []
        hall_reason = "사실적 주장 없음 — 검증 대상 없음"
        unchecked = True

    if is_hallucinated:
        return {
            **qa,
            "hallucination": True,
            "hallucinated_claims": hallucinated_claims,
            "inferred_claims": inferred_claims,
            "hall_reason": hall_reason,
            "judge_score": 0,
            "judge_reason": f"[HALLUCINATION] {hall_reason}",
            "unchecked": unchecked,
            "needs_review": needs_review,
        }

    qual_result = _call_judge(
        JUDGE_QUALITY_PROMPT.format(notice=notice, question=question, answer=answer),
        max_tokens=200,
    )
    return {
        **qa,
        "hallucination": False,
        "hallucinated_claims": [],
        "inferred_claims": inferred_claims,
        "hall_reason": None,
        "judge_score": qual_result.get("score", 0),
        "judge_reason": qual_result.get("reason", ""),
        "unchecked": unchecked,
        "needs_review": needs_review,
    }


# ── 메인 ──────────────────────────────────────────
DATA_DIR    = Path(__file__).parent / "data"
INPUT_PATH  = DATA_DIR / "qa_dataset_all.jsonl"
OUTPUT_PATH = DATA_DIR / "qa_judged_all_gemini.jsonl"
NOTICE_PATH = DATA_DIR / "2026_notice.json"

def format_notice(n: dict) -> str:
    return f"제목: {n['title']}\n날짜: {n.get('date','')}\n카테고리: {n.get('category','')}\n\n{n.get('body','')}"

def load_processed_questions(path: Path) -> set:
    """이미 처리된 question set 반환 (resume용)"""
    processed = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                processed.add(json.loads(line)["question"])
            except (json.JSONDecodeError, KeyError):
                pass
    return processed


if __name__ == "__main__":
    with open(INPUT_PATH, encoding="utf-8") as f:
        all_rows = [json.loads(line) for line in f]

    with open(NOTICE_PATH, encoding="utf-8") as f:
        notices = json.load(f)

    processed = load_processed_questions(OUTPUT_PATH)
    remaining = [r for r in all_rows if r["question"] not in processed]

    total = len(all_rows)
    skip  = total - len(remaining)
    print(f"전체 {total}개 | 이미 처리 {skip}개 | 남은 {len(remaining)}개")

    stats = {"passed": 0, "hallucinated": 0, "low_quality": 0, "error": 0}
    QUALITY_THRESHOLD = 4

    with open(OUTPUT_PATH, "a", encoding="utf-8") as out:
        for idx, qa in enumerate(remaining, start=skip + 1):
            notice_id  = qa.get("notice_id", 0)
            notice_str = format_notice(notices[notice_id])

            print(f"[{idx}/{total}] {qa['question'][:50]}", flush=True)
            try:
                result = judge_qa(notice_str, qa)
            except Exception as e:
                print(f"  ⚠️ 실패: {e}", flush=True)
                stats["error"] += 1
                continue

            if result["hallucination"]:
                stats["hallucinated"] += 1
                print(f"  → [HALL] {result.get('hallucinated_claims', [])}", flush=True)
            elif result["judge_score"] >= QUALITY_THRESHOLD:
                stats["passed"] += 1
                print(f"  → score={result['judge_score']} ✓", flush=True)
            else:
                stats["low_quality"] += 1
                print(f"  → score={result['judge_score']} 저품질 탈락", flush=True)

            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            time.sleep(0.3)

    done = skip + len(remaining) - stats["error"]
    print(f"\n{'='*50}")
    print(f"완료! → {OUTPUT_PATH}")
    print(f"통과:          {stats['passed']}개")
    print(f"Hallucination: {stats['hallucinated']}개")
    print(f"저품질 탈락:   {stats['low_quality']}개")
    print(f"오류 스킵:     {stats['error']}개")
    if done > 0:
        print(f"통과율: {stats['passed']}/{done} ({stats['passed']/done*100:.1f}%)")
