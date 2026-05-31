# notice_processor.py
# 크롤러가 수집한 공지를 Gemini로 분류하고 임베딩 계산 후 Supabase에 저장

import os
import re
import json
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from supabase import create_client

import google.generativeai as genai
from sentence_transformers import SentenceTransformer

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TWO_TOWER_MODEL_PATH = os.getenv("TWO_TOWER_MODEL_PATH", "models/two_tower_model_v4.pt")
BASE_MODEL_EMBED = "jhgan/ko-sroberta-multitask"
BASE_DATE = datetime(2026, 4, 30)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel(
    "gemini-2.5-flash",
    generation_config={"response_mime_type": "application/json"}
)

# ── 신버전 카테고리 정의 ──────────────────────────────────────
VALID_CATEGORIES = [
    "취업/채용", "학사행정", "학생활동/비교과",
    "대외활동", "공모전/경진대회", "국제교류", "창업",
    "장학금", "기숙사", "ROTC"
]

CATEGORY_TYPE_MAP = {
    "취업/채용":      ["경영/금융/사무", "IT/정보통신", "디자인/예술/방송", "공학/기술", "교육/법률/공공", "교내채용"],
    "학생활동/비교과": ["IT/AI/SW", "진로/취업/현장실습", "디자인/콘텐츠", "심리/상담/자기계발", "인문/어학", "행사/생활"],
    "공모전/경진대회": ["IT/AI/SW", "창업/아이디어", "디자인/콘텐츠", "글쓰기/발표/어학", "정책/사회/ESG"],
    "창업":           ["창업"],
    "국제교류":       ["교환학생/파견", "해외인턴/현장실습", "해외연수", "외국인학생/글로벌교류", "해외봉사"],
    "학사행정":       ["수업/수강", "전공/트랙/학적", "성적/졸업", "시험/학사일정", "시설/시스템/행정"],
    "장학금":         ["장학금"],
    "대외활동":       ["봉사활동", "멘토링", "서포터즈/홍보대사", "기획/미디어"],
    "ROTC":          ["ROTC"],
    "기숙사":         ["기숙사"],
}

# ── notice_score 계산 ─────────────────────────────────────────
HALF_LIFE_MAP = {
    "취업/채용":      7,
    "공모전/경진대회": 7,
    "장학금":         7,
    "학생활동/비교과": 14,
    "대외활동":       14,
    "국제교류":       14,
    "창업":           14,
    "ROTC":          30,
    "기숙사":         30,
    "학사행정":       30,
}

def calc_notice_score(notice: dict) -> float:
    try:
        date_str = str(notice.get('posted_at') or notice.get('date', ''))[:10].replace('.', '-')
        posted   = datetime.strptime(date_str, '%Y-%m-%d')
        days     = max((BASE_DATE - posted).days, 1)
    except:
        days = 999

    views    = notice.get('views', 0) or 0
    category = notice.get('category', '기타')
    h        = HALF_LIFE_MAP.get(category, 14)
    alpha    = math.log(2) / h
    return round(math.log(views + 1) * math.exp(-alpha * days), 4)

# ── Gemini 카테고리 + category_type 분류 ─────────────────────
def classify_with_gemini(title: str, body: str, max_retry: int = 3) -> dict:
    prompt = f"""
한성대학교 공지사항을 아래 카테고리 중 하나로 분류하고, 해당 카테고리의 세부 유형도 분류해.
반드시 JSON 형식으로만 답해.

카테고리 목록:
{json.dumps(VALID_CATEGORIES, ensure_ascii=False)}

카테고리별 세부 유형:
{json.dumps(CATEGORY_TYPE_MAP, ensure_ascii=False)}

공지 제목: {title}
공지 본문: {body[:300]}

아래 JSON 형식으로만 출력해:
{{"category": "카테고리명", "category_type": ["세부유형1", "세부유형2"]}}

규칙:
- category는 반드시 위 카테고리 목록 중 하나
- category_type은 해당 카테고리의 세부 유형 중 해당하는 것만 (없으면 빈 배열)
- 장학금/기숙사/ROTC/창업은 category_type이 하나
"""
    for attempt in range(max_retry):
        try:
            res    = gemini.generate_content(prompt)
            result = json.loads(res.text)
            if result.get('category') in VALID_CATEGORIES:
                return result
        except Exception as e:
            print(f"  Gemini 분류 실패 (시도 {attempt+1}/{max_retry}): {e}")
            time.sleep(2)

    # 실패 시 키워드 기반 fallback
    return {"category": "기타", "category_type": []}

# ── Two-Tower 모델 로드 ────────────────────────────────────────
def load_model():
    device = torch.device('cpu')
    sbert  = SentenceTransformer(BASE_MODEL_EMBED, device="cpu")

    HIDDEN_DIM = 256
    OUTPUT_DIM = 128
    SCORE_DIM  = 16

    class TwoTowerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.user_tower = nn.Sequential(
                nn.Linear(768, HIDDEN_DIM), nn.ReLU(),
                nn.Dropout(0.2), nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
            )
            self.item_text_layer  = nn.Sequential(nn.Linear(768, HIDDEN_DIM), nn.ReLU())
            self.item_score_layer = nn.Sequential(nn.Linear(1, SCORE_DIM), nn.ReLU())
            self.item_final_layer = nn.Sequential(
                nn.Dropout(0.2), nn.Linear(HIDDEN_DIM + SCORE_DIM, OUTPUT_DIM)
            )

        def forward_item(self, x):
            text_emb  = x[:, :-1]
            score_emb = x[:, -1:].float()
            text_vec  = self.item_text_layer(text_emb)
            score_vec = self.item_score_layer(score_emb)
            return F.normalize(self.item_final_layer(
                torch.cat([text_vec, score_vec], dim=-1)), dim=-1)

    model = TwoTowerModel().to(device)
    if os.path.exists(TWO_TOWER_MODEL_PATH):
        model.load_state_dict(torch.load(TWO_TOWER_MODEL_PATH, map_location=device))
    model.eval()
    return sbert, model, device

# ── 임베딩 계산 ───────────────────────────────────────────────
def calc_embedding(notice: dict, sbert, model, device, notice_score: float) -> list:
    notice_text = f"{notice.get('category','')} {notice.get('title','')} {(notice.get('body') or '')[:100]}"
    text_emb    = sbert.encode([notice_text], convert_to_numpy=True)

    max_score   = 5.0  # notice_score 최대값 기준 (경험적)
    score_norm  = min(notice_score / max_score, 1.0)

    item_emb    = np.concatenate([text_emb[0], [score_norm]])
    item_tensor = torch.tensor(item_emb, dtype=torch.float).unsqueeze(0).to(device)

    with torch.no_grad():
        item_vec = model.forward_item(item_tensor).cpu().numpy()[0]

    return item_vec.tolist()

# ── 메인 처리 함수 ────────────────────────────────────────────
def process_notices(notices: list[dict], batch_size: int = 10) -> int:
    """
    크롤러가 수집한 공지 리스트를 받아서:
    1. Gemini로 카테고리 + category_type 분류
    2. notice_score 계산
    3. Two-Tower 임베딩 계산
    4. Supabase upsert
    반환: 처리된 공지 수
    """
    print(f"공지 처리 시작: {len(notices)}건")
    sbert, model, device = load_model()
    print("모델 로드 완료!")

    success = 0
    failed  = 0

    for i, notice in enumerate(notices):
        try:
            title = notice.get('title', '')
            body  = notice.get('body', '') or ''
            url   = notice.get('url', '')

            print(f"[{i+1}/{len(notices)}] {title[:40]}")

            # 1. Gemini 분류
            classification = classify_with_gemini(title, body)
            category       = classification.get('category', '기타')
            category_type  = classification.get('category_type', [])

            notice['category'] = category

            # 2. notice_score 계산
            notice_score = calc_notice_score(notice)

            # 3. 임베딩 계산
            embedding = calc_embedding(notice, sbert, model, device, notice_score)

            # 4. notice_id 추출
            notice_id_match = re.search(r'/(\d+)/artclView\.do', url)
            notice_id       = notice_id_match.group(1) if notice_id_match else None

            # 5. Supabase upsert
            supabase.table("notices").upsert({
                "source":          "hansung",
                "notice_id":       notice_id,
                "title":           title,
                "url":             url,
                "posted_at":       notice.get('date', '')[:10].replace('.', '-') if notice.get('date') else None,
                "posted_date_text": notice.get('date'),
                "category":        category,
                "category_type":   category_type,
                "job_types":       category_type,  # 동일하게 사용
                "body":            body,
                "views":           notice.get('views', 0),
                "notice_score":    notice_score,
                "embedding":       json.dumps(embedding),
                "raw":             notice,
            }, on_conflict="url").execute()

            print(f"  → {category} {category_type} score:{notice_score}")
            success += 1

        except Exception as e:
            print(f"  실패: {e}")
            failed += 1
            import traceback; traceback.print_exc()

        time.sleep(1)  # Gemini API rate limit 방지

    print(f"\n처리 완료! 성공: {success}개 | 실패: {failed}개")
    if failed > 0:
        raise RuntimeError(
            f"Notice processing failed for {failed}/{len(notices)} notices "
            f"(success={success}). Check the errors above."
        )
    return success


# ── 단독 실행 (테스트용) ──────────────────────────────────────
if __name__ == "__main__":
    res = supabase.table("notices").select(
        "id,notice_id,title,url,posted_at,body,views,category,category_type,embedding"
    ).order("posted_at", desc=True).limit(50).execute()

    unprocessed = [
        n for n in (res.data or [])
        if not n.get('category_type') or not n.get('embedding')
    ]

    if unprocessed:
        print(f"미처리 공지 {len(unprocessed)}건 발견")
        process_notices(unprocessed)
    else:
        print("미처리 공지 없음")
