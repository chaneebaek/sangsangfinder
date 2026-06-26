# ============================================================
# recommend.py — Two-Tower 추천 시스템 로직 (v4)
# ============================================================

import os
import re
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from sentence_transformers import SentenceTransformer
from supabase import create_client

import streamlit as st

# ── 경로 설정 ─────────────────────────────────────────────────
_BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
TWO_TOWER_MODEL_PATH = os.path.join(_BASE_DIR, "models", "two_tower_model_v4.pt")
BASE_MODEL_EMBED     = "jhgan/ko-sroberta-multitask"
SUPABASE_URL         = os.getenv("SUPABASE_URL", "your_supabase_url")
SUPABASE_KEY         = os.getenv("SUPABASE_KEY", "your_supabase_key")

# ── 추천 가중치 ───────────────────────────────────────────────
MODEL_WEIGHT    = 0.4
CATEGORY_WEIGHT = 0.4
JOB_TYPE_WEIGHT = 0.1
SCORE_WEIGHT    = 0.1

PENALTY_CATEGORIES     = ['국제교류', '대외활동', '창업']
NO_JOB_TYPE_CATEGORIES = ['국제교류', '학사행정', '대외활동', 'ROTC', '기숙사']
FILTER_CATEGORIES      = ['장학금', '기숙사', 'ROTC']
OLD_CATEGORIES = ['비교과', '교육/특강', '봉사/서포터즈', '인턴십', '인턴쉽', '기타', '학자금/근로장학']

# ── 신버전 카테고리 ───────────────────────────────────────────
VALID_CATEGORIES = [
    '취업/채용', '학사행정', '학생활동/비교과',
    '대외활동', '공모전/경진대회', '국제교류', '창업',
    '장학금', '기숙사', 'ROTC'
]

TRACK_DOMAIN = {
    "IT": [
        "모바일소프트웨어트랙", "빅데이터트랙", "디지털콘텐츠ㆍ가상현실트랙",
        "웹공학트랙", "전자트랙", "시스템반도체트랙", "기계시스템디자인트랙",
        "AI로봇융합트랙", "산업공학트랙", "응용산업데이터공학트랙",
        "AI응용학과", "융합보안학과", "미래모빌리티학과",
        "SW융합학과", "글로벌벤처창업학과",
        "AIㆍ소프트웨어학과", "ICT융합디자인학과", "스마트제조혁신컨설팅학과",
    ],
    "경영": [
        "기업경영트랙", "회계ㆍ재무경영트랙", "경제금융투자트랙",
        "기업ㆍ경제분석트랙", "비지니스애널리틱스트랙",
        "국제무역트랙", "글로벌비지니스트랙",
        "글로벌K비지니스학과", "비지니스컨설팅학과", "호텔외식경영학과",
    ],
    "행정/공공": [
        "공공행정트랙", "법&정책트랙", "부동산트랙",
        "스마트도시ㆍ교통계획트랙", "융합행정학과",
    ],
    "디자인": [
        "패션마케팅트랙", "패션디자인트랙", "패션크리에이티브디렉션트랙",
        "미디어디자인트랙", "시각디자인트랙", "영상ㆍ애니메이션디자인트랙",
        "UX/UI디자인트랙", "인테리어디자인트랙", "VMDㆍ전시디자인트랙",
        "게임그래픽디자인트랙", "뷰티디자인매니지먼트학과",
        "패션뷰티크리에이션학과", "영상엔터테인먼트학과",
        "뷰티디자인학과", "뷰티매니지먼트학과",
        "디지털콘텐츠디자인학과", "인테리어디자인학과",
    ],
    "인문": [
        "영미문화콘텐츠트랙", "영미언어정보트랙", "한국어교육트랙",
        "역사문화큐레이션트랙", "역사콘텐츠트랙", "지식정보문화트랙",
        "디지털인문정보학트랙", "문학문화콘텐츠학과", "한국언어문화교육학과",
    ],
    "예술": [
        "동양화전공", "서양화전공",
        "한국무용전공", "현대무용전공", "발레전공",
    ],
    "융합": ["상상력인재학부"],
}

track_to_domain = {}
for _domain, _tracks in TRACK_DOMAIN.items():
    for _track in _tracks:
        track_to_domain[_track] = _domain
track_to_domain["트랙 미정"] = "융합"
track_to_domain["상상력인재학부 (트랙 미정)"] = "융합"

DOMAIN_TO_JOB_TYPE = {
    "취업/채용": {
        "IT":        "IT/정보통신",
        "경영":      "경영/금융/사무",
        "행정/공공": "교육/법률/공공",
        "디자인":    "디자인/예술/방송",
        "인문":      "교육/법률/공공",
        "예술":      "디자인/예술/방송",
        "융합":      None,
    },
    "학생활동/비교과": {
        "IT":        "IT/AI/SW",
        "경영":      "진로/취업/현장실습",
        "행정/공공": "진로/취업/현장실습",
        "디자인":    "디자인/콘텐츠",
        "인문":      "인문/어학",
        "예술":      "디자인/콘텐츠",
        "융합":      None,
    },
    "공모전/경진대회": {
        "IT":        "IT/AI/SW",
        "경영":      "창업/아이디어",
        "행정/공공": "정책/사회/ESG",
        "디자인":    "디자인/콘텐츠",
        "인문":      "글쓰기/발표/어학",
        "예술":      "디자인/콘텐츠",
        "융합":      None,
    },
    "창업": {
        "IT":        "창업",
        "경영":      "창업",
        "행정/공공": "창업",
        "디자인":    "창업",
        "인문":      "창업",
        "예술":      "창업",
        "융합":      None,
    },
    "국제교류": {
        "IT":        "교환학생/파견",
        "경영":      "교환학생/파견",
        "행정/공공": "교환학생/파견",
        "디자인":    "교환학생/파견",
        "인문":      "외국인학생/글로벌교류",
        "예술":      "교환학생/파견",
        "융합":      "교환학생/파견",
    },
    "대외활동": {
        "IT":        "멘토링",
        "경영":      "서포터즈/홍보대사",
        "행정/공공": "봉사활동",
        "디자인":    "기획/미디어",
        "인문":      "봉사활동",
        "예술":      "기획/미디어",
        "융합":      None,
    },
}

# ============================================================
# Supabase
# ============================================================

@st.cache_resource
def get_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

@st.cache_data(ttl=300, show_spinner=False)
def load_notices_from_supabase() -> list:
    try:
        supabase  = get_supabase()
        all_data  = []
        page_size = 1000
        offset    = 0
        while True:
            res = supabase.table("notices").select(
                "id,notice_id,title,url,posted_at,posted_date_text,category,body,views,job_types,category_type,notice_score"
            ).order("posted_at", desc=True).range(offset, offset + page_size - 1).execute()
            batch = res.data or []
            if not batch: break
            all_data.extend(batch)
            if len(batch) < page_size: break
            offset += page_size
        for n in all_data:
            if not n.get('notice_score'):
                n['notice_score'] = 0.0
            raw = str(n.get('posted_date_text') or n.get('posted_at', ''))
            raw = re.sub(r'<[^>]+>', '', raw).strip()
            n['date'] = raw[:10].replace('.', '-')
        print(f"Supabase 공지 {len(all_data)}건 로드 완료")
        return all_data
    except Exception as e:
        print(f"Supabase 로드 오류: {e}")
        import traceback; traceback.print_exc()
        return []

@st.cache_data(ttl=300, show_spinner=False)
def load_embeddings_from_supabase() -> dict:
    try:
        supabase  = get_supabase()
        all_embs  = []
        page_size = 1000
        offset    = 0
        while True:
            res = supabase.table("notices").select(
                "id,embedding"
            ).order("posted_at", desc=True).range(offset, offset + page_size - 1).execute()
            batch = res.data or []
            if not batch: break
            all_embs.extend(batch)
            if len(batch) < page_size: break
            offset += page_size

        def parse_embedding(emb):
            if emb is None: return None
            if isinstance(emb, str): emb = json.loads(emb)
            return np.array(emb, dtype=np.float32)

        result = {r['id']: parse_embedding(r.get('embedding')) for r in all_embs}
        valid  = {k: v for k, v in result.items() if v is not None}
        print(f"임베딩 {len(valid)}건 로드 완료")
        return valid
    except Exception as e:
        print(f"임베딩 로드 오류: {e}")
        return {}

# ============================================================
# Two-Tower 모델 (v4)
# ============================================================

@st.cache_resource
def load_two_tower_model():
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

        def forward_user(self, x):
            return F.normalize(self.user_tower(x), dim=-1)

        def forward_item(self, x):
            text_emb  = x[:, :-1]
            score_emb = x[:, -1:].float()
            text_vec  = self.item_text_layer(text_emb)
            score_vec = self.item_score_layer(score_emb)
            return F.normalize(self.item_final_layer(
                torch.cat([text_vec, score_vec], dim=-1)), dim=-1)

        def forward(self, u, i):
            return self.forward_user(u), self.forward_item(i)

    model = TwoTowerModel().to(device)
    if os.path.exists(TWO_TOWER_MODEL_PATH):
        model.load_state_dict(torch.load(TWO_TOWER_MODEL_PATH, map_location=device))
    model.eval()
    print("Two-Tower 모델 v4 로드 완료!")
    return sbert, model, device

# ============================================================
# job_type / 도메인
# ============================================================

def get_user_domain(track: str) -> str:
    if "상상력인재학부" in track:
        for t, d in track_to_domain.items():
            if t in track and t != "상상력인재학부":
                return d
        return "융합"
    return track_to_domain.get(track, "융합")

def classify_job_type(notice: dict, threshold: float = 0.35, top_k: int = 2) -> list:
    """Supabase category_type 컬럼 직접 사용"""
    job_types = notice.get('job_types') or notice.get('category_type') or []
    if isinstance(job_types, str):
        try:
            job_types = json.loads(job_types)
        except:
            job_types = []
    return [{'job_type': jt, 'score': 1.0} for jt in job_types[:top_k]]

def get_job_score(track: str, notice: dict) -> float:
    cat = notice.get('category', '')
    if cat in NO_JOB_TYPE_CATEGORIES:
        return 0.0
    notice_types = [t['job_type'] for t in classify_job_type(notice)]
    if not notice_types:
        return 0.0
    domain  = get_user_domain(track)
    user_jt = DOMAIN_TO_JOB_TYPE.get(cat, {}).get(domain)
    if user_jt is None:
        return 0.0
    return 1.0 if user_jt in notice_types else 0.0

# ============================================================
# Two-Tower 추천
# ============================================================

def two_tower_recommend(college, track, year, interests, top_k=10):
    try:
        sbert, model, device = load_two_tower_model()
        notices              = load_notices_from_supabase()
        emb_dict             = load_embeddings_from_supabase()

        if not notices or not emb_dict:
            return []

        # 필터링 카테고리 제외한 관심사만
        rec_interests = [i for i in interests if i not in FILTER_CATEGORIES]

        user_text   = f"{college} {track} {year} 관심사: {', '.join(rec_interests)}"
        user_emb    = sbert.encode([user_text], convert_to_numpy=True)
        user_tensor = torch.tensor(user_emb, dtype=torch.float).to(device)
        with torch.no_grad():
            user_vec = model.forward_user(user_tensor).cpu().numpy()

        scores      = np.array([n.get('notice_score', 0) for n in notices])
        max_score   = scores.max() if scores.max() > 0 else 1
        scores_norm = scores / max_score

        results = []
        for n_idx, notice in enumerate(notices):
            nid      = notice.get('id')
            item_emb = emb_dict.get(nid)
            if item_emb is None:
                continue

            category = notice.get('category', '')

            # 구버전 + 필터링 카테고리 제외
            if category in FILTER_CATEGORIES or category in OLD_CATEGORIES:
                continue
            # notice_score 0인 공지 제외
            if notice.get('notice_score', 0) <= 0.01:
                continue

            n_score   = float(scores_norm[n_idx])
            sim_score = float(np.dot(item_emb, user_vec.T).flatten()[0])
            sim_norm  = (sim_score + 1.0) / 2.0

            if category in rec_interests:
                cat_score = 1.0
            elif category in PENALTY_CATEGORIES:
                cat_score = -1.0
            else:
                cat_score = 0.0

            job_score   = get_job_score(track, notice)
            final_score = (
                MODEL_WEIGHT    * sim_norm  +
                CATEGORY_WEIGHT * cat_score +
                JOB_TYPE_WEIGHT * job_score +
                SCORE_WEIGHT    * n_score
            )
            results.append({
                'notice':      notice,
                'final_score': final_score,
                'sim_score':   sim_norm,
                'cat_score':   cat_score,
                'job_score':   job_score,
                'n_score':     n_score,
            })

        results.sort(key=lambda x: x['final_score'], reverse=True)

        # 관심 카테고리 2개, 비관심 1개 제한
        interest_cat_count     = {}
        non_interest_cat_count = {}
        filtered = []

        for res in results:
            if res['final_score'] < 0.5:
                break
            cat = res['notice'].get('category', '')
            if cat in rec_interests:
                if interest_cat_count.get(cat, 0) < 2:
                    filtered.append(res)
                    interest_cat_count[cat] = interest_cat_count.get(cat, 0) + 1
            else:
                if non_interest_cat_count.get(cat, 0) < 1:
                    filtered.append(res)
                    non_interest_cat_count[cat] = non_interest_cat_count.get(cat, 0) + 1
            if len(filtered) == top_k:
                break

        # 터미널 출력
        print(f"\n{'='*60}")
        print(f"유저: {college} {track} {year}")
        print(f"관심사: {rec_interests}")
        print(f"{'='*60}")
        for i, res in enumerate(filtered):
            n = res['notice']
            print(f"{i+1:2d}. [{n.get('category')}] {n.get('title','')[:40]}")
            print(f"     최종:{res['final_score']:.3f} 모델:{res['sim_score']:.3f} "
                  f"카테고리:{res['cat_score']:.1f} 직무:{res['job_score']:.1f} "
                  f"공지:{res['n_score']:.3f}")

        return filtered

    except Exception as e:
        print(f"Two-Tower 추천 오류: {e}")
        import traceback; traceback.print_exc()
        return []
