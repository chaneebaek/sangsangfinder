"""
2025년 공지 기반 test QA 생성 스크립트
- 기존 qa_dataset_all.jsonl(2026 기반 train)과 완전히 분리된 clean test set 생성
- 카테고리 비율 유지하며 100개 공지 샘플링
- 출력: data/qa_test_2025.jsonl, data/test_notices_2025.json
"""

import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

# 현재 파일 기준으로 working directory 설정 (상대경로 ./data/ 보장)
os.chdir(Path(__file__).parent)

from run_qa_pipeline import run_pipeline  # noqa: E402 (chdir 이후 import)

NOTICES_PATH   = "./data/2025_notice.json"
SELECTED_PATH  = "./data/test_notices_2025.json"
OUTPUT_PATH    = "./data/qa_test_2025.jsonl"
TARGET_N       = 100
MIN_BODY_LEN   = 200
RANDOM_SEED    = 42


def stratified_sample(notices: list[dict], n: int, seed: int) -> list[dict]:
    """카테고리 비율을 유지하며 n개 샘플링"""
    by_cat: dict[str, list] = defaultdict(list)
    for notice in notices:
        by_cat[notice["category"]].append(notice)

    total = len(notices)
    rng = random.Random(seed)
    selected: list[dict] = []

    # 비례 할당 (Largest Remainder Method)
    raw = {cat: len(items) / total * n for cat, items in by_cat.items()}
    floors = {cat: math.floor(v) for cat, v in raw.items()}
    remainder = n - sum(floors.values())
    remainders = sorted(raw.keys(), key=lambda c: -(raw[c] - floors[c]))
    for cat in remainders[:remainder]:
        floors[cat] += 1

    for cat, items in by_cat.items():
        k = min(floors[cat], len(items))
        selected.extend(rng.sample(items, k))

    rng.shuffle(selected)
    return selected


def main():
    notices_all = json.load(open(NOTICES_PATH, encoding="utf-8"))
    valid = [n for n in notices_all if len(n.get("body", "")) >= MIN_BODY_LEN]
    print(f"2025 공지 전체: {len(notices_all)}, body≥{MIN_BODY_LEN}: {len(valid)}")

    sampled = stratified_sample(valid, TARGET_N, RANDOM_SEED)
    print(f"샘플링: {len(sampled)}개")

    # 선택된 공지 저장 (notice_id ↔ 공지 매핑 보존)
    json.dump(sampled, open(SELECTED_PATH, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"선택 공지 저장 → {SELECTED_PATH}")

    run_pipeline(sampled, output_path=OUTPUT_PATH)


if __name__ == "__main__":
    main()
