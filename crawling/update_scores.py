#!/usr/bin/env python3
"""
update_scores.py — 최근 45일 이내 공지 조회수 + notice_score 일 1회 업데이트

실행:
  python crawling/update_scores.py
"""

import re
import sys
import time
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv()

import requests
from bs4 import BeautifulSoup
from supabase import create_client
import os

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

HEADERS  = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
BASE_URL = "https://www.hansung.ac.kr"
LIST_URL = f"{BASE_URL}/bbs/hansung/2127/artclList.do"
DAYS_LIMIT = 45
DELAY = 0.3

from crawling.notice_processor import calc_notice_score


def fetch_views_from_list(max_pages: int = 10) -> dict[str, int]:
    """목록 페이지에서 url → views 매핑 수집 (45일 이내만)"""
    today = datetime.now()
    url_views = {}

    for page in range(1, max_pages + 1):
        try:
            res = requests.get(LIST_URL, params={"page": page}, headers=HEADERS, timeout=10)
            res.raise_for_status()
        except Exception as e:
            print(f"페이지 {page} 요청 실패: {e}")
            break

        soup = BeautifulSoup(res.text, "html.parser")
        rows = soup.select("table.board-table tbody tr")
        page_has_target = False

        for row in rows:
            if "notice" in row.get("class", []):
                continue
            title_td = row.select_one("td.td-title a")
            date_td  = row.select_one("td.td-date")
            if not title_td or not date_td:
                continue

            date_str = date_td.get_text(strip=True)
            try:
                posted = datetime.strptime(date_str[:10].replace(".", "-"), "%Y-%m-%d")
                days_old = (today - posted).days
            except:
                continue

            if days_old > DAYS_LIMIT:
                print(f"45일 초과 도달 — 페이지 {page} 탐색 종료")
                return url_views

            view_td = (
                row.select_one("td.td-counts") or
                row.select_one("td.td-view") or
                row.select_one("td.td-count") or
                row.select_one("td.td-hit")
            )
            try:
                views = int(view_td.get_text(strip=True).replace(",", "")) if view_td else 0
            except:
                views = 0

            href = title_td.get("href", "")
            url  = BASE_URL + href if href.startswith("/") else href
            url_views[url] = views
            page_has_target = True

        if not page_has_target:
            break

        time.sleep(DELAY)

    return url_views


def update_scores(url_views: dict[str, int]) -> int:
    """Supabase notices 테이블 조회수 + notice_score 업데이트"""
    if not url_views:
        print("업데이트할 항목 없음")
        return 0

    # 해당 URL 공지 조회
    urls = list(url_views.keys())
    res = supabase.table("notices").select(
        "id, notice_id, url, posted_at, category, views"
    ).in_("url", urls).execute()

    updated = 0
    for notice in (res.data or []):
        url      = notice.get("url")
        new_views = url_views.get(url)
        if new_views is None:
            continue

        new_score = calc_notice_score({
            "views":     new_views,
            "posted_at": notice.get("posted_at", ""),
            "category":  notice.get("category", "기타"),
        })

        supabase.table("notices").update({
            "views":        new_views,
            "notice_score": new_score,
        }).eq("id", notice["id"]).execute()

        print(f"[업데이트] {notice.get('notice_id')} views:{new_views} score:{new_score:.4f}")
        updated += 1

    return updated


def main():
    print(f"=== 조회수/notice_score 업데이트 시작 ({datetime.now().strftime('%Y-%m-%d %H:%M')}) ===")
    url_views = fetch_views_from_list(max_pages=15)
    print(f"45일 이내 공지 {len(url_views)}건 수집")
    updated = update_scores(url_views)
    print(f"=== 완료: {updated}건 업데이트 ===")


if __name__ == "__main__":
    main()
