#!/usr/bin/env python3
"""
auto_crawler.py — 한성대 공지 증분 크롤러 + Pinecone 자동 인덱서

실행:
  python crawling/auto_crawler.py              # 1회 실행 (cron/launchd 용)
  python crawling/auto_crawler.py --daemon     # 1시간 간격 무한 반복
  python crawling/auto_crawler.py --no-index   # 크롤링만 (Pinecone 인덱싱 생략)
  python crawling/auto_crawler.py --max-pages 10  # 최대 탐색 페이지 수 조정

출력:
  api.core.config.NOTICES_CACHE_PATH  (기존 파일 있으면 신규 건만 추가)
  crawling/auto_crawler.log
"""

import argparse
import json
import logging
import signal
import sys
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

# ── 경로 설정 ─────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent   # crawling/
_ROOT = _HERE.parent                      # 프로젝트 루트
sys.path.insert(0, str(_ROOT))

# ── 로깅 (콘솔 + 순환 파일) ───────────────────────────────────
_LOG_PATH = _HERE / "auto_crawler.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler(_LOG_PATH, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ── 상수 ─────────────────────────────────────────────────────
HEADERS  = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
BASE_URL = "https://www.hansung.ac.kr"
LIST_URL = f"{BASE_URL}/bbs/hansung/2127/artclList.do"

DELAY      = 0.5   # 요청 간 딜레이(초) — 서버 부하 방지
SAVE_EVERY = 20    # N건마다 중간 저장
KST = ZoneInfo("Asia/Seoul")

# ── 카테고리 분류 (api/core/config.py 와 동기화) ──────────────
from api.core.config import NOTICES_CACHE_PATH  # noqa: E402
from api.core.utils import infer_category  # noqa: E402
from crawling.crawler import get_post_content  # noqa: E402


# ── 파일 I/O ─────────────────────────────────────────────────

def _out_path() -> Path:
    """공지 캐시 출력 경로. 앱/인덱서가 읽는 설정과 동일해야 한다."""
    return Path(NOTICES_CACHE_PATH)


def _load(path: Path) -> tuple[list[dict], set[str]]:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return data, {item["url"] for item in data}
    return [], set()


def _save(data: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ── 크롤링 ────────────────────────────────────────────────────

def _parse_views(td) -> int:
    """게시판 목록 td에서 조회수 파싱. 실패 시 0."""
    if not td:
        return 0
    try:
        return int(td.get_text(strip=True).replace(",", ""))
    except ValueError:
        return 0


def fetch_list_page(page: int) -> list[dict]:
    """목록 한 페이지 → [{title, url, date, views}]. 실패 시 []."""
    try:
        res = requests.get(LIST_URL, params={"page": page}, headers=HEADERS, timeout=10)
        res.raise_for_status()
    except Exception as e:
        logger.warning("페이지 %d 요청 실패: %s", page, e)
        return []

    soup  = BeautifulSoup(res.text, "html.parser")
    rows  = soup.select("table.board-table tbody tr")
    items = []

    for row in rows:
        if "notice" in row.get("class", []):  # 고정 공지 건너뜀
            continue

        title_td = row.select_one("td.td-title a")
        date_td  = row.select_one("td.td-date")
        if not title_td or not date_td:
            continue

        # 조회수 셀 — CMS 버전별 클래스명 대응 (실제: td-counts)
        view_td = (
            row.select_one("td.td-counts")
            or row.select_one("td.td-view")
            or row.select_one("td.td-count")
            or row.select_one("td.td-hit")
        )

        href = title_td.get("href", "")
        url  = BASE_URL + href if href.startswith("/") else href

        items.append({
            "title": title_td.get_text(strip=True),
            "url":   url,
            "date":  date_td.get_text(strip=True),
            "views": _parse_views(view_td),
        })

    return items


def fetch_body(url: str) -> str:
    """공지 본문 추출: 텍스트 + 이미지 OCR + PDF."""
    try:
        return get_post_content(url)
    except Exception as e:
        logger.warning("본문 크롤링 실패 (%s): %s", url, e)
    return ""


# ── 메인 크롤링 로직 ──────────────────────────────────────────

def run_once(
    no_index: bool = False,
    max_pages: int = 5,
    storage: str = "file",
    init_db: bool = False,
) -> int:
    """증분 크롤링 1회 실행. 신규 수집 건수 반환."""
    year = str(datetime.now(KST).year)
    path = _out_path()

    if storage == "supabase":
        from crawling.supabase_store import ensure_schema, load_seen_urls, upsert_notices

        if init_db:
            ensure_schema()
        results: list[dict] = []
        seen = load_seen_urls(year)
    else:
        results, seen = _load(path)

    logger.info(
        "=== 크롤링 시작 [%s] — 기존 URL %d건, 대상 연도: %s, 저장소: %s ===",
        datetime.now().strftime("%H:%M:%S"),
        len(seen),
        year,
        storage,
    )

    new_items: list[dict] = []
    stop = False

    for page in range(1, max_pages + 1):
        if stop:
            break

        logger.info("페이지 %d 요청...", page)
        items = fetch_list_page(page)
        if not items:
            logger.info("페이지 %d: 응답 없음 — 탐색 종료", page)
            break

        page_new       = 0
        all_known      = True   # 이 페이지 전체가 기존 URL이면 조기 종료

        for item in items:
            item_year = item["date"][:4]

            if item_year > year:
                continue                    # 더 최신 연도 (가능성 낮음)
            if item_year < year:
                logger.info("%s년 데이터 도달 — 수집 종료", item_year)
                stop = True
                break
            if item["url"] in seen:
                continue                    # 이미 보유한 공지

            all_known = False

            body             = fetch_body(item["url"])
            item["body"]     = body
            item["category"] = infer_category(item["title"], body)

            new_items.append(item)
            seen.add(item["url"])
            page_new += 1

            logger.info(
                "[신규] %s  [%-12s]  %s  (%d자)",
                item["date"], item["category"], item["title"][:40], len(body),
            )

            if storage == "file" and len(new_items) % SAVE_EVERY == 0:
                _save(new_items + results, path)
                logger.info("중간 저장: 신규 %d건 누적", len(new_items))

            time.sleep(DELAY)

        logger.info("페이지 %d 완료: 신규 %d건", page, page_new)

        # 페이지 전체가 이미 알고 있는 URL → 더 뒤 페이지도 마찬가지
        if all_known and not stop:
            logger.info("페이지 %d 전체 기존 데이터 — 탐색 조기 종료", page)
            break

        time.sleep(DELAY)

    if new_items:
        if storage == "supabase":
            from crawling.notice_processor import process_notices
            saved = process_notices(new_items)
            logger.info("=== 완료: 신규 %d건 처리 완료 ===", saved)
            index_items = new_items
        else:
            final = new_items + results
            _save(final, path)
            logger.info("=== 완료: 신규 %d건 추가 ===", len(new_items))
            index_items = final

        if not no_index:
            _index(index_items)
    else:
        logger.info("=== 완료: 신규 공지 없음 ===")

    return len(new_items)


def _index(items: list[dict]) -> None:
    """신규 공지를 Pinecone에 인덱싱."""
    try:
        from api.core.models import index_notices  # noqa: PLC0415
        n = index_notices(items)
        logger.info("Pinecone 인덱싱 완료: %d건", n)
    except Exception:
        logger.exception("Pinecone 인덱싱 실패")


# ── 데몬 모드 ─────────────────────────────────────────────────

_alive = True


def _on_signal(sig: int, _frame) -> None:
    global _alive
    logger.info("종료 신호(%d) 수신 — 현재 사이클 완료 후 종료", sig)
    _alive = False


def run_daemon(interval: int, no_index: bool, max_pages: int, storage: str, init_db: bool) -> None:
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    logger.info("데몬 시작 — 실행 간격: %d초 (%d분)", interval, interval // 60)

    while _alive:
        try:
            run_once(no_index=no_index, max_pages=max_pages, storage=storage, init_db=init_db)
        except Exception:
            logger.exception("크롤링 예외 발생 — 다음 사이클에 재시도")

        if not _alive:
            break

        wake = datetime.fromtimestamp(time.time() + interval).strftime("%Y-%m-%d %H:%M:%S")
        logger.info("다음 실행 예정: %s", wake)
        time.sleep(interval)

    logger.info("데몬 정상 종료")


# ── CLI ───────────────────────────────────────────────────────

def _is_business_hours() -> bool:
    """평일 09:00-18:00 KST 여부 확인."""
    now = datetime.now(KST)
    return now.weekday() < 5 and 9 <= now.hour < 18


def main() -> None:
    ap = argparse.ArgumentParser(
        description="한성대 공지 증분 크롤러",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--daemon",     action="store_true",  help="데몬 모드 (반복 실행)")
    ap.add_argument("--interval",   type=int, default=3600, help="데몬 실행 간격(초)")
    ap.add_argument("--no-index",   action="store_true",  dest="no_index", help="Pinecone 인덱싱 생략")
    ap.add_argument("--max-pages",  type=int, default=5,  dest="max_pages", help="탐색 최대 페이지 수")
    ap.add_argument("--storage", choices=["file", "supabase"], default="file", help="공지 저장소")
    ap.add_argument("--init-db", action="store_true", help="Supabase notices 테이블/인덱스 생성")
    ap.add_argument("--force",      action="store_true",  help="업무시간 체크 무시하고 강제 실행")
    args = ap.parse_args()

    if args.daemon:
        run_daemon(args.interval, args.no_index, args.max_pages, args.storage, args.init_db)
    else:
        if not args.force and not _is_business_hours():
            logger.info("업무시간 외 (%s KST) — 크롤링 생략", datetime.now(KST).strftime("%a %H:%M"))
            return
        run_once(
            no_index=args.no_index,
            max_pages=args.max_pages,
            storage=args.storage,
            init_db=args.init_db,
        )


if __name__ == "__main__":
    main()
