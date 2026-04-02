import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
BOARD_LIST_URL = "https://www.hansung.ac.kr/bbs/hansung/2127/artclList.do"

res = requests.get(BOARD_LIST_URL, params={"page": 1}, headers=HEADERS)
soup = BeautifulSoup(res.text, "html.parser")

# tr 전체 클래스 & 날짜 확인
for tr in soup.find_all("tr")[:15]:
    tds = tr.find_all("td")
    if not tds:
        continue
    print("tr 클래스:", tr.get("class"))
    for td in tds:
        print("  td 클래스:", td.get("class"), "|", td.get_text(strip=True)[:40])
    print()