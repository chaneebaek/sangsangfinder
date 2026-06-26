#!/usr/bin/env bash
# setup_scheduler.sh — macOS launchd 기반 시간별 크롤러 설치/제거
#
# 설치:  bash crawling/setup_scheduler.sh install
# 제거:  bash crawling/setup_scheduler.sh uninstall
# 상태:  bash crawling/setup_scheduler.sh status

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLIST_NAME="com.sangsangfinder.crawler.plist"
PLIST_SRC="$SCRIPT_DIR/$PLIST_NAME"
AGENTS_DIR="$HOME/Library/LaunchAgents"
PLIST_DST="$AGENTS_DIR/$PLIST_NAME"
LABEL="com.sangsangfinder.crawler"

CMD="${1:-install}"

install() {
    if [ ! -f "$PLIST_SRC" ]; then
        echo "❌ plist 파일을 찾을 수 없습니다: $PLIST_SRC"
        exit 1
    fi

    mkdir -p "$AGENTS_DIR"

    # 기존 job 언로드 (없으면 무시)
    launchctl unload "$PLIST_DST" 2>/dev/null || true

    cp "$PLIST_SRC" "$PLIST_DST"
    launchctl load "$PLIST_DST"

    echo "✓ 스케줄러 설치 완료 (평일 09:17-17:17, 1시간 간격 실행)"
    echo ""
    echo "  상태 확인:  launchctl list | grep sangsangfinder"
    echo "  로그 (실시간): tail -f $SCRIPT_DIR/auto_crawler.log"
    echo "  중지:       bash $SCRIPT_DIR/setup_scheduler.sh uninstall"
}

uninstall() {
    launchctl unload "$PLIST_DST" 2>/dev/null || true
    rm -f "$PLIST_DST"
    echo "✓ 스케줄러 제거 완료"
}

status() {
    echo "=== launchd 상태 ==="
    launchctl list | grep "$LABEL" || echo "  (등록된 job 없음)"
    echo ""
    echo "=== 최근 로그 (20줄) ==="
    tail -20 "$SCRIPT_DIR/auto_crawler.log" 2>/dev/null || echo "  (로그 파일 없음)"
}

case "$CMD" in
    install)   install   ;;
    uninstall) uninstall ;;
    status)    status    ;;
    *)
        echo "사용법: $0 {install|uninstall|status}"
        exit 1
        ;;
esac
