#!/usr/bin/env bash
# macOS launchd installer for the Sangsangfinder iMessage notification worker.
#
# 설치:  bash scripts/setup_notifications.sh install
# 제거:  bash scripts/setup_notifications.sh uninstall
# 상태:  bash scripts/setup_notifications.sh status

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PLIST_NAME="com.sangsangfinder.notifications.plist"
PLIST_SRC="$SCRIPT_DIR/$PLIST_NAME"
AGENTS_DIR="$HOME/Library/LaunchAgents"
PLIST_DST="$AGENTS_DIR/$PLIST_NAME"
LABEL="com.sangsangfinder.notifications"

CMD="${1:-install}"

install() {
    if [ ! -f "$PLIST_SRC" ]; then
        echo "plist 파일을 찾을 수 없습니다: $PLIST_SRC"
        exit 1
    fi

    chmod +x "$SCRIPT_DIR/run_notification_worker.sh"
    mkdir -p "$AGENTS_DIR"
    launchctl unload "$PLIST_DST" 2>/dev/null || true
    cp "$PLIST_SRC" "$PLIST_DST"
    launchctl load "$PLIST_DST"

    echo "알림 워커 설치 완료"
    echo "  상태 확인: launchctl list | grep $LABEL"
    echo "  로그 확인: tail -f $SCRIPT_DIR/notice_notification_worker.log"
    echo "  제거: bash $SCRIPT_DIR/setup_notifications.sh uninstall"
}

uninstall() {
    launchctl unload "$PLIST_DST" 2>/dev/null || true
    rm -f "$PLIST_DST"
    echo "알림 워커 제거 완료"
}

status() {
    echo "=== launchd 상태 ==="
    launchctl list | grep "$LABEL" || echo "등록된 job 없음"
    echo ""
    echo "=== 최근 로그 ==="
    tail -20 "$SCRIPT_DIR/notice_notification_worker.log" 2>/dev/null || echo "로그 파일 없음"
}

case "$CMD" in
    install) install ;;
    uninstall) uninstall ;;
    status) status ;;
    *)
        echo "사용법: $0 {install|uninstall|status}"
        exit 1
        ;;
esac
