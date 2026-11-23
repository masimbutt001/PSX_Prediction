#!/usr/bin/env bash
# Sets up the PSX daily run on this Mac (alternative to the GitHub cloud run).
#
# Installs a launchd agent that checks every hour and at log-in. Each check runs
# `psx schedule run-daily --catch-up`, which only does work once per PSX trading day after
# market close, so a run missed while the Mac was off or asleep happens at the next check.
#
# Install:   bash scripts/install_laptop_schedule_macos.sh
# Remove:    bash scripts/install_laptop_schedule_macos.sh --uninstall
set -euo pipefail

LABEL="com.psx-prediction.daily-pipeline"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

if [[ "${1:-}" == "--uninstall" ]]; then
  launchctl bootout "gui/$(id -u)" "$PLIST" 2>/dev/null || true
  rm -f "$PLIST"
  echo "Removed $LABEL."
  exit 0
fi

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
UV="$(command -v uv || true)"
if [[ -z "$UV" ]]; then
  echo "uv not found. Install it first: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi
mkdir -p "$REPO_DIR/data" "$HOME/Library/LaunchAgents"

cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>WorkingDirectory</key><string>$REPO_DIR</string>
  <key>ProgramArguments</key>
  <array>
    <string>$UV</string><string>run</string><string>psx</string>
    <string>schedule</string><string>run-daily</string><string>--catch-up</string>
  </array>
  <key>StartInterval</key><integer>3600</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>$REPO_DIR/data/scheduler.out.log</string>
  <key>StandardErrorPath</key><string>$REPO_DIR/data/scheduler.out.log</string>
</dict>
</plist>
PLIST

launchctl bootout "gui/$(id -u)" "$PLIST" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "Installed $LABEL. It checks every hour and runs once per PSX trading day after close."
echo "Output goes to $REPO_DIR/data/scheduler.out.log"
