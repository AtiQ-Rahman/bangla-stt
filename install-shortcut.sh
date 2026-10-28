#!/usr/bin/env bash
# Add "Bangla STT" to the applications menu and the desktop. It starts the app without a
# terminal (start.sh; ⏻ Quit on the page stops it). Right-click it for the log.
set -e
LOG="${XDG_STATE_HOME:-$HOME/.local/state}/bangla-stt/app.log"
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
FILE="$HOME/.local/share/applications/bangla-stt-app.desktop"
# The old shortcut ran a terminal; under its old name GNOME kept using the cached command.
rm -f "$HOME/.local/share/applications/bangla-stt.desktop" "$HOME/Desktop/bangla-stt-app.desktop"
mkdir -p "$(dirname "$FILE")"
cat > "$FILE" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Bangla STT
Comment=Bangla & English speech-to-text (runs on this PC)
Exec="$APP_DIR/start.sh"
Icon=audio-input-microphone
Terminal=false
Categories=AudioVideo;Audio;
Actions=log;

[Desktop Action log]
Name=Show the log
Exec=xdg-open "$LOG"
DESKTOP
chmod +x "$FILE"
if [ -d "$HOME/Desktop" ]; then
  cp "$FILE" "$HOME/Desktop/bangla-stt-app.desktop"
  chmod +x "$HOME/Desktop/bangla-stt-app.desktop"
  gio set "$HOME/Desktop/bangla-stt-app.desktop" metadata::trusted true 2>/dev/null || true
fi
echo "Added Bangla STT to the applications menu and the desktop."
