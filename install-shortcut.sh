#!/usr/bin/env bash
# Add "Bangla STT" to the applications menu and the desktop. It opens a terminal running
# run.sh; closing that terminal window stops the app.
set -e
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
FILE="$HOME/.local/share/applications/bangla-stt.desktop"
mkdir -p "$(dirname "$FILE")"
cat > "$FILE" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Bangla STT
Comment=Bangla & English speech-to-text (runs on this PC)
Exec=gnome-terminal --title="Bangla STT — close this window to stop the app" -- "$APP_DIR/run.sh"
Icon=audio-input-microphone
Terminal=false
Categories=AudioVideo;Audio;
DESKTOP
chmod +x "$FILE"
if [ -d "$HOME/Desktop" ]; then
  cp "$FILE" "$HOME/Desktop/bangla-stt.desktop"
  chmod +x "$HOME/Desktop/bangla-stt.desktop"
  gio set "$HOME/Desktop/bangla-stt.desktop" metadata::trusted true 2>/dev/null || true
fi
echo "Added Bangla STT to the applications menu and the desktop."
