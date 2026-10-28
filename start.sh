#!/usr/bin/env bash
# Start Bangla STT without a terminal window (the app menu / desktop shortcut runs this).
# Phone mode is on (phones on the same Wi-Fi can use it; the address and password are at the
# top of the page). Its output goes to a log file; if it is already running, just open it.
cd "$(dirname "$0")"
LOG_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/bangla-stt"
LOG="$LOG_DIR/app.log"
for URL in https://127.0.0.1:7860 http://127.0.0.1:7860; do  # with or without phone mode
  if curl -sk -o /dev/null --max-time 2 "$URL"; then
    exec xdg-open "$URL" >/dev/null 2>&1
  fi
done
mkdir -p "$LOG_DIR"
[ -f "$LOG" ] && mv "$LOG" "$LOG.1"  # the previous run's log, e.g. of a crash
(
  PYTHONUNBUFFERED=1 ./run.sh --phone "$@" >"$LOG" 2>&1
  code=$?
  # 0: stopped with ⏻ Quit or a plain kill. Anything else is a crash: say where the log is.
  if [ "$code" -ne 0 ]; then
    notify-send -i dialog-error "Bangla STT stopped unexpectedly" "The log is in $LOG" 2>/dev/null
  fi
) </dev/null >/dev/null 2>&1 &
disown
