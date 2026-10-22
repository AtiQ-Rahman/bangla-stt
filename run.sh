#!/usr/bin/env bash
# Set up the virtualenv (again whenever requirements.txt changes), then start the web UI.
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
fi
if [ ! -f .venv/.installed ] || [ requirements.txt -nt .venv/.installed ]; then
  .venv/bin/pip install -r requirements.txt
  touch .venv/.installed
fi
exec .venv/bin/python app.py "$@"
