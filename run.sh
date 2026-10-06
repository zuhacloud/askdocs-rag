#!/usr/bin/env bash
# Starts AskDocs (API + chat UI) and opens it in your browser. Usage: bash run.sh
# Run bash setup.sh once first.
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "[askdocs] Run bash setup.sh first."
  exit 1
fi
exec .venv/bin/python run.py
