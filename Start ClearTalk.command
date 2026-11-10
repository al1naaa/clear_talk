#!/bin/zsh
set -e
cd "$(dirname "$0")"
if [[ ! -x ".venv/bin/python" ]]; then
  echo "ClearTalk environment is missing. See README.md for installation."
  read -k 1 "?Press any key to close."
  exit 1
fi
exec .venv/bin/python scripts/launch.py
