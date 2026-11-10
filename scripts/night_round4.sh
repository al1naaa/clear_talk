#!/usr/bin/env bash
# Round 4, unattended: train the large model further in noise (continued if it stops), then choose the
# checkpoint on validation (scripts/select_round4.py). Safe to start again at any time: every stage resumes.
#
#     bash scripts/night_round4.sh
#
# Progress: logs/night_round4.log, logs/training_noise.log, logs/select_round4.log
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1
PY=./.venv-train/Scripts/python.exe
LOG=logs/night_round4.log
note() { echo "$(date '+%F %T') $*" >> "$LOG"; }

training_runs() {
  powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'run_training' }).Count" | tr -d '\r\n '
}
training_finished() {
  "$PY" -X utf8 -c "import json, pathlib, sys; runs = pathlib.Path('models/multilingual/runs'); choice = runs / 'large-noise-lr-choice.json'; sys.exit(0 if choice.is_file() and (runs / json.loads(choice.read_text(encoding='utf-8'))['chosen'] / 'finished.json').is_file() else 1)"
}

restarts=0
until training_finished; do
  if [ "$(training_runs)" = "0" ]; then
    restarts=$((restarts + 1))
    if [ "$restarts" -gt 6 ]; then note "training stopped $restarts times; giving up"; exit 1; fi
    note "training is not running; continuing it (attempt $restarts)"
    "$PY" -X utf8 scripts/run_training.py --base kazakh-turbo --noise --rate 1e-4 >> logs/training_noise.log 2>&1
    sleep 30
  else
    sleep 120
  fi
done
note "training finished; choosing the checkpoint on validation"
"$PY" -X utf8 scripts/select_round4.py > logs/select_round4.log 2>&1
code=$?
note "selection ended with exit code $code"
exit $code
