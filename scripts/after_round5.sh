#!/usr/bin/env bash
# Round 5, after training and selection (scripts/night_round5.sh): the two app options with the model the rule left in
# place, the Gate C report, then the one-time final test on test E. Safe to start again: every check resumes.
#
#     bash scripts/after_round5.sh
#
# Progress: logs/after_round5.log, logs/option5_text.log, logs/option5_overlap.log, logs/final_test_e.log
cd "$(dirname "$0")/.." || exit 1
export PYTHONUTF8=1
PY=./.venv-train/Scripts/python.exe
LOG=logs/after_round5.log
note() { echo "$(date '+%F %T') $*" >> "$LOG"; }

until grep -q "selection ended" logs/night_round5.log 2>/dev/null; do sleep 60; done
if ! grep -q "selection ended with exit code 0" logs/night_round5.log; then note "selection failed; stopping"; exit 1; fi
promoted=$("$PY" -X utf8 -c "import json; print(json.load(open('data/ml/results/selection-round5.json', encoding='utf-8')).get('promoted'))")
note "round-5 model promoted: $promoted"
if [ "$promoted" = "True" ] && [ ! -f logs/.round2_transcripts_set_aside ]; then
  "$PY" -X utf8 scripts/set_aside_round2_transcripts.py >> "$LOG" 2>&1 && touch logs/.round2_transcripts_set_aside
fi
for option in text overlap; do
  if [ ! -f "data/ml/results/option-$option.json" ]; then
    "$PY" -X utf8 scripts/check_option.py "$option" > "logs/option5_$option.log" 2>&1
    grep -q "OPTION_DONE $option" "logs/option5_$option.log" || { note "check $option failed; stopping"; exit 1; }
  fi
  note "check $option done"
done
"$PY" -X utf8 ml_pipeline/round5/reports/r5_gate_c.py >> "$LOG" 2>&1
"$PY" -X utf8 -m cleartalk_ml.final_test_e > logs/final_test_e.log 2>&1
note "final test E ended with exit code $?"
