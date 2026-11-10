import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cleartalk_ml import pipeline_log as log

log.note("App behaviour, decided on validation transcripts only (no transcript changes, so no WER changes): for a turn "
         "written by our recognizer the language LABEL is read from the letters of the text it wrote, not from its "
         "language token. Measured on the cached validation transcripts of our model: Kazakh named correctly for 74.0% "
         "→ 100.0% of FLEURS and 69.2% → 97.2% of KSC2 utterances, Russian 100.0% → 100.0% (FLEURS) and 98.5% → 99.8% "
         "(Golos), English 100.0% → 99.7%. The same rule applied to the ready models' transcripts changes little or "
         "is slightly worse (large-v3-turbo on Golos 100.0% → 98.2%), so it is used for our recognizer only.")
log.note("Promotion rule 1 was given a concrete effect in the app: `small_recognizer` in "
         "`models/multilingual/profile.json` names the recognizer that stands in when a routed large model or the "
         "detector is not installed (a computer that carries only a small model). Promoted → our model; otherwise "
         "untouched Whisper small.")
print("logged")
