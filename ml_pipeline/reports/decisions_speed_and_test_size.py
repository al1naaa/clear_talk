import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cleartalk_ml import pipeline_log as log

log.note("Speed and precision on the GTX 1660 Ti (same training batch): float32 loss 1.3716, bfloat16 1.3699 - they "
         "agree, but bfloat16 is slower (1.09 s vs 0.89 s per example) and needs more memory (3.8 GB vs 2.1 GB for one "
         "example). Training runs in float32, micro-batch 2 × accumulation 8 (3.3 GB). A micro-batch of 4 does not fit "
         "6 GB: Windows then spills into shared system memory and the job crawls instead of failing.")
log.note("Epoch size set from the measured speed (0.9 s per example whatever the window holds): 14,400 examples = 900 "
         "optimizer steps, about 3.6 hours. Because a mixed recording carries about 2.5 utterances for the price of "
         "one, an epoch is 9,000 mixed recordings plus 5,400 single utterances (FLEURS 3,600, KSC2 1,000, Golos 800, "
         "seeded samples) instead of the 25,340 examples first planned. Decided before any training step; figures 06 "
         "and 08 were redrawn.")
log.note("Test set made smaller before any use: 300 units per set (FLEURS kk / ru / en, KSC2, Golos, mixed) = 1,800 "
         "units, 6.0 hours of audio, instead of 3,878 units and 12 hours. Reason: every system is scored through the "
         "app's decoding on one laptop GPU, and the full set would take about five hours. The new units are a subset "
         "of the first frozen list (the beginning of the same seeded draws); no test audio had been transcribed "
         "(touch count 0), so the fingerprint in .guard_state.json was replaced. Validation sets are unchanged.")
log.note("The large reference models are scored on the graphics card only when it is free: run on the CPU next to "
         "training they slowed every job (about 10 s per utterance for large-v3-turbo) and were stopped.")
log.set_step(6, "done",
             "official corpus parts; KSC2 official Test divided into validation and test halves by a seeded draw inside every "
             "source; 0 shared utterances and 0 shared FLEURS sentences between roles; 14,400 training examples per epoch, "
             "2,219 validation units, 1,800 test units (300 per set) frozen in .guard_state.json")
print("logged")
