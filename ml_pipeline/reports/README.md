# Report scripts

Each script wrote one part of `../PIPELINE.md` (a gate report, checklist rows or dated decisions) from the
saved result files, so that no number in the log is typed by hand. They are kept as a record of where the
text came from. A gate script runs once, after that gate's figures and tables exist; running it again
stops with an assertion because the report is already in the log.

- `gate_a.py`, `gate_b.py`, `gate_c.py` - reports of the pipeline stops A, B, C.
- `decisions_speed_and_test_size.py` - measured speed, epoch size, the smaller test set.
