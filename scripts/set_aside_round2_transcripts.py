"""Round 5: when the round-5 model is promoted, keep the app's earlier validation transcripts (made with the round-2 model)
under their own name, so every later comparison transcribes the app as it now stands afresh.

    python scripts/set_aside_round2_transcripts.py
"""
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "data" / "ml" / "results"
SETTING = "auto+background+region"


def main():
    moved = []
    for folder, pattern in (("hyp-robust", f"*__{SETTING}__*.jsonl"), ("hyp-overlap", f"{SETTING}__*.jsonl"), ("hyp", f"*pipeline-{SETTING}-r2__*.jsonl")):
        for path in (RESULTS / folder).glob(pattern):
            target = path.with_name(path.name.replace(SETTING, SETTING + "@round2model"))
            path.rename(target)
            moved.append(target.name)
    print(len(moved), "transcript files set aside:", *moved, sep="\n  ")


if __name__ == "__main__":
    main()
