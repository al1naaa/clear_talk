"""Flag exact neighboring KSD transcript matches without changing references or scores."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cleartalk_ml.text import normalize
from scripts.pitch_control_inputs import METADATA, verify


def matches(inputs, metadata):
    flagged = []
    for uid, hyp in inputs:
        words = normalize(hyp)
        if len(words.split()) < 3 or words == metadata[uid]:
            continue
        member = Path(uid)
        prefix, index = member.stem.rsplit("_", 1)
        for offset in (-1, 1):
            other = str(
                member.with_name(f"{prefix}_{int(index) + offset}{member.suffix}")
            )
            if words == metadata.get(other):
                flagged.append(uid)
                break
    return flagged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-cache", type=Path)
    args = parser.parse_args()
    verify(METADATA)
    metadata = {
        r["wav"].removeprefix("dataset/"): normalize(r["text"])
        for r in json.loads(METADATA.read_text(encoding="utf-8-sig"))
    }
    work = ROOT / "models/pitch-asr"
    step = json.loads((work / "selection.json").read_text())["selected_step"]
    rows = list(
        map(json.loads, (work / f"holdout/{step}.jsonl").read_text().splitlines())
    )
    clean = [
        (r["uid"], r["hypothesis"])
        for r in rows
        if r["language"] == "kk" and r["condition"] == "clean"
    ]
    assert len(clean) == 18
    result = dict(
        final_clean_checked=len(clean),
        final_exact_neighbor_matches=len(matches(clean, metadata)),
    )
    if args.pilot_cache:
        rows = list(map(json.loads, args.pilot_cache.read_text().splitlines()))
        clean = [
            (r["key"][2], r["hyp"])
            for r in rows
            if r["key"][:2] == ["controls", "bypass"]
            and r["key"][2].startswith("audio2/")
            and r["key"][3] == "clean"
        ]
        assert len(clean) == 6
        flagged = set(matches(clean, metadata))
        overlap = [r for r in rows if r["key"][0] == "overlap"]
        retained = [r for r in overlap if r["key"][1] not in flagged]
        result.update(
            pilot_clean_checked=len(clean),
            pilot_exact_neighbor_matches=len(flagged),
            pilot_all_errors=sum(r["errors"] for r in overlap),
            pilot_all_words=sum(r["words"] for r in overlap),
            sensitivity_errors=sum(r["errors"] for r in retained),
            sensitivity_words=sum(r["words"] for r in retained),
        )
    (work / "reference-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
