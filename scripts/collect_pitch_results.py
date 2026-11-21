"""Collect measured decisions and aggregate validation/test evidence without exporting examples."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def collect():
    asr = ROOT / "models/pitch-asr"
    sep = ROOT / "models/pitch-separator"
    overlap = asr / "overlap"
    selection = read(asr / "selection.json")
    choice = read(overlap / "selection.json")
    external = read(asr / "external-noise.json")
    final = read(overlap / f"test-{choice['selected']}.json")
    baseline = (
        read(overlap / "test-baseline.json")
        if choice["selected"] != "libri" or selection["selected_step"]
        else None
    )
    assert (
        final["all"]["n"] == 36
        and external["selected_step"] == selection["selected_step"]
    )
    assert all(
        r["n"] == 18 for system in external["systems"].values() for r in system.values()
    )
    return dict(
        train_config=read(asr / "config.json"),
        reference_audit=read(asr / "reference-audit.json")
        if (asr / "reference-audit.json").exists()
        else None,
        asr_adapter_sha256={
            step: digest(asr / f"step-{step:03d}/adapter_model.safetensors")
            for step in (24, 48, 96)
        },
        asr_validation=selection,
        enhancement_validation=read(asr / "stationary-noise/selection.json")
        if (asr / "stationary-noise/selection.json").exists()
        else None,
        noisy_controls=external,
        acoustic_validation=read(sep / "selection.json"),
        acoustic_checkpoint_sha256=digest(sep / "selected.ckpt"),
        separation_selection=choice,
        separation_test=final,
        isolated_sources=read(overlap / "isolated-test.json")
        if (overlap / "isolated-test.json").exists()
        else None,
        separation_test_baseline=baseline,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/pitch/current.json")
    output = parser.parse_args().output
    result = collect()
    output.parent.mkdir(exist_ok=True, parents=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
