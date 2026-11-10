"""Step 14: the one-time final test.

    python -m cleartalk_asr.final_test

Requires Gate C in ml_pipeline_english/PIPELINE.md. Both recognizers (pretrained and the validation-
selected ClearTalk model) transcribe the same frozen test audio through the app's decoding
path. ``guard.final_test`` verifies the frozen fingerprint, refuses a second run and logs the
ClearTalk score; the pretrained transcripts are saved next to it for the paired comparison.
"""

import argparse
import json
import sys

from . import splits
from .evaluate import RESULTS, transcribe_app_parallel
from .paths import BASE_WHISPER_CT2, EXPORT, PIPELINE


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    sys.path.insert(0, str(PIPELINE))
    import guard

    out = RESULTS / "test-final.json"
    if out.exists():
        raise SystemExit(f"The final test was already run once; see {out}")
    units = splits.eval_frame(splits.TEST_SETS)
    saved = {}

    def transcribe_both(features):
        paths = [path for joined in features.paths for path in joined.split(splits.SEP)]
        for name, model_dir in (("baseline", BASE_WHISPER_CT2), ("cleartalk", EXPORT)):
            texts = iter(transcribe_app_parallel(model_dir, paths, args.workers, args.threads))
            saved[name] = [splits.SEP.join(next(texts) for _ in joined.split(splits.SEP)) for joined in features.paths]
            (RESULTS / f"test-hypotheses-{name}.json").write_text(json.dumps(saved[name], indent=1) + "\n")
            print(f"{name}: {len(paths)} tracks transcribed", flush=True)
        return saved["cleartalk"]

    score = guard.final_test(transcribe_both, units, target="reference", metric_fn=splits.unit_wer,
                             state_dir=PIPELINE, pipeline_md=PIPELINE / "PIPELINE.md")
    out.write_text(json.dumps({
        "cleartalk_wer_all_units": score, "unit_id": list(units.unit_id), "set": list(units.set),
        "kind": list(units.kind), "reference": list(units.reference),
        "hyp_baseline": saved["baseline"], "hyp_cleartalk": saved["cleartalk"]}, indent=1) + "\n")
    print(f"final test done: ClearTalk pooled WER over all {len(units)} units = {100 * score:.2f}%", flush=True)


if __name__ == "__main__":
    main()
