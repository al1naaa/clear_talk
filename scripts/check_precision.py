"""Is reduced precision safe and worth it on this graphics card? Same batch, three precisions.

    python scripts/check_precision.py

For float32, bfloat16 and float16 it reports the loss of one fixed training batch (they must agree
with float32), the time of a forward + backward pass, and the peak GPU memory, for micro-batches
of 1 and 2. The earlier English run found half precision WRONG on Apple MPS; the GTX 16-series
has no fast half-precision path, so nothing is assumed here - it is measured, and the result is
written to data/ml/results/precision-check.json for the pipeline log.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from peft import LoraConfig, get_peft_model
    from transformers import WhisperForConditionalGeneration

    from cleartalk_ml.dataset import ExampleDataset, collate
    from cleartalk_ml.paths import BASE_WHISPER, RESULTS
    from cleartalk_ml.splits import training_examples
    from cleartalk_ml.train import LORA_TARGETS

    if not torch.cuda.is_available():
        raise SystemExit("This check is for a CUDA graphics card")
    examples = sorted(training_examples(200), key=lambda e: e["id"])
    picks = [examples[i] for i in np.random.default_rng(0).permutation(len(examples))[:4]]
    data = ExampleDataset(picks)
    items = [data[i] for i in range(4)]
    model = WhisperForConditionalGeneration.from_pretrained(BASE_WHISPER, attn_implementation="sdpa")
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=LORA_TARGETS, bias="none")).to("cuda")
    params = [p for p in model.parameters() if p.requires_grad]
    report = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "runs": []}
    for precision, dtype in (("fp32", None), ("bf16", torch.bfloat16), ("fp16", torch.float16)):
        for size in (1, 2):  # larger batches spill into shared system memory on a 6 GB card and crawl
            batch = {k: v.to("cuda") for k, v in collate(items[:size]).items()}
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                model.eval()  # no dropout: the loss must be comparable between precisions
                with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                    loss = float(model(**batch).loss)
                model.train()
                times = []
                for _ in range(4):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                        out = model(**batch).loss
                    out.backward()
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - started)
                    for p in params:
                        p.grad = None
                row = {"precision": precision, "micro_batch": size, "loss": round(loss, 4),
                       "seconds_per_example": round(min(times[1:]) / size, 3),
                       "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
            except RuntimeError as error:  # out of memory or an unsupported operation
                row = {"precision": precision, "micro_batch": size, "error": str(error).splitlines()[0][:160]}
                torch.cuda.empty_cache()
            report["runs"].append(row)
            print(row, flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "precision-check.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
