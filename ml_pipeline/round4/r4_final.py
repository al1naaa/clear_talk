"""Round 4, step 14: the figure of the one-time final test on test D (the app before and after this round's protections).

    python ml_pipeline/round4/r4_final.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cleartalk_ml import pipeline_log as log  # noqa: E402
from cleartalk_ml import round4  # noqa: E402
from cleartalk_ml.paths import RESULTS  # noqa: E402

log.use(round4.R4)
result = json.loads((RESULTS / "test-final-D.json").read_text(encoding="utf-8"))
conditions = [("clean", "clean"), ("white-10", "white noise 10 dB"), ("babble-10", "babble 10 dB")]
sets = [("fleurs-kk", "Kazakh read"), ("ksc2-kk", "Kazakh KSC2"), ("fleurs-ru", "Russian read"), ("golos-ru", "Russian phrases"), ("fleurs-en", "English read")]

fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1, 1.7]})
x = np.arange(len(conditions))
before = [100 * result["condition"][c]["before"] for c, _ in conditions]
after = [100 * result["condition"][c]["final"] for c, _ in conditions]
axes[0].bar(x - 0.2, before, 0.4, color="#999999", label="app as round 4 began")
axes[0].bar(x + 0.2, after, 0.4, color="#de693c", label="app with the protections")
for i, (b, a) in enumerate(zip(before, after)):
    axes[0].text(i - 0.2, b + 0.5, f"{b:.1f}", ha="center", fontsize=8)
    axes[0].text(i + 0.2, a + 0.5, f"{a:.1f}", ha="center", fontsize=8)
axes[0].set_xticks(x, [name for _, name in conditions])
axes[0].set_ylabel("pooled WER %, test D")
axes[0].set_title("Test D, language detected automatically")
axes[0].legend(fontsize=8)
width = 0.13
for j, (condition, name) in enumerate(conditions):
    for k, (side, color) in enumerate((("before", "#bbbbbb"), ("final", ["#2a9d8f", "#3b6ea8", "#de693c"][j]))):
        values = [100 * result["set"][f"{s}@{condition}"][side] for s, _ in sets]
        axes[1].bar(np.arange(len(sets)) + (2 * j + k - 2.5) * width, values, width, color=color,
                    label=f"{name}: {'before' if side == 'before' else 'after'}")
axes[1].set_xticks(np.arange(len(sets)), [label for _, label in sets])
axes[1].set_ylabel("WER %")
axes[1].set_title("By set (grey = before, colour = after; clean, white 10 dB, babble 10 dB)")
axes[1].legend(fontsize=7, ncol=3)
fig.tight_layout()
by = result["condition"]
caption = (f"The one-time final test on test D ({result['units']} units: 898 never-seen utterances, each clean, with white noise at 10 dB "
           f"and with babble at 10 dB), app before -> after this round's protections, language detected automatically: all "
           f"{100 * result['all']['before']:.1f}% -> {100 * result['all']['final']:.1f}% ({100 * result['all']['difference']:+.1f} points, 95% "
           f"{100 * result['all']['low']:+.1f} to {100 * result['all']['high']:+.1f}); "
           + "; ".join(f"{name} {100 * by[c]['before']:.1f}% -> {100 * by[c]['final']:.1f}% ({100 * by[c]['difference']:+.1f}, "
                       f"{100 * by[c]['low']:+.1f} to {100 * by[c]['high']:+.1f})" for c, name in conditions)
           + f". Extra words per reference word {100 * result['insertions_per_word']['before']:.1f}% -> {100 * result['insertions_per_word']['final']:.1f}%. "
           f"Largest change: Kazakh read sentences in babble {100 * result['set']['fleurs-kk@babble-10']['before']:.1f}% -> "
           f"{100 * result['set']['fleurs-kk@babble-10']['final']:.1f}%. Clean sets unchanged within 0.3 points.")
log.fig(14, "final_test_d", fig, caption)
print(caption)
