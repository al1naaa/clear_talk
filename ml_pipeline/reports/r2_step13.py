"""Round 2: write step 13 (error analysis on validation) into round2/PIPELINE.md from the saved transcripts."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cleartalk_ml import analysis, round2
from cleartalk_ml import pipeline_log as log
from cleartalk_ml.paths import RESULTS

log.use(round2.R2)
p = lambda value: f"{100 * value:.1f}%"  # noqa: E731
selection = json.loads((RESULTS / "selection-round2.json").read_text(encoding="utf-8"))
round1 = json.loads((RESULTS / "selection.json").read_text(encoding="utf-8"))
ours = f"runs/{selection['run']}/ct2-step-{selection['chosen_step']:05d}"
small_ours = f"runs/{round1['run']}/ct2-step-{round1['chosen_step']:05d}"
three = (("start", "kazakh"), ("small", small_ours), ("ours", ours))
kinds = {name: analysis.edit_kinds("mixed-val", system) for name, system in three}
position = {name: analysis.by_position(system) for name, system in three}
script = {name: analysis.wrong_script(system) for name, system in three}
sources = {name: analysis.ksc2_by_source(system) for name, system in three}
moves = {name: analysis.better_same_worse("mixed-val", system, ours) for name, system in three[:2]}
prompt = analysis.prompt_effect(ours)
worse_sources = [s for s in sorted(sources["ours"]) if sources["ours"][s]["wer"] > sources["start"][s]["wer"]]
better_sources = [s for s in sorted(sources["ours"]) if sources["ours"][s]["wer"] < sources["start"][s]["wer"]]

log.set_step(13, "done",
             f"validation only. On {moves['start']['units']} mixed recordings our large model makes fewer errors than the starting model on "
             f"{moves['start']['better']} (same {moves['start']['same']}, more {moves['start']['worse']}) and than our small model on {moves['small']['better']} "
             f"(same {moves['small']['same']}, more {moves['small']['worse']}). Starting model → ours: missing words {kinds['start']['deletions']:,} → "
             f"{kinds['ours']['deletions']:,}, wrong words {kinds['start']['substitutions']:,} → {kinds['ours']['substitutions']:,}, extra words "
             f"{kinds['start']['insertions']:,} → {kinds['ours']['insertions']:,}; our small model had {kinds['small']['substitutions']:,} wrong words - the "
             f"acoustic gap this round was meant to close. By place in the recording the starting model collapses after the first change of language ("
             + ", ".join(f"{k.replace('part', 'utterance')} {p(position['start'][k])}" for k in position["start"]) + "); ours is level ("
             + ", ".join(f"{k.replace('part', 'utterance')} {p(position['ours'][k])}" for k in position["ours"])
             + f"). English words written in Cyrillic: {p(script['start']['en'])} → {p(script['ours']['en'])}. Kazakh KSC2 by source, starting model → ours: "
             + ", ".join(f"{s} {p(sources['start'][s]['wer'])} → {p(sources['ours'][s]['wer'])}" for s in sorted(sources["ours"]))
             + f" - worse on {', '.join(worse_sources) or 'no source'}, better on {', '.join(better_sources) or 'no source'}: the loss on KSC2 is spread over the "
             "sources, not one failure. The language our model detects for a mixed recording is its first utterance's in only "
             f"{p(prompt['first_language_detected'])} of cases, yet WER is the same either way ({p(prompt['wer_when_first'])} vs {p(prompt['wer_when_other'])}): the "
             "token does not steer what it writes. Reading the worst recordings: names (Arly Velasquez), numbers written as digits where the "
             "reference spells them out, hesitation sounds kept in KSC2 references, and joined or split compounds (Super-G)")
print("logged")
