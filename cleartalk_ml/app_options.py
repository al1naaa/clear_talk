"""The Conversation-mode options adopted so far on validation, and the evaluation names of option combinations."""

import json

from .paths import RESULTS

STATE = RESULTS / "app-options.json"


def setting_name(flags) -> str:
    """Evaluation name of the per-turn pipeline with exactly these options on (cleartalk_ml.evaluate, "auto+..." keys)."""
    flags = sorted(flags)
    if not flags:
        return "auto"
    return "auto+" + "+".join(flags)   # ("auto-region" is the first region-first variant, measured and not adopted)


def adopted_flags() -> list[str]:
    return json.loads(STATE.read_text(encoding="utf-8"))["flags"] if STATE.is_file() else []
