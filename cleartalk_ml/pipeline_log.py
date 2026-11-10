"""Small helpers for ml_pipeline/PIPELINE.md: figures, checklist rows and dated notes.

``guard.fig`` appends one ledger line per figure. A notebook that is run again with fresh data
would leave the outdated line next to the new one, so the old line is removed first: the log
always describes the figure that is on disk.
"""

import re
import sys
from datetime import date

from .paths import PIPELINE

LOG = PIPELINE / "PIPELINE.md"
STATE = PIPELINE  # where figures, the data profile and the guard state of the current round live


def use(folder):
    """Point the helpers at another round's folder (its own PIPELINE.md, figures and guard state)."""
    global LOG, STATE
    STATE = folder
    LOG = folder / "PIPELINE.md"


def _guard():
    if str(PIPELINE) not in sys.path:
        sys.path.insert(0, str(PIPELINE))
    import guard

    return guard


def fig(step: int, name: str, figure, explanation: str):
    """Save a figure through the guard and keep exactly one ledger line for it."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    filename = f"{int(step):02d}_{slug}.png"
    if LOG.is_file():
        lines = LOG.read_text(encoding="utf-8").split("\n")
        kept = [line for line in lines if not line.startswith(f"- Figure {filename}:")]
        if len(kept) != len(lines):
            LOG.write_text("\n".join(kept), encoding="utf-8")
    return _guard().fig(step, name, figure, " ".join(explanation.split()), state_dir=STATE)


def set_step(number: int, status: str, result: str):
    """Rewrite one row of the checklist table."""
    text = LOG.read_text(encoding="utf-8")
    pattern = re.compile(rf"^\| {number} \| ([^|]+) \|[^\n]*$", re.M)
    match = pattern.search(text)
    if not match:
        raise ValueError(f"Step {number} is not in the checklist")
    row = f"| {number} | {match.group(1).strip()} | {status} | {' '.join(result.split())} |"
    LOG.write_text(text[:match.start()] + row + text[match.end():], encoding="utf-8")


def note(text: str, heading: str = "## Decisions"):
    """Add a dated line at the end of a section (before the next heading)."""
    content = LOG.read_text(encoding="utf-8")
    line = f"- {date.today().isoformat()} · {' '.join(text.split())}\n"
    if line in content:
        return
    start = content.index(heading)
    following = content.find("\n## ", start + len(heading))
    if following < 0:
        LOG.write_text(content.rstrip("\n") + "\n" + line, encoding="utf-8")
    else:
        LOG.write_text(content[:following].rstrip("\n") + "\n" + line + content[following:], encoding="utf-8")
