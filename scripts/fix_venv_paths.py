"""Repair the virtual environments after the project folder was moved or renamed.

    python3 scripts/fix_venv_paths.py

The app itself keeps working after a move (it is started with ``.venv/bin/python``), but helper
commands inside ``.venv/bin`` and ``.venv-train/bin`` (pip, marimo, uvicorn, activate) remember the
folder they were created in. This rewrites that old folder to the current one. Safe to run twice.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def old_root(venv: Path):
    """The project folder recorded in this environment's scripts, or None if it is already current."""
    for script in sorted((venv / "bin").iterdir()):
        if script.is_symlink() or not script.is_file():
            continue
        try:
            first = script.read_text(errors="ignore").splitlines()[0]
        except IndexError:
            continue
        match = re.match(rf"#!(.+)/{re.escape(venv.name)}/bin/python", first)
        if match:
            return None if Path(match.group(1)) == ROOT else match.group(1)
    return None


def main():
    for name in (".venv", ".venv-train"):
        venv = ROOT / name
        if not venv.is_dir():
            continue
        previous = old_root(venv)
        if previous is None:
            print(f"{name}: already points to {ROOT}")
            continue
        fixed = 0
        for path in [*(venv / "bin").iterdir(), venv / "pyvenv.cfg"]:
            if path.is_symlink() or not path.is_file():
                continue
            try:
                text = path.read_text()
            except UnicodeDecodeError:
                continue
            if previous in text:
                path.write_text(text.replace(previous, str(ROOT)))
                fixed += 1
        print(f"{name}: {fixed} files updated ({previous} -> {ROOT})")


if __name__ == "__main__":
    main()
