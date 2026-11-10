import os

import pytest


@pytest.fixture(autouse=True)
def no_cleartalk_environment(monkeypatch):
    """Tests build their own folders. A CLEARTALK_* variable left in the shell (for example CLEARTALK_MODELS_DIR)
    must never point a test at the real models or recordings, where it could overwrite a validated profile."""
    for name in [name for name in os.environ if name.startswith("CLEARTALK_")]:
        monkeypatch.delenv(name)
