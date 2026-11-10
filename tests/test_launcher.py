from scripts import launch


def test_launcher_reuses_a_running_current_service(monkeypatch):
    monkeypatch.setattr(launch, 'ready', lambda url: url.endswith(':8766'))
    monkeypatch.setattr(launch, 'occupied', lambda port: True)
    assert launch.choose_port() == (8766, True)


def test_launcher_uses_free_fallback_without_stopping_old_process(monkeypatch):
    monkeypatch.setattr(launch, 'ready', lambda url: False)
    monkeypatch.setattr(launch, 'occupied', lambda port: port == 8765)
    assert launch.choose_port() == (8766, False)


def test_launcher_fails_safely_if_all_ports_are_occupied(monkeypatch):
    import pytest
    monkeypatch.setattr(launch, 'ready', lambda url: False)
    monkeypatch.setattr(launch, 'occupied', lambda port: True)
    with pytest.raises(RuntimeError, match='no process was stopped'):
        launch.choose_port()


def test_service_version_matches_backend():
    import re
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "backend/app.py").read_text()
    assert re.search(r'SERVICE_VERSION = "([^"]+)"', source).group(1) == launch.SERVICE_VERSION
