"""Start the local app and open a browser, or reuse an existing ClearTalk process."""
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORTS = (8765, 8766, 8767)


SERVICE_VERSION = "1.0.1"


def ready(url):
    """True only for a ready ClearTalk service of THIS version (older ones are left alone)."""
    try:
        with urllib.request.urlopen(url + "/api/health", timeout=1) as response:
            data = json.load(response)
            return (data.get("app") == "ClearTalk" and data.get("status") == "ready"
                    and data.get("version") == SERVICE_VERSION)
    except (OSError, ValueError):
        return False


def occupied(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=.3):
            return True
    except OSError:
        return False


def choose_port(preferred_port=None):
    for port in PORTS:
        if ready(f"http://127.0.0.1:{port}"):
            return port, True
    ports = PORTS if preferred_port is None else (preferred_port, *(port for port in PORTS if port != preferred_port))
    for port in ports:
        if not occupied(port):
            return port, False
    raise RuntimeError("Ports 8765, 8766 and 8767 are occupied. Choose a free port before retrying; no process was stopped automatically.")


def main():
    port, running = choose_port(preferred_port=8765)
    url = f"http://127.0.0.1:{port}"
    if running:
        webbrowser.open(url)
        print(f"ClearTalk is already running at {url}")
        return
    print(f"Starting ClearTalk at {url}. Keep this window open; Ctrl+C stops this app.")
    environment = os.environ.copy()
    environment.setdefault("CLEARTALK_DATA_DIR", str(ROOT / "data/runtime-1.0.1"))
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=ROOT, env=environment,
    )
    try:
        for _ in range(60):
            if process.poll() is not None:
                raise RuntimeError("ClearTalk could not start. Check the error above.")
            if ready(url):
                webbrowser.open(url)
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("The app did not become ready within 30 seconds.")
        process.wait()
    except KeyboardInterrupt:
        print("\nStopping ClearTalk.")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
