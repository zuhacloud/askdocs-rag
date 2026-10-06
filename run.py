"""Start AskDocs with one command: the API and the chat UI, then open the browser.

Usage (from the project folder, with the virtual environment's Python):
    python run.py               # opens the app in your browser
    python run.py --no-browser  # just start the services

Ctrl+C stops everything. Ports can be changed with the API_PORT and UI_PORT environment variables.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
API_PORT = int(os.getenv("API_PORT", "8000"))
UI_PORT = int(os.getenv("UI_PORT", "8501"))
API_URL = f"http://127.0.0.1:{API_PORT}"  # not localhost: on Windows that tries IPv6 first and stalls ~2 s
UI_URL = f"http://localhost:{UI_PORT}"


def say(message: str) -> None:
    print(f"[askdocs] {message}", flush=True)


def is_up(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def wait_until_up(url: str, process: subprocess.Popen, what: str, timeout: float = 900) -> None:
    """Poll `url` until it answers. Fails fast if the process dies while we wait."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_up(url):
            return
        if process.poll() is not None:
            raise SystemExit(f"[askdocs] The {what} stopped during startup. See the messages above.")
        time.sleep(1)
    raise SystemExit(f"[askdocs] The {what} did not start within {timeout:.0f} seconds.")


def check_llm() -> None:
    """Warn early if the configured language model isn't reachable (questions would fail)."""
    sys.path.insert(0, str(ROOT))
    from askdocs.config import get_settings  # light import: no models are loaded

    base_url = get_settings().llm_base_url.rstrip("/")
    if not is_up(f"{base_url}/models"):
        say(f"Note: the language model at {base_url} isn't reachable yet.")
        say("      Start Ollama (or update LLM_BASE_URL in .env); the app will connect once it's up.")


def main() -> None:
    os.chdir(ROOT)
    if not Path(".env").exists() and Path(".env.example").exists():
        shutil.copy(".env.example", ".env")
        say("Created .env from .env.example")
    check_llm()

    processes: list[subprocess.Popen] = []
    try:
        if is_up(f"{API_URL}/health"):
            say(f"API already running at {API_URL}, reusing it.")
        else:
            say("Starting the API (loading models; the first run downloads ~220 MB)...")
            api = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "askdocs.api:app", "--port", str(API_PORT), "--no-access-log"]
            )
            processes.append(api)
            wait_until_up(f"{API_URL}/health", api, "API")
        say(f"API ready at {API_URL}  (interactive docs: {API_URL}/docs)")

        ui = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py",
             "--server.port", str(UI_PORT), "--server.headless", "true"],
            env={**os.environ, "API_URL": API_URL},
            stdout=subprocess.DEVNULL,
        )
        processes.append(ui)
        wait_until_up(f"{UI_URL}/_stcore/health", ui, "UI")
        say(f"AskDocs is running at {UI_URL}   (press Ctrl+C to stop)")
        if "--no-browser" not in sys.argv:
            webbrowser.open(UI_URL)

        # Keep running until a child exits or the user presses Ctrl+C.
        while all(p.poll() is None for p in processes):
            time.sleep(1)
        say("A service stopped unexpectedly; shutting down.")
    except KeyboardInterrupt:
        say("Stopping...")
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    main()
