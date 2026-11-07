"""Start the demonstration server and open it in the browser: python -m app

PVD_HOST and PVD_PORT set where the server listens (default 127.0.0.1:8765; use
PVD_HOST=0.0.0.0 in a container). The README lists the other settings.
"""

import os
import threading
import time
import urllib.request
import webbrowser

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import uvicorn  # noqa: E402

HOST = os.environ.get("PVD_HOST", "127.0.0.1")
PORT = int(os.environ.get("PVD_PORT", "8765"))
URL = f"http://{'127.0.0.1' if HOST in ('0.0.0.0', '::') else HOST}:{PORT}"


def _open_when_up() -> None:
    for _ in range(120):
        try:
            urllib.request.urlopen(URL + "/api/status", timeout=1)
            webbrowser.open(URL)
            return
        except OSError:
            time.sleep(0.5)


if __name__ == "__main__":
    if os.environ.get("PVD_NO_BROWSER") != "1":
        threading.Thread(target=_open_when_up, daemon=True).start()
    uvicorn.run("app.api:app", host=HOST, port=PORT, log_level="warning")
