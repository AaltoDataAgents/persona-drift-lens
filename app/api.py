"""HTTP API and static files for the Persona Vector Teaching Demonstration.

The server starts without waiting for the model: the example conversations can
be replayed while the model loads, or if it cannot load at all.

Each page sends a conversation id with its requests. Reloading creates a new id;
requests without an id share a legacy conversation. Generation is serialised
across conversations. Live sessions are saved in runs/sessions, separately from
the recorded examples.
"""

from __future__ import annotations

import json
import os
import queue
import secrets
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .engine import VOICE_DESCRIPTIONS, Conversation, Engine

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
SCENARIOS = ROOT / "scenarios"
EXAMPLES = ROOT / "examples"
SESSIONS = ROOT / "runs" / "sessions"
SPACE_NAME = os.environ.get("PVD_SPACE", "qwen3-4b-instruct-2507")

engine = Engine(ROOT / "space" / f"{SPACE_NAME}.npz")
busy = threading.Lock()
MAX_CONVERSATIONS = 50


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    seed: int | None = None


class Options(BaseModel):
    max_tokens: int | None = Field(default=None, ge=16, le=600)
    system_prompt: str | None = None


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _example_paths() -> list[Path]:
    """Recorded examples made with the model of the loaded persona space; others would mislabel roles."""
    model = engine.space.get("model")
    paths = []
    for path in sorted(EXAMPLES.glob("*.json"), key=lambda p: p.stem) if EXAMPLES.exists() else []:
        try:
            if not model or _load_json(path).get("model") == model:
                paths.append(path)
        except (OSError, json.JSONDecodeError):
            continue
    return paths


class Live(Conversation):
    """A window's conversation, where it is saved, and whether it is writing a reply."""

    def __init__(self) -> None:
        super().__init__()
        self.writing = threading.Lock()
        self.used = time.time()
        self.file = SESSIONS / f"session-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}.json"


conversations: dict[str, Live] = {}
conversations_lock = threading.Lock()


def _key(client: str | None) -> str:
    # Requests without an id, such as the soak test's, share one conversation.
    return (client or "default")[:64]


def _conversation(client: str | None) -> Live:
    key = _key(client)
    with conversations_lock:
        conv = conversations.get(key)
        if conv is None:
            idle = [k for k, c in conversations.items() if not c.writing.locked()]
            while len(conversations) >= MAX_CONVERSATIONS and idle:
                oldest = min(idle, key=lambda k: conversations[k].used)
                del conversations[oldest]
                idle.remove(oldest)
            conv = conversations[key] = Live()
        conv.used = time.time()
        return conv


def _save(conv: Live) -> None:
    if not conv.turns:
        return
    SESSIONS.mkdir(parents=True, exist_ok=True)
    payload = {
        "title": f"Live session {time.strftime('%d %b %H:%M')}",
        "kind": "session",
        "model": engine.space.get("model"),
        "layer": engine.space.get("layer"),
        "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
        "turns": conv.turns,
    }
    with open(conv.file, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app() -> FastAPI:
    app = FastAPI(title="Persona Vector Teaching Demonstration", version="1.0")

    @app.middleware("http")
    async def no_stale_pages(request, call_next):
        # Avoid serving stale page files after an update.
        response = await call_next(request)
        if not request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.on_event("startup")
    def _startup() -> None:
        with_model = os.environ.get("PVD_NO_MODEL") != "1"
        threading.Thread(target=engine.load, args=(with_model,), daemon=True).start()

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        info = engine.status()
        info["busy"] = busy.locked()
        info["space"] = SPACE_NAME
        info["examples"] = len(_example_paths())
        return info

    @app.get("/api/space")
    def space() -> dict[str, Any]:
        path = ROOT / "results" / f"space_{SPACE_NAME}.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"No persona space has been built for {SPACE_NAME}.")
        data = _load_json(path)
        # The engine adds the default Assistant as the last candidate for the closest pattern.
        data["roles"]["Assistant"] = {"scale": 1.0, "description": "The model's own default pattern, from its replies under neutral instructions."}
        data["role_order"] = list(data["roles"])
        data["voice_descriptions"] = VOICE_DESCRIPTIONS
        # How often the label model named the right voice on questions it never saw (tools/build_labels.py).
        labels = ROOT / "results" / f"labels_{SPACE_NAME}.json"
        if labels.exists():
            report = _load_json(labels)
            data["label_accuracy"] = report.get("family_top1")
        return data

    @app.get("/api/scenarios")
    def scenarios() -> list[dict[str, Any]]:
        # "finding" describes the 4B recordings; "findings" holds the text for other models.
        items = [_load_json(p) for p in sorted(SCENARIOS.glob("*.json"))]
        for item in items:
            item["finding"] = item.get("findings", {}).get(engine.space.get("model"), item["finding"])
        return items

    @app.get("/api/examples")
    def examples() -> list[dict[str, Any]]:
        if not EXAMPLES.exists():
            return []
        items = []
        for path in _example_paths():
            data = _load_json(path)
            items.append({"name": path.stem, "title": data.get("title", path.stem), "kind": data.get("kind", "session"),
                          "turns": len(data.get("turns", [])), "saved": data.get("saved", "")})
        return items

    @app.get("/api/examples/{name}")
    def example(name: str) -> dict[str, Any]:
        path = (EXAMPLES / f"{name}.json").resolve()
        if path.parent != EXAMPLES.resolve() or not path.exists():
            raise HTTPException(status_code=404, detail="No example with that name.")
        return _load_json(path)

    @app.post("/api/options")
    def options(body: Options) -> dict[str, Any]:
        for key, value in body.model_dump(exclude_none=True).items():
            engine.options[key] = value
        return engine.options

    @app.post("/api/reset")
    def reset(x_conversation: str | None = Header(default=None)) -> dict[str, Any]:
        conv = _conversation(x_conversation)
        if conv.writing.locked():
            raise HTTPException(status_code=409, detail="Stop the current reply before starting a new chat.")
        with conversations_lock:
            conversations[_key(x_conversation)] = Live()
        return {"ok": True}

    @app.post("/api/stop")
    def stop(x_conversation: str | None = Header(default=None)) -> dict[str, Any]:
        _conversation(x_conversation).stop.set()
        return {"ok": True}

    @app.post("/api/chat")
    def chat(body: ChatRequest, x_conversation: str | None = Header(default=None)) -> StreamingResponse:
        if engine.state != "ready":
            raise HTTPException(status_code=503, detail=f"The live model is not available ({engine.state}).")
        conv = _conversation(x_conversation)
        if not conv.writing.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="A reply is still being written.")
        conv.stop.clear()
        events: queue.Queue = queue.Queue()

        def run() -> None:
            acquired = busy.acquire(blocking=False)
            try:
                if not acquired:
                    # Another window's reply is being written; this one waits its turn.
                    events.put(("wait", {}))
                    while not acquired and not conv.stop.is_set():
                        acquired = busy.acquire(timeout=0.5)
                    if not acquired:
                        raise RuntimeError("Stopped before the reply began.")
                turn = engine.chat(conv, body.message, lambda kind, data: events.put((kind, data)), seed=body.seed)
                _save(conv)
                events.put(("done", {k: v for k, v in turn.items() if k != "tokens"}))
            except Exception as exc:  # noqa: BLE001
                events.put(("error", {"message": str(exc)}))
            finally:
                if acquired:
                    busy.release()
                conv.writing.release()
                events.put(None)

        threading.Thread(target=run, daemon=True).start()

        def stream():
            while True:
                item = events.get()
                if item is None:
                    return
                yield _sse(*item)

        # Noppe puts nginx in front of each session; without this header it may hold tokens back.
        return StreamingResponse(stream(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB / "index.html")

    app.mount("/static", StaticFiles(directory=WEB), name="static")
    return app


app = create_app()
