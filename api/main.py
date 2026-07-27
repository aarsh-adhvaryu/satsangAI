"""SatsangAI V1 — FastAPI app. Streams the pipeline as Server-Sent Events.

    source ~/.zshrc   # ANTHROPIC_API_KEY
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from . import config, observability
from .pipeline import respond

app = FastAPI(title="SatsangAI V1")
_WEB = Path(__file__).resolve().parent / "web" / "index.html"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_WEB)


@app.on_event("startup")
def _warn_helplines() -> None:
    # A helpline config that failed to load is an operational emergency, not a warning to
    # scroll past: people in crisis would silently receive fewer numbers than intended.
    from . import safety
    for err in safety.validate_configs():   # ALL configs, not just the ones this env uses
        print("\n" + "!" * 70 + f"\n!! {err}\n!! Regional/country helplines will NOT be shown. "
              "The India-core lines still work.\n" + "!" * 70 + "\n", flush=True)
    if not config.CRISIS_HELPLINES_VERIFIED:
        print("\n" + "!" * 70 + "\n!! CRISIS HELPLINE NUMBERS IN api/safety.py ARE UNVERIFIED PLACEHOLDERS.\n"
              "!! A human must verify them and set SATSANG_HELPLINES_VERIFIED=1 before\n"
              "!! any real use. (Safety-first: do not ship crisis responses unverified.)\n"
              + "!" * 70 + "\n")


class ChatIn(BaseModel):
    message: str
    conversation_id: str | None = None
    user_id: str | None = None
    mode: str | None = None            # explicit user selection; only 'shastrarth' is honoured


@app.get("/modes")
def modes() -> dict:
    """Which modes a client may offer in its picker.

    counseling and teaching are always auto-routed and need no picker entry. shastrarth
    is opt-in and appears only when SATSANG_SHASTRARTH=1 — it currently fails two of its
    six gates, so it stays off until the acharya-school rows are enriched.
    """
    return {
        "auto": ["counseling", "teaching"],
        "selectable": ["shastrarth"] if config.SHASTRARTH_ENABLED else [],
        "shastrarth": {
            "enabled": config.SHASTRARTH_ENABLED,
            "label": "Shastrarth (scholarly debate)",
            "status": "experimental",
            "note": ("Compares Advaita / Vishishtadvaita / Dvaita / Shuddhadvaita positions. "
                     "Grounding for the acharya schools is untranslated OCR, so citation "
                     "accuracy is below the standard of the other modes."),
        },
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# --------------------------------------------------------------------------- #
#  §7 "Control is absolute" — the user owns their memory.                      #
#  Every item can be viewed, edited or deleted; memory can be paused; all data #
#  can be exported or cleared. These are a launch requirement, not a feature:  #
#  we promise it in the proposal and we store real personal disclosures.       #
# --------------------------------------------------------------------------- #
class FactEdit(BaseModel):
    user_id: str
    text: str


class PrefsIn(BaseModel):
    user_id: str
    paused: bool | None = None
    consent: bool | None = None
    language: str | None = None
    length: str | None = None
    style: str | None = None


class FeedbackIn(BaseModel):
    rating: str                       # "up" | "down"
    user_id: str | None = None
    conversation_id: str | None = None
    message: str = ""
    reply: str = ""
    note: str = ""


def _facts_store():
    """The configured fact store; the control endpoints need the richer JSON one."""
    from .store import fact_store
    return fact_store()


def _require_editable(fs):
    from fastapi import HTTPException
    if not hasattr(fs, "items"):
        # PgMemoryStore has facts()/add() only. Fail loudly rather than pretend the
        # user's delete succeeded — silently dropping a deletion is the worst outcome.
        raise HTTPException(501, f"{type(fs).__name__} does not support memory editing yet; "
                                 "run with SATSANG_STORE=memory or add the methods to api/pg.py")


@app.get("/memory")
def memory_list(user_id: str) -> dict:
    from .memory import PrefsStore
    fs = _facts_store()
    _require_editable(fs)
    return {"user_id": user_id, "facts": fs.items(user_id),
            "preferences": PrefsStore().get(user_id)}


@app.patch("/memory/{fact_id}")
def memory_edit(fact_id: str, inp: FactEdit) -> dict:
    fs = _facts_store()
    _require_editable(fs)
    return fs.update(inp.user_id, fact_id, inp.text)


@app.delete("/memory/{fact_id}")
def memory_delete(fact_id: str, user_id: str) -> dict:
    fs = _facts_store()
    _require_editable(fs)
    return fs.delete(user_id, fact_id)


@app.delete("/memory")
def memory_clear(user_id: str) -> dict:
    fs = _facts_store()
    _require_editable(fs)
    return fs.clear(user_id)


@app.post("/memory/prefs")
def memory_prefs(inp: PrefsIn) -> dict:
    from .memory import PrefsStore
    return PrefsStore().set(inp.user_id, paused=inp.paused, consent=inp.consent,
                            language=inp.language, length=inp.length, style=inp.style)


@app.get("/memory/export")
def memory_export(user_id: str) -> dict:
    from .memory import export_user
    from .store import conversation_store
    fs = _facts_store()
    _require_editable(fs)
    return export_user(user_id, conversations=conversation_store(), facts=fs)


@app.post("/feedback")
def feedback(inp: FeedbackIn) -> dict:
    """Turn-level rating — the only thing that makes served conversations trainable."""
    from .memory import FeedbackStore
    return FeedbackStore().add(user_id=inp.user_id, conversation_id=inp.conversation_id,
                               rating=inp.rating, message=inp.message,
                               reply=inp.reply, note=inp.note)


@app.post("/chat")
def chat(inp: ChatIn) -> StreamingResponse:
    def sse():
        for event, payload in respond(inp.message, inp.conversation_id, inp.user_id,
                                      mode=inp.mode):
            yield f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
    return StreamingResponse(sse(), media_type="text/event-stream")


@app.websocket("/ws")
async def ws_chat(ws: WebSocket) -> None:
    """WebSocket streaming (proposal §21): client sends {message, conversation_id?, user_id?};
    server streams the same {event, ...} frames as /chat, JSON per message."""
    await ws.accept()
    try:
        while True:
            inp = await ws.receive_json()
            for event, payload in respond(inp.get("message", ""),
                                          inp.get("conversation_id"), inp.get("user_id"),
                                          mode=inp.get("mode")):
                await ws.send_json({"event": event, "data": payload})
            await ws.send_json({"event": "end", "data": {}})   # turn boundary
    except WebSocketDisconnect:
        return


@app.get("/metrics")
def metrics() -> dict:
    """Observability: latency percentiles, per-stage timings, crisis rate over recent requests."""
    return observability.summary()


@app.get("/traces")
def traces(n: int = 50) -> dict:
    return {"traces": observability.recent(n)}
