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
    if not config.CRISIS_HELPLINES_VERIFIED:
        print("\n" + "!" * 70 + "\n!! CRISIS HELPLINE NUMBERS IN api/safety.py ARE UNVERIFIED PLACEHOLDERS.\n"
              "!! A human must verify them and set SATSANG_HELPLINES_VERIFIED=1 before\n"
              "!! any real use. (Safety-first: do not ship crisis responses unverified.)\n"
              + "!" * 70 + "\n")


class ChatIn(BaseModel):
    message: str
    conversation_id: str | None = None
    user_id: str | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat")
def chat(inp: ChatIn) -> StreamingResponse:
    def sse():
        for event, payload in respond(inp.message, inp.conversation_id, inp.user_id):
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
                                          inp.get("conversation_id"), inp.get("user_id")):
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
