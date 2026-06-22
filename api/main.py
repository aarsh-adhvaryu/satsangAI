"""SatsangAI V1 — FastAPI app. Streams the pipeline as Server-Sent Events.

    source ~/.zshrc   # ANTHROPIC_API_KEY
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .pipeline import respond

app = FastAPI(title="SatsangAI V1")


class ChatIn(BaseModel):
    message: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat")
def chat(inp: ChatIn) -> StreamingResponse:
    def sse():
        for event, payload in respond(inp.message):
            yield f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
    return StreamingResponse(sse(), media_type="text/event-stream")
