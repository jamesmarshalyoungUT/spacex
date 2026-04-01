from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.agent import build_agent_session


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str
    trace: list[dict[str, Any]]


app = FastAPI(title="SpaceX Agentic API")
_executors: dict[str, Any] = {}
_static_dir = Path(__file__).resolve().parent / "static"

app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


def _get_executor(session_id: str):
    if session_id not in _executors:
        _executors[session_id] = build_agent_session(verbose=False)
    return _executors[session_id]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_static_dir / "index.html")


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    session = _get_executor(payload.session_id)
    result = session.ask(payload.message)
    return ChatResponse(answer=result.get("output", ""), trace=result.get("trace", []))
