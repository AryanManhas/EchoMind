from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, Body, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from config import Config
from services.audio_service import AudioService
from services.db_service import DBService
from services.embedding_service import EmbeddingService
from services.llm_service import LLMService
from services.nlp_service import NLPService
from services.search_service import SearchService

app = FastAPI(title="EchoMind Industry-Grade API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Services
db_service = DBService(Path(Config.DB_PATH))
nlp_service = NLPService()
# VOSK is kept here for initialization compatibility before we refactor audio_service
audio_service = AudioService(Config.VOSK_MODEL_PATH)
embedding_service = EmbeddingService(
    Config.EMBEDDING_MODEL, enabled=Config.ENABLE_EMBEDDINGS
)
search_service = SearchService(embedding_service)
llm_service = LLMService(
    enabled=Config.ENABLE_LLM,
    provider=Config.LLM_PROVIDER,
    model=Config.LLM_MODEL,
    endpoint_url=Config.OLLAMA_URL,
)

# Pydantic Models for requests
class TextPayload(BaseModel):
    text: str

class QueryPayload(BaseModel):
    query: str

class ChunkPayload(BaseModel):
    text: str
    session_id: str = "default-session"
    chunk_index: int = 0
    speaker: str = "unknown"

@app.get("/health")
def health():
    return {
        "status": "ok",
        "embeddings": embedding_service.diagnostics(),
        "vosk_model": Config.VOSK_MODEL_PATH,
        "llm_enabled": Config.ENABLE_LLM,
        "llm_model": Config.LLM_MODEL,
    }

@app.post("/add")
async def add_memory(
    text: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None)
):
    text_input = text
    audio_chunks = []

    if audio:
        suffix = Path(audio.filename or "clip.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = Path(tmp.name)
            content = await audio.read()
            tmp_path.write_bytes(content)
        try:
            transcription = audio_service.transcribe_audio(tmp_path)
            text_input = transcription["text"]
            audio_chunks = transcription.get("chunks", [])
        finally:
            tmp_path.unlink(missing_ok=True)

    if not text_input:
        raise HTTPException(status_code=400, detail="Provide text or audio input")

    memory = nlp_service.extract_memory(text_input)
    embedding = embedding_service.embed_text(memory["text"])
    memory_id = db_service.add_memory(memory, embedding)
    concise_response = nlp_service.to_concise_response(memory)

    return {
        "id": memory_id,
        "memory": memory,
        "response": concise_response,
        "chunks": audio_chunks,
    }

@app.post("/search")
def search_memories(payload: QueryPayload):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="query is required")
    records = db_service.get_all_memories()
    results = search_service.search(payload.query, records, top_k=5)
    return {"query": payload.query, "results": results}

@app.get("/memories")
def list_memories():
    records = db_service.get_all_memories()
    return {"count": len(records), "memories": records}

@app.get("/memories/{memory_id}")
def get_memory(memory_id: int):
    record = db_service.get_memory_by_id(memory_id)
    if not record:
        raise HTTPException(status_code=404, detail="memory not found")
    return {"memory": record}

@app.get("/reminders/today")
def reminders_today():
    reminders = db_service.get_today_reminders()
    return {"count": len(reminders), "reminders": reminders}

@app.get("/brief")
def proactive_brief():
    pending = db_service.get_pending_reminders(limit=5)
    if not pending:
        message = "No pending reminders. You are all caught up."
    else:
        top = pending[0]
        message = f"You have {len(pending)} pending reminders. Top item: {top.get('text', '')}"
    return {"message": message, "reminders": pending}

@app.post("/ask")
def ask_assistant(payload: QueryPayload):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="query is required")
    records = db_service.get_all_memories()
    retrieved = search_service.search(payload.query, records, top_k=5)
    answer = llm_service.answer_with_context(query=payload.query, memories=retrieved)
    return {
        "query": payload.query,
        "answer": answer["answer"],
        "source": answer["source"],
        "citations": answer.get("citations", []),
        "retrieved": retrieved,
    }

# WebSockets for real-time streaming ingestion
@app.websocket("/ws/audio_stream")
async def websocket_audio_stream(websocket: WebSocket):
    await websocket.accept()
    session_id = "ws-session"
    speaker = "unknown"
    try:
        while True:
            audio_data = await websocket.receive_bytes()
            # Fast Whisper / Silero VAD logical call
            transcription = audio_service.transcribe_audio_chunk(session_id, audio_data)
            if transcription and transcription["final"]:
                text = transcription["text"]
                memory = nlp_service.extract_memory(text)
                memory["session_id"] = session_id
                memory["speaker"] = speaker
                embedding = embedding_service.embed_text(memory["text"])
                memory_id = db_service.add_memory(memory, embedding)
                
                await websocket.send_json({
                    "transcription": text,
                    "final": True,
                    "saved": True,
                    "id": memory_id,
                    "response": nlp_service.to_concise_response(memory)
                })
            elif transcription:
                await websocket.send_json({
                    "transcription": transcription["text"],
                    "final": False
                })
    except Exception as e:
        print(f"WebSocket disconnected: {e}")
    finally:
        audio_service.finalize_session(session_id)

@app.post("/ingest_audio_chunk")
async def ingest_audio_chunk(
    session_id: str = Form("default-session"),
    chunk_index: int = Form(0),
    speaker: str = Form("unknown"),
    audio: UploadFile = File(...)
):
    suffix = Path(audio.filename or "clip.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        content = await audio.read()
        tmp_path.write_bytes(content)
    try:
        transcription = audio_service.transcribe_audio(tmp_path)
        text = (transcription.get("text") or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="could not transcribe audio")

        memory = nlp_service.extract_memory(text)
        memory["session_id"] = session_id
        memory["chunk_index"] = chunk_index
        memory["speaker"] = speaker

        embedding = embedding_service.embed_text(memory["text"])
        memory_id = db_service.add_memory(memory, embedding)
        response = nlp_service.to_concise_response(memory)
        return {
            "saved": True,
            "id": memory_id,
            "session_id": session_id,
            "chunk_index": chunk_index,
            "speaker": speaker,
            "transcript": text,
            "chunks": transcription.get("chunks", []),
            "is_reminder": memory.get("is_reminder", False),
            "priority": memory.get("priority"),
            "response": response,
        }
    finally:
        tmp_path.unlink(missing_ok=True)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
