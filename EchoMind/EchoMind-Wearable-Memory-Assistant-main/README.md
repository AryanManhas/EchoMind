# Wearable AI Personal Memory Assistant Prototype

Industry-style but student-achievable local prototype:

- `backend/`: Flask API + Whisper STT + spaCy/rule NLP + SQLite + semantic/fallback search
- `mobile_app/`: Flutter app with Today, Home, Memories, Search, Assistant + voice recording
  - Includes deep-link style memory routing: `/memory/<id>` and `memory://id/<id>`
  - Includes queryless reminder view (`Today`) with proactive brief

## 1) Architecture

Pendant input (simulated text/audio) -> Mobile App (Flutter) -> Flask API -> Vosk (offline real-time STT) -> NLP (spaCy + rules) -> Memory structuring -> SQLite (+ optional embeddings) -> Search -> Response to app

## 2) Backend Setup

From the repo root:

```bash
cd backend
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt
# Download Vosk model for real-time STT
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
# The model folder should be in backend/vosk-model-small-en-us-0.15
python -m spacy download en_core_web_sm
python app.py
```

Backend runs on `http://localhost:5000`.

### API Endpoints

- `POST /add`
  - JSON: `{ "text": "Meet Rahul tomorrow at 4 PM" }`
  - Multipart: `audio=<file>`
- `POST /search`
  - JSON: `{ "query": "meeting with Rahul" }`
- `POST /ingest_chunk`
  - JSON:
    `{ "text": "...", "session_id": "session-001", "chunk_index": 1, "speaker": "user" }`
  - Used for continuous conversation ingestion simulation
- `POST /ingest_audio_chunk`
  - Multipart:
    - `audio=<recorded file>`
    - `session_id=session-001`
    - `chunk_index=1`
    - `speaker=user`
  - Real recorded chunk ingestion + Whisper + reminder extraction
- `POST /ingest_audio_stream_chunk`
  - Form data:
    - `session_id=session-001`
    - `speaker=user`
  - Body: Raw audio bytes (16-bit mono PCM at 16kHz)
  - Real-time streaming audio chunk ingestion + Vosk STT + live reminder extraction
- `POST /finalize_audio_session`
  - Form data: `session_id=session-001`
  - Finalizes the streaming session and extracts any remaining memory
- `GET /memories`
  - List all memories
- `POST /ask`
  - JSON: `{ "query": "What did I plan with Rahul?" }`
  - Retrieval-augmented answer using top matched memories
- `GET /memories/<id>`
  - Fetch one memory record (used by citation deep links)
- `GET /reminders/today`
  - Queryless mode: reminders due today
- `GET /brief`
  - Proactive summary + top pending reminders

### Example response from `/add`

```json
{
  "id": 1,
  "memory": {
    "type": "meeting",
    "person": "Rahul",
    "time": "tomorrow",
    "text": "Meet Rahul tomorrow at 4 PM"
  },
  "response": "Meeting with Rahul tomorrow at 4 PM"
}
```

## 3) Mobile App Setup (Flutter)

Create platform folders if this is a fresh folder:

```bash
cd mobile_app
flutter create .
flutter pub get
flutter run
```

Important:

- Voice recording is available in Home -> Voice Module.
- For Android emulator, API base URL is auto-selected as `http://10.0.2.2:5000`.
- For physical device, update `_lanIp` in `mobile_app/lib/api_service.dart`.

## 4) Core Design Choices

- **Modular backend services**: `audio_service`, `nlp_service`, `embedding_service`, `search_service`, `db_service`
- **Hybrid extraction**: rules (`meet/call/send`) + spaCy entities (`PERSON`, `DATE`, `TIME`) + regex fallback
- **Semantic search first**: MiniLM embeddings + cosine similarity when available
- **Fallback search**: keyword overlap + fuzzy matching when embeddings are unavailable
- **Real-time streaming STT**: Vosk for millisecond-level live audio transcription and reminder extraction
- **LLM-ready RAG path**: `/ask` retrieves memories then answers using optional local LLM

## 6) LLM Integration

Current behavior:

- Retrieval always works through `search_service`.
- If `ENABLE_LLM = True`, backend calls local Ollama model and answers with retrieved context.
- If disabled/unavailable, backend returns deterministic fallback answer.

Recommended local model for this project:

- `llama3.1:8b` on Ollama (balanced quality + local feasibility)

Enable in `backend/config.py`:

```python
ENABLE_LLM = True
LLM_PROVIDER = "ollama"
LLM_MODEL = "llama3.1:8b"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
```

Then run:

```bash
ollama pull llama3.1:8b
ollama serve
python app.py
```

## 5) Suggested Local Test Flow

1. Start backend.
2. Open Flutter app -> Home.
3. Add: `Meet Rahul tomorrow at 4 PM`.
4. Go to Memories to confirm record saved.
5. Search: `When is my meeting with Rahul?`.
6. Validate top result relevance.
