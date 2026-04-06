from pathlib import Path


class Config:
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data"
    DB_PATH = DATA_DIR / "memories.db"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    ENABLE_EMBEDDINGS = False
    VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"
    ENABLE_LLM = False
    LLM_PROVIDER = "ollama"
    LLM_MODEL = "llama3.1:8b"
    OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
