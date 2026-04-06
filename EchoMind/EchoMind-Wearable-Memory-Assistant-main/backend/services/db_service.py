import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


class DBService:
    def __init__(self, db_path: Path) -> None:
        self.db_path = str(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    type TEXT,
                    person TEXT,
                    time TEXT,
                    is_reminder INTEGER NOT NULL DEFAULT 0,
                    priority TEXT,
                    due_time TEXT,
                    status TEXT NOT NULL DEFAULT 'captured',
                    embedding TEXT,
                    timestamp TEXT NOT NULL
                )
                """
            )
            # Safe in-place migration for existing databases.
            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(memories)").fetchall()
            }
            if "is_reminder" not in columns:
                conn.execute(
                    "ALTER TABLE memories ADD COLUMN is_reminder INTEGER NOT NULL DEFAULT 0"
                )
            if "priority" not in columns:
                conn.execute("ALTER TABLE memories ADD COLUMN priority TEXT")
            if "due_time" not in columns:
                conn.execute("ALTER TABLE memories ADD COLUMN due_time TEXT")
            if "status" not in columns:
                conn.execute(
                    "ALTER TABLE memories ADD COLUMN status TEXT NOT NULL DEFAULT 'captured'"
                )
            conn.commit()

    def add_memory(self, memory: dict[str, Any], embedding: list[float] | None) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO memories (
                    text, type, person, time, is_reminder, priority, due_time, status, embedding, timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory.get("text"),
                    memory.get("type"),
                    memory.get("person"),
                    memory.get("time"),
                    1 if memory.get("is_reminder") else 0,
                    memory.get("priority"),
                    memory.get("due_time"),
                    memory.get("status", "captured"),
                    json.dumps(embedding) if embedding else None,
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def get_all_memories(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM memories ORDER BY id DESC").fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_memories_with_embeddings(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE embedding IS NOT NULL ORDER BY id DESC"
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_memory_by_id(self, memory_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
        if not row:
            return None
        return self._row_to_dict(row)

    def get_today_reminders(self) -> list[dict[str, Any]]:
        start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM memories
                WHERE is_reminder = 1
                  AND status IN ('captured', 'pending')
                  AND due_time IS NOT NULL
                  AND due_time >= ?
                  AND due_time < ?
                ORDER BY due_time ASC, id DESC
                """,
                (start.isoformat(), end.isoformat()),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_pending_reminders(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM memories
                WHERE is_reminder = 1
                  AND status IN ('captured', 'pending')
                ORDER BY
                    CASE priority
                      WHEN 'high' THEN 1
                      WHEN 'medium' THEN 2
                      ELSE 3
                    END,
                    COALESCE(due_time, timestamp) ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        if "is_reminder" in data:
            data["is_reminder"] = bool(data["is_reminder"])
        if data.get("embedding"):
            data["embedding"] = json.loads(data["embedding"])
        return data
