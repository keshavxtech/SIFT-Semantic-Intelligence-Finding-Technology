import json
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
from app.core.config import settings
from app.core.database import get_db_connection
from app.core.logging import logger


class ChatRepository:
    """Persistence repository for conversational sessions and messages."""

    def __init__(self, db_path: Optional[Path] = None, provider: Optional[str] = None):
        self._db_path = db_path or settings.database_path
        self._provider = provider or ("sqlite" if (db_path and db_path != settings.database_path) else settings.DATABASE_PROVIDER)
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Generator[Any, None, None]:
        with get_db_connection(db_path=self._db_path, provider=self._provider) as conn:
            yield conn

    def _init_db(self) -> None:
        """Initializes tables for chat sessions and conversational turns."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    citations_json TEXT DEFAULT '[]',
                    metadata_json TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id, created_at ASC);")
        logger.info(f"Initialized chat repository (provider: {self._provider}) at: {self._db_path}")

    def create_session(
        self,
        session_id: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        sid = session_id or f"ses-{uuid.uuid4().hex[:10]}"
        session_title = title or "New Document Conversation"
        now_str = datetime.utcnow().isoformat()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO chat_sessions (session_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?);
                """,
                (sid, session_title, now_str, now_str),
            )

        return self.get_session(sid) or {
            "session_id": sid,
            "title": session_title,
            "created_at": now_str,
            "updated_at": now_str,
            "message_count": 0,
        }

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT s.session_id, s.title, s.created_at, s.updated_at,
                       COUNT(m.message_id) AS message_count
                FROM chat_sessions s
                LEFT JOIN chat_messages m ON s.session_id = m.session_id
                WHERE s.session_id = ?
                GROUP BY s.session_id;
                """,
                (session_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "session_id": row["session_id"],
                "title": row["title"],
                "created_at": str(row["created_at"]),
                "updated_at": str(row["updated_at"]),
                "message_count": int(row["message_count"]),
            }

    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT s.session_id, s.title, s.created_at, s.updated_at,
                       COUNT(m.message_id) AS message_count
                FROM chat_sessions s
                LEFT JOIN chat_messages m ON s.session_id = m.session_id
                GROUP BY s.session_id
                ORDER BY s.updated_at DESC
                LIMIT ?;
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            return [
                {
                    "session_id": r["session_id"],
                    "title": r["title"],
                    "created_at": str(r["created_at"]),
                    "updated_at": str(r["updated_at"]),
                    "message_count": int(r["message_count"]),
                }
                for r in rows
            ]

    def delete_session(self, session_id: str) -> bool:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM chat_sessions WHERE session_id = ?;",
                (session_id,),
            )
            return cursor.rowcount > 0

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        citations: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Ensure parent session exists
        if not self.get_session(session_id):
            title = content[:40].strip() + ("..." if len(content) > 40 else "")
            self.create_session(session_id=session_id, title=title)

        mid = message_id or f"msg-{uuid.uuid4().hex[:10]}"
        now_str = datetime.utcnow().isoformat()
        citations_json = json.dumps(citations or [], ensure_ascii=False)
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO chat_messages (message_id, session_id, role, content, citations_json, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (mid, session_id, role, content, citations_json, metadata_json, now_str),
            )
            # Update session updated_at
            conn.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?;",
                (now_str, session_id),
            )

        return {
            "message_id": mid,
            "session_id": session_id,
            "role": role,
            "content": content,
            "citations": citations or [],
            "metadata": metadata or {},
            "created_at": now_str,
        }

    def get_messages(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY created_at ASC"
        params: List[Any] = [session_id]
        if limit:
            query += " LIMIT ?"
            params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            results = []
            for r in rows:
                try:
                    cits = json.loads(r["citations_json"])
                except Exception:
                    cits = []
                try:
                    meta = json.loads(r["metadata_json"])
                except Exception:
                    meta = {}
                results.append({
                    "message_id": r["message_id"],
                    "session_id": r["session_id"],
                    "role": r["role"],
                    "content": r["content"],
                    "citations": cits,
                    "metadata": meta,
                    "created_at": str(r["created_at"]),
                })
            return results

    def get_recent_history(self, session_id: str, limit: int = 10) -> List[Dict[str, str]]:
        """Retrieves bounded conversation turns for LLM prompt context."""
        messages = self.get_messages(session_id)
        if not messages:
            return []
        recent = messages[-limit:]
        return [{"role": m["role"], "content": m["content"]} for m in recent]


# Default singleton instance
chat_repository = ChatRepository()
