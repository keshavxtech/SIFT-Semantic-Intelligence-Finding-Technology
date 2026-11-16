import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, List, Optional
from app.core.config import settings
from app.core.database import get_db_connection
from app.core.logging import logger
from app.rag.chunking.models import Chunk


class ChunkRepository:
    """Persistence repository for document chunks."""

    def __init__(self, db_path: Optional[Path] = None, provider: Optional[str] = None):
        self._db_path = db_path or settings.database_path
        self._provider = provider or ("sqlite" if (db_path and db_path != settings.database_path) else settings.DATABASE_PROVIDER)
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Generator[Any, None, None]:
        with get_db_connection(db_path=self._db_path, provider=self._provider) as conn:
            yield conn

    def _init_db(self) -> None:
        """Creates chunks table and indexes if they do not exist."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    character_count INTEGER NOT NULL,
                    token_count INTEGER NOT NULL,
                    page_start INTEGER,
                    page_end INTEGER,
                    metadata_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(document_id) REFERENCES documents(document_id) ON DELETE CASCADE
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_index ON chunks(document_id, chunk_index);")
        logger.info(f"Initialized chunk repository (provider: {self._provider}) at: {self._db_path}")

    @staticmethod
    def _row_to_chunk(row: Any) -> Chunk:
        try:
            metadata = json.loads(row["metadata_json"])
        except (json.JSONDecodeError, TypeError):
            metadata = {}

        return Chunk(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            chunk_index=row["chunk_index"],
            content=row["content"],
            character_count=row["character_count"],
            token_count=row["token_count"],
            page_start=row["page_start"],
            page_end=row["page_end"],
            metadata=metadata,
        )

    def save_chunks(self, document_id: str, chunks: List[Chunk]) -> List[Chunk]:
        """
        Idempotently persists chunks for a document.
        Replaces any existing chunks in an atomic transaction.
        """
        with self._get_connection() as conn:
            # 1. Delete existing chunks for this document
            conn.execute("DELETE FROM chunks WHERE document_id = ?;", (document_id,))

            # 2. Bulk insert new chunks
            records = [
                (
                    c.chunk_id,
                    c.document_id,
                    c.chunk_index,
                    c.content,
                    c.character_count,
                    c.token_count,
                    c.page_start,
                    c.page_end,
                    json.dumps(c.metadata, ensure_ascii=False),
                )
                for c in chunks
            ]
            conn.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, document_id, chunk_index, content,
                    character_count, token_count, page_start, page_end, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                records,
            )

            # 3. Update documents table chunks_count
            conn.execute(
                "UPDATE documents SET chunks_count = ? WHERE document_id = ?;",
                (len(chunks), document_id),
            )

        logger.info(f"Persisted {len(chunks)} chunks for document: {document_id}")
        return chunks

    def get_chunks_by_document(self, document_id: str) -> List[Chunk]:
        """Retrieves all chunks for a document ordered sequentially."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index ASC;",
                (document_id,),
            )
            rows = cursor.fetchall()
            return [self._row_to_chunk(row) for row in rows]

    def delete_chunks_by_document(self, document_id: str) -> int:
        """Deletes all chunks belonging to a document."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM chunks WHERE document_id = ?;", (document_id,))
            return cursor.rowcount

    def count(self) -> int:
        """Returns total count of chunks in the database."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM chunks;")
            return cursor.fetchone()[0]


# Default singleton instance
chunk_repository = ChunkRepository()
