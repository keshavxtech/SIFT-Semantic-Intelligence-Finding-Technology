import sqlite3
import struct
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, List, Optional
from app.core.config import settings
from app.core.logging import logger
from app.rag.embeddings.models import ChunkEmbedding


class EmbeddingRepository:
    """SQLite-backed persistence repository for local dense vector embeddings."""

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = db_path or settings.database_path
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initializes embeddings table and indexes if they do not exist."""
        with self._get_connection() as conn:
            # Ensure documents table exists for foreign key constraints
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    original_filename TEXT NOT NULL,
                    stored_filename TEXT NOT NULL,
                    name TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    size_human TEXT NOT NULL,
                    upload_timestamp TEXT NOT NULL,
                    processing_status TEXT NOT NULL,
                    extracted_text TEXT NOT NULL,
                    page_count INTEGER NOT NULL DEFAULT 1,
                    error_message TEXT,
                    file_hash TEXT,
                    tag TEXT DEFAULT 'Technical',
                    author TEXT DEFAULT 'Workspace User',
                    description TEXT DEFAULT '',
                    raw_text_preview TEXT,
                    chunks_count INTEGER DEFAULT 0,
                    vector_count INTEGER DEFAULT 0,
                    token_count INTEGER DEFAULT 0,
                    file_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    model_name TEXT NOT NULL,
                    embedding_dimension INTEGER NOT NULL,
                    embedding_bytes BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(document_id) REFERENCES documents(document_id) ON DELETE CASCADE
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_document_id ON embeddings(document_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model_name);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_doc_idx ON embeddings(document_id, chunk_index);")
        logger.info(f"Initialized SQLite embedding repository at: {self._db_path}")

    @staticmethod
    def _serialize_vector(vector: List[float]) -> bytes:
        """Packs float32 vector coordinates into a compact binary blob."""
        return struct.pack(f"{len(vector)}f", *vector)

    @staticmethod
    def _deserialize_vector(blob: bytes, dimension: int) -> List[float]:
        """Unpacks float32 binary blob into a python float list."""
        if not blob:
            return []
        count = len(blob) // 4
        return list(struct.unpack(f"{count}f", blob))

    def save_embeddings(self, document_id: str, embeddings: List[ChunkEmbedding]) -> List[ChunkEmbedding]:
        """
        Idempotently persists embeddings for a document in an atomic transaction.
        Replaces any existing vectors for this document.
        """
        with self._get_connection() as conn:
            # 1. Clear previous embeddings for this document
            conn.execute("DELETE FROM embeddings WHERE document_id = ?;", (document_id,))

            # 2. Insert new binary serialized vectors
            records = [
                (
                    e.chunk_id,
                    e.document_id,
                    e.chunk_index,
                    e.model_name,
                    e.embedding_dimension,
                    self._serialize_vector(e.vector),
                )
                for e in embeddings
            ]
            conn.executemany(
                """
                INSERT INTO embeddings (
                    chunk_id, document_id, chunk_index, model_name, embedding_dimension, embedding_bytes
                ) VALUES (?, ?, ?, ?, ?, ?);
                """,
                records,
            )

            # 3. Update documents table vector_count
            conn.execute(
                "UPDATE documents SET vector_count = ? WHERE document_id = ?;",
                (len(embeddings), document_id),
            )

        logger.info(
            f"Persisted {len(embeddings)} embeddings for document '{document_id}' "
            f"using model '{embeddings[0].model_name if embeddings else 'N/A'}'"
        )
        return embeddings

    def get_embeddings_by_document(
        self,
        document_id: str,
        include_vectors: bool = False,
    ) -> List[ChunkEmbedding]:
        """Retrieves embeddings for a document. Omits vector bytes by default for performance."""
        with self._get_connection() as conn:
            if include_vectors:
                query = "SELECT * FROM embeddings WHERE document_id = ? ORDER BY chunk_index ASC;"
            else:
                query = (
                    "SELECT chunk_id, document_id, chunk_index, model_name, embedding_dimension, created_at "
                    "FROM embeddings WHERE document_id = ? ORDER BY chunk_index ASC;"
                )
            cursor = conn.execute(query, (document_id,))
            rows = cursor.fetchall()

            result = []
            for r in rows:
                vector = self._deserialize_vector(r["embedding_bytes"], r["embedding_dimension"]) if include_vectors else []
                result.append(
                    ChunkEmbedding(
                        chunk_id=r["chunk_id"],
                        document_id=r["document_id"],
                        chunk_index=r["chunk_index"],
                        model_name=r["model_name"],
                        embedding_dimension=r["embedding_dimension"],
                        vector=vector,
                    )
                )
            return result

    def get_model_name_for_document(self, document_id: str) -> Optional[str]:
        """Returns the model name used for the document's existing embeddings, if any."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT model_name FROM embeddings WHERE document_id = ? LIMIT 1;",
                (document_id,),
            )
            row = cursor.fetchone()
            return row["model_name"] if row else None

    def get_count(self, document_id: Optional[str] = None) -> int:
        """Returns count of embeddings for a document or across the entire database."""
        with self._get_connection() as conn:
            if document_id:
                cursor = conn.execute("SELECT COUNT(*) FROM embeddings WHERE document_id = ?;", (document_id,))
            else:
                cursor = conn.execute("SELECT COUNT(*) FROM embeddings;")
            return cursor.fetchone()[0]

    def delete_embeddings_by_document(self, document_id: str) -> int:
        """Deletes all embeddings associated with a document."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM embeddings WHERE document_id = ?;", (document_id,))
            return cursor.rowcount


# Default singleton instance
embedding_repository = EmbeddingRepository()
