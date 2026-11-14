import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, List, Optional
from app.core.config import settings
from app.core.logging import logger
from app.models.document import Document, DocumentTag, DocumentType, ProcessingStatus


class DocumentRepository:
    """SQLite-backed persistence repository for document metadata and extraction records."""

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = db_path or settings.database_path
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        try:
            # Enable WAL mode and foreign keys
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initializes tables and indexes if they do not exist."""
        with self._get_connection() as conn:
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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(processing_status);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_tag ON documents(tag);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at DESC);")
        logger.info(f"Initialized SQLite document repository at: {self._db_path}")

    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> Document:
        return Document(
            id=row["document_id"],
            original_filename=row["original_filename"],
            stored_filename=row["stored_filename"],
            name=row["name"],
            file_type=DocumentType(row["file_type"]),
            file_size=row["file_size"],
            size=row["size_human"],
            upload_timestamp=row["upload_timestamp"],
            processing_status=ProcessingStatus(row["processing_status"]),
            extracted_text=row["extracted_text"] or "",
            page_count=row["page_count"],
            error_message=row["error_message"],
            hash=row["file_hash"] or "",
            tag=DocumentTag(row["tag"]) if row["tag"] in DocumentTag._value2member_map_ else DocumentTag.TECHNICAL,
            author=row["author"],
            description=row["description"] or "",
            raw_text_preview=row["raw_text_preview"],
            chunks_count=row["chunks_count"] or 0,
            vector_count=row["vector_count"] or 0,
            token_count=row["token_count"] or 0,
            file_path=row["file_path"],
        )

    def create(self, doc: Document) -> Document:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    document_id, original_filename, stored_filename, name,
                    file_type, file_size, size_human, upload_timestamp,
                    processing_status, extracted_text, page_count, error_message,
                    file_hash, tag, author, description, raw_text_preview,
                    chunks_count, vector_count, token_count, file_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    doc.id,
                    doc.original_filename,
                    doc.stored_filename,
                    doc.name,
                    doc.file_type.value,
                    doc.file_size,
                    doc.size,
                    doc.upload_timestamp,
                    doc.processing_status.value,
                    doc.extracted_text,
                    doc.page_count,
                    doc.error_message,
                    doc.hash,
                    doc.tag.value,
                    doc.author,
                    doc.description,
                    doc.raw_text_preview,
                    doc.chunks_count,
                    doc.vector_count,
                    doc.token_count,
                    doc.file_path,
                ),
            )
        logger.debug(f"Inserted document record: {doc.id}")
        return doc

    def get(self, document_id: str) -> Optional[Document]:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM documents WHERE document_id = ? LIMIT 1;",
                (document_id,),
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_document(row)
        return None

    def list_all(
        self,
        tag: Optional[str] = None,
        status: Optional[str] = None,
        search: Optional[str] = None,
    ) -> List[Document]:
        query = "SELECT * FROM documents WHERE 1=1"
        params: List[str] = []

        if tag and tag.lower() != "all":
            query += " AND LOWER(tag) = ?"
            params.append(tag.lower())

        if status and status.lower() != "all":
            query += " AND LOWER(processing_status) = ?"
            params.append(status.lower())

        if search:
            query += " AND (LOWER(name) LIKE ? OR LOWER(description) LIKE ? OR LOWER(extracted_text) LIKE ?)"
            term = f"%{search.lower().strip()}%"
            params.extend([term, term, term])

        query += " ORDER BY created_at DESC;"

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            return [self._row_to_document(row) for row in rows]

    def update(self, document_id: str, **kwargs) -> Optional[Document]:
        allowed_columns = {
            "name",
            "processing_status",
            "extracted_text",
            "page_count",
            "error_message",
            "tag",
            "description",
            "raw_text_preview",
            "chunks_count",
            "vector_count",
            "token_count",
        }
        updates = []
        params = []
        for key, val in kwargs.items():
            if key in allowed_columns:
                updates.append(f"{key} = ?")
                params.append(val.value if hasattr(val, "value") else val)

        if not updates:
            return self.get(document_id)

        params.append(document_id)
        with self._get_connection() as conn:
            conn.execute(
                f"UPDATE documents SET {', '.join(updates)} WHERE document_id = ?;",
                params,
            )
        return self.get(document_id)

    def delete(self, document_id: str) -> bool:
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM documents WHERE document_id = ?;",
                (document_id,),
            )
            return cursor.rowcount > 0

    def count(self) -> int:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM documents;")
            return cursor.fetchone()[0]


# Default singleton instance
document_repository = DocumentRepository()
