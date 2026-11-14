# SIFT — Backend API

Production-minded Python backend for **SIFT — Find what matters. Understand everything.**  
Powered by FastAPI, Pydantic, PyMuPDF, SQLite, and UV.

---

## Architecture

```
backend/
  app/
    api/
      routes/
        health.py       # Health check & system storage status
        documents.py    # Document upload, listing, get, rename, delete, and chunking
        search.py       # Semantic passage retrieval
        chat.py         # Conversational Q&A with verifiable citations
    core/
      config.py         # Pydantic BaseSettings & environment configuration
      logging.py        # Centralized structured logging
      errors.py         # Domain exceptions & JSON exception handlers
    models/
      document.py       # Document domain models & enums
    schemas/
      health.py         # Health check response schemas
      document.py       # Document request/response schemas
      search.py         # Search request/response schemas
      chat.py           # Chat request/response & citation schemas
    services/
      document_service.py # Ingestion, parsing, and document lifecycle service
      chunking_service.py # Structure-aware chunking orchestration service
      search_service.py   # Vector & lexical passage search service
      chat_service.py     # Grounded evidence & answer generation service
    repositories/
      document_repository.py # SQLite document metadata persistence
      chunk_repository.py    # SQLite chunk persistence & idempotent regeneration
    rag/
      ingestion/        # Modular PyMuPDF & Text extraction parsers
      chunking/         # Deterministic structure-aware chunking engine
        base.py         # BaseChunker abstract interface
        config.py       # Validated ChunkConfig (chunk_size, overlap, min_size)
        models.py       # Chunk domain models and API schemas
        semantic_chunker.py # StructureAwareChunker implementation
      embeddings/       # Dense vector embedding provider interface
      retrieval/        # Hybrid dense + BM25 retriever interface
      generation/       # Grounded reasoning & citation synthesis interface
    utils/
      file_utils.py     # Filename sanitization, SHA-256 hash, and validation helpers
    main.py             # FastAPI entrypoint, CORS & middleware
  tests/                # Comprehensive test suite (pytest)
  .env.example          # Environment variables template
  pyproject.toml        # UV project configuration and dependencies
```

---

## Phase 2: Document Ingestion Pipeline

### Supported File Types
- **PDF (`.pdf`)**: Page-by-page text and metadata extraction powered by PyMuPDF. Validates PDF magic header (`%PDF-`), rejects password-protected or corrupted PDFs, and enforces presence of extractable text.
- **TXT (`.txt`)**: Safe text decoding with multi-encoding fallback (`utf-8`, `utf-8-sig`, `latin-1`, `cp1252`, `utf-16`). Rejects empty or whitespace-only files.

### Persistence & Storage
- **Metadata Database**: Persistent local **SQLite** (`data/sift_metadata.db`) tracking `document_id`, `original_filename`, `stored_filename`, `file_type`, `file_size`, `upload_timestamp`, `processing_status` (`processing`, `indexed`, `failed`), `extracted_text`, `page_count`, `error_message`, and SHA-256 checksums.
- **Controlled Storage**: Raw files are preserved under `data/storage/` with sanitized names prefixed by `document_id` to prevent path traversal and accidental file overwrites.
- **Security & Privacy**: Physical filesystem paths are never exposed across public API schemas.

---

## Phase 3: Structure-Aware Chunking Engine

The chunking engine decomposes documents into coherent, sequential passages using a deterministic, structure-aware strategy:

1. **Paragraph Preservation**: Respects natural paragraph boundaries (`\n\n`), combining smaller paragraphs up to the configured `chunk_size`.
2. **Sentence Segmentation**: Oversized paragraphs are cleanly partitioned along sentence boundaries (`. ! ?`), preventing mid-sentence truncation.
3. **Graceful Sub-splitting**: Unusually long sentences are split along clause boundaries (`; : ,`) or word boundaries.
4. **Configurable Overlap**: Maintains bounded, trailing context from preceding chunks up to `chunk_overlap` characters.
5. **Exact Page Provenance**: For PDF files, tracks 1-based `page_start` and `page_end` coordinates across page transitions. For TXT files, page coordinates remain `null`.
6. **Idempotent Persistence**: Chunks are stored in the SQLite `chunks` table with unique stable IDs (`chk_{doc_id}_{index:04d}`). Re-generating chunks atomically replaces stale records.

### Default Chunk Configuration
- `chunk_size`: 1000 characters
- `chunk_overlap`: 150 characters
- `minimum_chunk_size`: 100 characters

---

## Phase 4: Local Dense Embedding Generation Engine

SIFT generates dense vector embeddings locally without any external API calls, billing dependencies, or risk of data leakage:

1. **Free Local Model**: Uses `sentence-transformers/all-MiniLM-L6-v2` generating 384-dimensional $L_2$-normalized float32 vectors.
2. **CPU Optimized**: Fully supported on CPU hardware without requiring CUDA/GPU accelerators.
3. **Lazy Loading & Instance Reuse**: The model is loaded lazily on first embedding inference and kept in memory across requests with thread-safe double-checked locking.
4. **Non-Blocking Thread Pool**: CPU-intensive vector encoding runs in worker threads via `asyncio.to_thread`, keeping FastAPI's async event loop completely responsive.
5. **Compact SQLite Binary Persistence**: High-density float32 vectors are serialized into binary BLOBs using `struct.pack(f"{len(vec)}f", *vec)` (1,536 bytes per chunk), saving storage space and avoiding JSON deserialization overhead.
6. **Automatic Pipeline Chaining**: Calling `/api/documents/{document_id}/embeddings` automatically triggers structure-aware chunking if the document has not yet been chunked.
7. **Idempotency & Version Awareness**: Checks existing embeddings and skips re-computation unless `force_recompute=true` or the active embedding model changes.

---

## Phase 5: Persistent ChromaDB Vector Store & Real Semantic Retrieval

SIFT replaces mock/heuristic search with genuine dense vector similarity retrieval powered by ChromaDB:

1. **Persistent Local Vector Store**: Uses `chromadb.PersistentClient` pointing to configurable local storage (`data/chroma`). Survives server restarts.
2. **Dedicated SIFT Collection**: Manages collection `sift_document_embeddings` configured with HNSW cosine distance (`"hnsw:space": "cosine"`). Lower distance indicates closer semantic alignment ($0.0$ is identical direction).
3. **Idempotent Indexing Pipeline**: Transfers validated Phase 4 384-dimensional dense vectors and chunk metadata into ChromaDB. Re-indexing safely clears obsolete vectors to prevent dangling chunks.
4. **Validation & Protection**: Enforces 1-to-1 chunk-to-embedding consistency, validates 384-d vector dimensionality and finite floats, rejects model/dimension mismatches, and prevents duplicate records.
5. **Real Semantic Search**: Encodes natural-language queries into 384-d vectors on CPU and queries ChromaDB with strict top-$k$ bounding (1–20) and optional `document_ids` scoping.
6. **Full Source Provenance**: Returns `document_id`, `chunk_id`, `chunk_index`, `content`, `source_filename`, `page_start`, `page_end`, `distance`, and `similarity_score`.
7. **Clean Removal of Mock Retrieval**: Mock search logic, hardcoded passages, and token heuristics are completely excised from the production search route.

---

## Phase 6: RAG Generation & Grounded Conversational Q&A

SIFT delivers context-grounded conversational reasoning anchored directly in the Phase 5 ChromaDB vector store:

1. **Pluggable LLM Provider Architecture**: Abstracted generation interface supporting free-tier providers like **Groq** (`llama-3.1-8b-instant`), extensible to local or alternative providers, alongside a deterministic `MockLLMProvider` for zero-cost offline CI/CD.
2. **Strict Grounding Guardrails**: Strict system prompt instructs the model to answer exclusively from retrieved excerpts, quote accurately, distinguish facts from interpretation, and refuse unsupported questions with a clean *"Based on the provided documents, there is insufficient evidence to answer your question."* response.
3. **Verifiable Machine-Readable Citations**: Every factual statement is backed by structured citations preserving `document_id`, `chunk_id`, `source_filename`, 1-based page bounds (`page_start`/`page_end`), passage excerpt, and ChromaDB cosine distance.
4. **Conversational Sessions & SQLite Persistence**: Multi-turn conversations are automatically tracked in SQLite (`chat_sessions` and `chat_messages` tables), preserving bounded history without polluting system prompts or allowing history to override grounding rules.
5. **Robust Error Handling**: Safely handles unconfigured providers (503), timeouts (504), rate limits (429), upstream API failures (502), missing sessions (404), and empty retrieval contexts without leaking secret keys or internal paths.

### LLM Provider Configuration (Groq)

SIFT integrates with Groq's free-tier API. Configure your environment in `backend/.env`:

```env
LLM_PROVIDER="groq"
GROQ_API_KEY="your-groq-api-key-here"
GROQ_MODEL="llama-3.1-8b-instant"
GROQ_API_URL="https://api.groq.com/openai/v1/chat/completions"
LLM_MAX_TOKENS=1024
LLM_TEMPERATURE=0.1
CHAT_HISTORY_LIMIT=10
```

> **Note on Free-Tier Quotas:** Groq's developer tier offers free inference within rate limits (e.g. 30 requests/minute). When unconfigured, SIFT returns clear actionable HTTP 503 errors and runs offline tests seamlessly with `MockLLMProvider`.

---

## Setup & Running with UV

### Prerequisites
- Python 3.11+
- `uv` (Fast Python package manager)

### 1. Install Dependencies
```bash
cd backend
uv sync
```

### 2. Run the Development Server
```bash
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
The API server runs at `http://127.0.0.1:8000`.
- **Interactive OpenAPI Docs**: `http://127.0.0.1:8000/docs`
- **ReDoc**: `http://127.0.0.1:8000/redoc`
- **Health Endpoint**: `http://127.0.0.1:8000/api/health`

### 3. Run Automated Tests
```bash
uv run pytest -v
```

---

## API Routes Overview

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/documents/upload` | Upload and parse document file (`.pdf`, `.txt`) |
| `GET` | `/api/documents` | List indexed documents with optional `tag`, `status`, `search` |
| `GET` | `/api/documents/{document_id}` | Retrieve document details and full extracted text by ID |
| `PATCH` | `/api/documents/{document_id}/rename` | Rename a document display name |
| `DELETE` | `/api/documents/{document_id}` | Safely remove database record, chunks, embeddings, vector store records, and physical file |
| `POST` | `/api/documents/{document_id}/chunks` | Idempotently generate and persist structure-aware chunks |
| `GET` | `/api/documents/{document_id}/chunks` | Retrieve ordered chunks for a document (auto-generates if needed) |
| `POST` | `/api/documents/{document_id}/embeddings` | Generate & persist 384-d dense embeddings for document chunks |
| `GET` | `/api/documents/{document_id}/embeddings` | Retrieve embedding summary metadata for a document |
| `POST` | `/api/documents/{document_id}/index` | Index persisted document chunks & embeddings into ChromaDB |
| `GET` | `/api/documents/{document_id}/index-status` | Retrieve vector indexing status and chunk counts from ChromaDB |
| `POST` | `/api/search` | Real semantic dense vector passage search via ChromaDB |
| `POST` | `/api/chat` | RAG conversational Q&A with ChromaDB retrieval & verifiable citations |
| `GET` | `/api/chat/sessions` | List active conversational sessions |
| `GET` | `/api/chat/sessions/{session_id}` | Retrieve conversation turns and citations for a session |
| `DELETE` | `/api/chat/sessions/{session_id}` | Permanently delete a conversation session and its message history |
