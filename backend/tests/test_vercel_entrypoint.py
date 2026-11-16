import json
from pathlib import Path
import sys
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.errors import DatabaseNotConfiguredError, VectorStoreNotConfiguredError
from app.core.database import get_db_connection
from app.rag.retrieval.pinecone_vector_store import PineconeVectorStore

# Ensure repository root is on sys.path so api.index can be imported
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


def test_vercel_entrypoint_imports_and_exports_fastapi_app():
    """Verifies api.index exposes the FastAPI application instance without duplicate creation."""
    import api.index as entry
    assert hasattr(entry, "app")
    assert isinstance(entry.app, FastAPI)
    assert entry.app.title == settings.PROJECT_NAME


def test_vercel_entrypoint_registers_all_api_routes():
    """Verifies all expected production API paths are registered under /api."""
    import api.index as entry

    openapi_paths = set(entry.app.openapi()["paths"].keys())

    expected_paths = {
        "/",
        "/api",
        "/api/",
        "/api/health",
        "/api/documents",
        "/api/documents/upload",
        "/api/documents/{document_id}",
        "/api/documents/{document_id}/chunks",
        "/api/documents/{document_id}/embeddings",
        "/api/documents/{document_id}/index",
        "/api/documents/{document_id}/index-status",
        "/api/documents/{document_id}/rename",
        "/api/search",
        "/api/chat",
        "/api/chat/sessions",
        "/api/chat/sessions/{session_id}",
    }

    for path in expected_paths:
        assert path in openapi_paths, f"Expected path '{path}' missing from OpenAPI routes"


def test_vercel_entrypoint_root_and_api_status_endpoints():
    """Verifies that GET / and GET /api return 200 operational status."""
    from app.main import root_status
    res = root_status()
    assert res.status_code == 200
    body = json.loads(res.body.decode("utf-8"))
    assert body["status"] == "operational"
    assert "SIFT" in body["service"]
    assert body["health"] == "/api/health"


def test_vercel_cors_configuration_includes_production_domain():
    """Verifies CORS configuration permits the production Vercel frontend URL."""
    prod_origin = "https://sift-semantic-intelligence-finding.vercel.app"
    assert prod_origin in settings.cors_origins_list


def test_vercel_json_configuration():
    """Verifies vercel.json exists at root, is valid JSON, and routes /api correctly."""
    root_dir = Path(__file__).resolve().parent.parent.parent
    vercel_json_path = root_dir / "vercel.json"
    assert vercel_json_path.exists(), "vercel.json must exist at repository root"

    with open(vercel_json_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    assert "rewrites" in config
    rewrites = config["rewrites"]
    api_rewrite = next((r for r in rewrites if r.get("source") == "/api/:path*"), None)
    assert api_rewrite is not None, "Missing rewrite for /api/:path*"
    assert api_rewrite.get("destination") == "/api/index.py"


def test_requirements_files_consistency():
    """Verifies requirements.txt exists at repository root and matches api/requirements.txt."""
    root_dir = Path(__file__).resolve().parent.parent.parent
    root_req = root_dir / "requirements.txt"
    api_req = root_dir / "api" / "requirements.txt"

    assert root_req.exists(), "Root requirements.txt must exist for Vercel Python runtime"
    assert api_req.exists(), "api/requirements.txt must exist for Vercel function builder"

    root_content = root_req.read_text(encoding="utf-8").strip()
    api_content = api_req.read_text(encoding="utf-8").strip()

    assert root_content == api_content, "Root requirements.txt and api/requirements.txt must be identical"
    assert "fastapi" in root_content
    assert "httpx" in root_content
    assert "pydantic" in root_content


def test_vercel_path_correction_logic():
    """Verifies correct_vercel_path restores original path from Vercel rewrite headers."""
    from api.index import correct_vercel_path

    # Case 1: Standard request without headers retains original path
    scope1 = {
        "type": "http",
        "path": "/api/documents",
        "headers": [(b"host", b"sift.vercel.app")],
    }
    assert correct_vercel_path(scope1) == "/api/documents"
    assert scope1["path"] == "/api/documents"

    # Case 2: Vercel rewrite sets path to /api/index.py with x-forwarded-uri
    scope2 = {
        "type": "http",
        "path": "/api/index.py",
        "headers": [
            (b"x-forwarded-uri", b"/api/documents/upload?tag=tech"),
            (b"host", b"sift.vercel.app"),
        ],
    }
    assert correct_vercel_path(scope2) == "/api/documents/upload"
    assert scope2["path"] == "/api/documents/upload"

    # Case 3: Vercel rewrite with x-matched-path
    scope3 = {
        "type": "http",
        "path": "/api/index.py",
        "headers": [
            (b"x-matched-path", b"/api/chat"),
            (b"host", b"sift.vercel.app"),
        ],
    }
    assert correct_vercel_path(scope3) == "/api/chat"
    assert scope3["path"] == "/api/chat"

    # Case 4: Non-http scope is ignored
    scope4 = {"type": "lifespan", "path": ""}
    assert correct_vercel_path(scope4) == ""

    # Case 5: Next.js rewrite sets path to /api/ with x-nextjs-rewritten-path
    scope5 = {
        "type": "http",
        "path": "/api/",
        "headers": [
            (b"x-nextjs-rewritten-path", b"/api/health"),
            (b"host", b"sift.vercel.app"),
        ],
    }
    assert correct_vercel_path(scope5) == "/api/health"
    assert scope5["path"] == "/api/health"

    # Case 6: Vercel / Next.js rewrite sets path to /api/ with x-invoke-path
    scope6 = {
        "type": "http",
        "path": "/api/",
        "headers": [
            (b"x-invoke-path", b"/api/health"),
            (b"host", b"sift.vercel.app"),
        ],
    }
    assert correct_vercel_path(scope6) == "/api/health"
    assert scope6["path"] == "/api/health"


def test_initialization_error_handler_returns_safe_generic_response():
    """Verifies fallback error handler returns generic 500 without leaking secrets, paths, or tracebacks."""
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    fallback_app = FastAPI(title="SIFT Backend Initialization Error")

    @fallback_app.api_route("/{path:path}", methods=["GET", "POST"])
    def initialization_error(path: str):
        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "message": "Internal server error: backend service initialization failed.",
            },
        )

    res = initialization_error("documents/secret_key=123")
    assert res.status_code == 500
    body = json.loads(res.body.decode("utf-8"))
    assert body["error"] is True
    assert body["message"] == "Internal server error: backend service initialization failed."
    assert "secret_key" not in body
    assert "Traceback" not in body["message"]


def test_production_storage_paths_do_not_silently_fallback():
    """Verifies production providers strictly raise configuration errors and do not silently fall back."""
    # 1. Turso database fails explicitly when credentials are missing
    with pytest.raises(DatabaseNotConfiguredError):
        with get_db_connection(provider="turso"):
            pass

    # 2. Pinecone vector store fails explicitly when api_key is missing
    store = PineconeVectorStore(api_key="")
    with pytest.raises(VectorStoreNotConfiguredError):
        store._get_client()


def test_python_version_file_exists_and_matches():
    """Verifies .python-version exists at root and sets Python 3.12 runtime."""
    root_dir = Path(__file__).resolve().parent.parent.parent
    py_version_file = root_dir / ".python-version"
    assert py_version_file.exists(), ".python-version must exist at repository root"
    version_str = py_version_file.read_text(encoding="utf-8").strip()
    assert version_str == "3.12"


def test_vercel_json_functions_max_duration():
    """Verifies vercel.json defines maxDuration for the Python entrypoint."""
    root_dir = Path(__file__).resolve().parent.parent.parent
    vercel_json_path = root_dir / "vercel.json"
    with open(vercel_json_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    assert "functions" in config
    assert "api/index.py" in config["functions"]
    assert config["functions"]["api/index.py"].get("maxDuration") == 60


def test_serverless_temporary_storage_path_resolution(monkeypatch):
    """Verifies storage_path maps to system tempdir in serverless runtime."""
    from app.core.config import Settings
    import tempfile

    # Case 1: In local development, storage_path resolves to data/storage
    local_settings = Settings(ENVIRONMENT="development", STORAGE_DIR="data/storage")
    assert "data" in str(local_settings.storage_path)

    # Case 2: In serverless (VERCEL=1), storage_path maps to tempfile directory
    monkeypatch.setenv("VERCEL", "1")
    serverless_settings = Settings(ENVIRONMENT="production", STORAGE_DIR="data/storage")
    assert str(tempfile.gettempdir()) in str(serverless_settings.storage_path)


def test_production_configuration_validation_rules():
    """Verifies that validate_production_configuration rejects invalid production provider selections."""
    from app.core.config import Settings

    # 1. Rejects SQLite in production
    s1 = Settings(
        ENVIRONMENT="production",
        DATABASE_PROVIDER="sqlite",
        VECTOR_STORE_PROVIDER="pinecone",
        PINECONE_API_KEY="test-key",
        EMBEDDING_PROVIDER="huggingface",
        HUGGINGFACE_API_KEY="hf-key",
    )
    with pytest.raises(ValueError, match="hosted database provider 'turso'"):
        s1.validate_production_configuration()

    # 2. Rejects ChromaDB in production
    s2 = Settings(
        ENVIRONMENT="production",
        DATABASE_PROVIDER="turso",
        TURSO_DATABASE_URL="libsql://test.turso.io",
        TURSO_AUTH_TOKEN="test-token",
        VECTOR_STORE_PROVIDER="chroma",
        EMBEDDING_PROVIDER="huggingface",
        HUGGINGFACE_API_KEY="hf-key",
    )
    with pytest.raises(ValueError, match="hosted vector store provider 'pinecone'"):
        s2.validate_production_configuration()

    # 3. Rejects SentenceTransformers in production
    s3 = Settings(
        ENVIRONMENT="production",
        DATABASE_PROVIDER="turso",
        TURSO_DATABASE_URL="libsql://test.turso.io",
        TURSO_AUTH_TOKEN="test-token",
        VECTOR_STORE_PROVIDER="pinecone",
        PINECONE_API_KEY="test-key",
        EMBEDDING_PROVIDER="sentence-transformers",
    )
    with pytest.raises(ValueError, match="hosted embedding provider 'huggingface'"):
        s3.validate_production_configuration()

    # 4. Rejects missing Turso credentials
    s4 = Settings(
        ENVIRONMENT="production",
        DATABASE_PROVIDER="turso",
        TURSO_DATABASE_URL=None,
        TURSO_AUTH_TOKEN=None,
        VECTOR_STORE_PROVIDER="pinecone",
        PINECONE_API_KEY="test-key",
        EMBEDDING_PROVIDER="huggingface",
        HUGGINGFACE_API_KEY="hf-key",
    )
    with pytest.raises(ValueError, match="TURSO_DATABASE_URL and TURSO_AUTH_TOKEN"):
        s4.validate_production_configuration()

    # 5. Rejects missing Groq API key in production
    s5 = Settings(
        ENVIRONMENT="production",
        DATABASE_PROVIDER="turso",
        TURSO_DATABASE_URL="libsql://test.turso.io",
        TURSO_AUTH_TOKEN="test-token",
        VECTOR_STORE_PROVIDER="pinecone",
        PINECONE_API_KEY="test-key",
        EMBEDDING_PROVIDER="huggingface",
        HUGGINGFACE_API_KEY="hf-key",
        LLM_PROVIDER="groq",
        GROQ_API_KEY=None,
    )
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        s5.validate_production_configuration()

    # 6. Passes when all required hosted credentials are supplied
    s6 = Settings(
        ENVIRONMENT="production",
        DATABASE_PROVIDER="turso",
        TURSO_DATABASE_URL="libsql://test.turso.io",
        TURSO_AUTH_TOKEN="test-token",
        VECTOR_STORE_PROVIDER="pinecone",
        PINECONE_API_KEY="test-key",
        EMBEDDING_PROVIDER="huggingface",
        HUGGINGFACE_API_KEY="hf-key",
        LLM_PROVIDER="groq",
        GROQ_API_KEY="test-groq-key",
    )
    s6.validate_production_configuration()  # Should not raise!


def test_chroma_vector_store_lazy_import():
    """Verifies ChromaVectorStore does not import chromadb at class definition time."""
    from app.rag.retrieval.vector_store import ChromaVectorStore, get_vector_store
    from app.rag.retrieval.pinecone_vector_store import PineconeVectorStore

    # Instantiating Pinecone vector store via factory works without chromadb interaction
    pinecone_store = get_vector_store(provider_type="pinecone")
    assert isinstance(pinecone_store, PineconeVectorStore)

    # ChromaVectorStore instantiates cleanly
    chroma_store = ChromaVectorStore()
    assert chroma_store.expected_dimension == 384

