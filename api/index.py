"""
Vercel Serverless Function entrypoint for SIFT FastAPI backend.
Exposes the existing FastAPI application for Vercel's Python runtime.
"""
import logging
import sys
from pathlib import Path

# Resolve repository root and backend directory
ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"

# Ensure backend directory is in sys.path so app modules resolve cleanly
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

logger = logging.getLogger("sift.vercel")


def correct_vercel_path(scope: dict) -> str:
    """
    Restores the original request path from Vercel/Next.js rewrite headers if present.
    When Next.js or Vercel rewrites /api/:path* to /api/index.py, the original requested URI
    is supplied in 'x-forwarded-uri', 'x-matched-path', 'x-nextjs-rewritten-path',
    or 'x-invoke-path'. This restores the original path so FastAPI routes match as expected.
    """
    if scope.get("type") == "http":
        headers = dict(scope.get("headers", []))
        raw_forwarded = (
            headers.get(b"x-forwarded-uri")
            or headers.get(b"x-matched-path")
            or headers.get(b"x-nextjs-rewritten-path")
            or headers.get(b"x-invoke-path")
            or headers.get(b"x-original-uri")
        )
        if raw_forwarded:
            forwarded_path = raw_forwarded.decode("utf-8", errors="ignore").split("?")[0]
            if forwarded_path and not forwarded_path.endswith("index.py"):
                scope["path"] = forwarded_path
        # Normalize /api/index.py fallback to /api
        if scope.get("path") in ("/api/index.py", "api/index.py"):
            scope["path"] = "/api"
    return scope.get("path", "")


class VercelPathCorrectionMiddleware:
    """ASGI middleware ensuring Vercel rewrite paths are preserved."""

    def __init__(self, asgi_app):
        self.asgi_app = asgi_app

    async def __call__(self, scope, receive, send):
        correct_vercel_path(scope)
        await self.asgi_app(scope, receive, send)


# Import and expose the existing FastAPI application instance without duplicating initialization
try:
    from app.main import app
    from app.core.config import settings

    # In production/serverless execution, validate required hosted cloud providers
    settings.validate_production_configuration()

    app.add_middleware(VercelPathCorrectionMiddleware)

except Exception:
    # Safe server-side logging without leaking secrets or paths in HTTP responses
    logger.exception("SIFT Backend Serverless Function failed to initialize during startup")
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    app = FastAPI(title="SIFT Backend Initialization Error")

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
    def initialization_error(path: str):
        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "message": "Internal server error: backend service initialization failed.",
            },
        )
