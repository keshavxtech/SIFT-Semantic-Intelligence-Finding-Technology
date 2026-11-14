import shutil
from fastapi import APIRouter
from app.core.config import settings
from app.repositories.document_repository import document_repository
from app.schemas.health import HealthResponse

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=HealthResponse)
async def check_health() -> HealthResponse:
    storage_path = settings.storage_path
    total_space, used_space, free_space = shutil.disk_usage(storage_path)

    return HealthResponse(
        status="healthy",
        service=settings.PROJECT_NAME,
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
        documents_indexed=document_repository.count(),
        storage={
            "storage_dir": str(storage_path),
            "free_space_mb": round(free_space / (1024 * 1024), 1),
            "total_space_gb": round(total_space / (1024 * 1024 * 1024), 2),
        },
    )
