"""Health check endpoint."""

from fastapi import APIRouter

from backend.cache.redis_cache import redis_cache
from backend.rag.embedding_service import embedding_service
from backend.rag.vector_store import vector_store
from backend.config import settings

router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Check health of all dependencies.
    Returns status of Redis, Qdrant, embedding model, and LLM config.
    """
    health = {
        "status": "healthy",
        "services": {},
    }

    # Redis
    health["services"]["redis"] = {
        "connected": redis_cache.is_connected,
        "url": settings.redis_url,
    }

    # Qdrant
    try:
        if vector_store._client:
            collections = vector_store.client.get_collections()
            health["services"]["qdrant"] = {
                "connected": True,
                "collections": len(collections.collections),
                "url": settings.qdrant_url,
            }
        else:
            health["services"]["qdrant"] = {"connected": False}
    except Exception as e:
        health["services"]["qdrant"] = {"connected": False, "error": str(e)}

    # Embedding model
    health["services"]["embedding"] = {
        "loaded": embedding_service._model is not None,
        "model": settings.embedding_model,
        "dimension": embedding_service.dimension,
    }

    # LLM config
    health["services"]["llm"] = {
        "configured": bool(settings.groq_api_key),
        "model": settings.llm_model,
        "provider": "groq",
    }

    # Overall status
    critical_services = [
        redis_cache.is_connected,
        embedding_service._model is not None,
    ]
    if not all(critical_services):
        health["status"] = "degraded"

    return health
