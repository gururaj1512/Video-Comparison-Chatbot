"""
ARQ worker entry point.

Run with: python -m backend.workers.worker

This starts a separate process that picks up jobs from the Redis queue
and executes them. The FastAPI server enqueues jobs; this worker runs them.
"""

from arq import create_pool
from arq.connections import RedisSettings

from backend.config import settings
from backend.workers.tasks import process_videos_task
from backend.rag.embedding_service import embedding_service
from backend.rag.vector_store import vector_store
from backend.cache.redis_cache import redis_cache
from backend.utils.logger import get_logger

logger = get_logger(__name__)


async def startup(ctx: dict) -> None:
    """Worker startup — initialize shared resources."""
    logger.info("ARQ worker starting up...")

    # Connect to Redis
    await redis_cache.connect()

    # Load embedding model
    embedding_service.load_model()

    # Connect to Qdrant
    vector_store.connect()

    logger.info("ARQ worker ready")


async def shutdown(ctx: dict) -> None:
    """Worker shutdown — cleanup."""
    logger.info("ARQ worker shutting down...")
    await redis_cache.disconnect()


def _parse_redis_url(url: str) -> RedisSettings:
    """Parse Redis URL into ARQ RedisSettings."""
    # redis://localhost:6379/0 → host, port, database
    from urllib.parse import urlparse
    parsed = urlparse(url)
    return RedisSettings(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        database=int(parsed.path.lstrip("/") or 0),
        password=parsed.password,
    )


class WorkerSettings:
    """ARQ worker configuration."""

    functions = [process_videos_task]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = _parse_redis_url(settings.redis_url)

    # Worker tuning
    max_jobs = 5               # Max concurrent jobs
    job_timeout = 300          # 5 minutes max per job
    keep_result = 3600         # Keep results for 1 hour
    health_check_interval = 30  # Health check every 30s
