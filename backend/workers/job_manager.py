"""
Background job manager using Redis for job status tracking.

Why Redis-based job tracking instead of ARQ's built-in results:
- More control over status lifecycle (pending → processing → completed/failed)
- Custom result storage with structured metadata
- Frontend can poll a simple Redis key without ARQ internals
- Works even if ARQ worker restarts

Why background jobs for video processing:
- Transcript extraction can take 30-60 seconds (Whisper, yt-dlp download)
- HTTP requests timeout at 30s by default
- Frontend needs to show progress without keeping connection open
- Concurrent processing is managed by the worker, not the API process
"""

import json
import uuid
import time
from typing import Dict, Optional

from backend.cache.redis_cache import redis_cache, RedisCache
from backend.utils.logger import get_logger
from backend.utils.constants import (
    REDIS_PREFIX_JOB,
    JOB_STATUS_PENDING,
    JOB_STATUS_PROCESSING,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
)

logger = get_logger(__name__)


class JobManager:
    """Manage background job lifecycle via Redis."""

    def _status_key(self, job_id: str) -> str:
        return RedisCache.make_key(REDIS_PREFIX_JOB, job_id, "status")

    def _result_key(self, job_id: str) -> str:
        return RedisCache.make_key(REDIS_PREFIX_JOB, job_id, "result")

    async def create_job(self, session_id: str) -> str:
        """
        Create a new job entry in Redis.

        Returns:
            job_id: Unique job identifier
        """
        job_id = str(uuid.uuid4())[:12]

        job_data = {
            "job_id": job_id,
            "session_id": session_id,
            "status": JOB_STATUS_PENDING,
            "created_at": time.time(),
            "updated_at": time.time(),
        }

        await redis_cache.set(
            self._status_key(job_id),
            job_data,
            ttl=3600,  # Jobs expire after 1 hour
        )

        logger.info("Created job %s for session %s", job_id, session_id)
        return job_id

    async def update_status(
        self,
        job_id: str,
        status: str,
        message: Optional[str] = None,
    ) -> None:
        """Update job status."""
        job_data = await redis_cache.get(self._status_key(job_id))
        if not job_data:
            job_data = {"job_id": job_id}

        job_data["status"] = status
        job_data["updated_at"] = time.time()
        if message:
            job_data["message"] = message

        await redis_cache.set(self._status_key(job_id), job_data, ttl=3600)
        logger.info("Job %s status → %s", job_id, status)

    async def set_result(self, job_id: str, result: Dict) -> None:
        """Store job result in Redis."""
        await redis_cache.set(self._result_key(job_id), result, ttl=3600)
        await self.update_status(job_id, JOB_STATUS_COMPLETED)

    async def set_failed(self, job_id: str, error: str) -> None:
        """Mark job as failed with error message."""
        await redis_cache.set(
            self._result_key(job_id),
            {"error": error},
            ttl=3600,
        )
        await self.update_status(job_id, JOB_STATUS_FAILED, message=error)

    async def get_status(self, job_id: str) -> Dict:
        """Get current job status."""
        job_data = await redis_cache.get(self._status_key(job_id))
        if not job_data:
            return {
                "job_id": job_id,
                "status": "not_found",
                "message": "Job not found",
            }
        return job_data

    async def get_result(self, job_id: str) -> Optional[Dict]:
        """Get job result (only available after completion)."""
        return await redis_cache.get(self._result_key(job_id))


# Singleton
job_manager = JobManager()
