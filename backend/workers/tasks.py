"""
ARQ background task definitions.

Tasks run in a separate worker process, keeping the FastAPI server responsive.
The worker process has its own instances of embedding model, Qdrant client, etc.

Why ARQ over Celery:
- Native async/await (no gevent/eventlet monkey-patching)
- Built on redis.asyncio (same Redis client we already use)
- Lightweight (~500 LOC) vs Celery's massive codebase
- Perfect for this scale (not 10K workers)

Why background tasks for video processing:
- yt-dlp downloads take 5-30 seconds
- Whisper transcription takes 5-60 seconds
- Embedding generation takes 1-5 seconds
- Total: 15-90 seconds — way beyond HTTP timeout
"""

from backend.services.video_processor import video_processor
from backend.workers.job_manager import job_manager
from backend.utils.constants import JOB_STATUS_PROCESSING
from backend.utils.logger import get_logger

logger = get_logger(__name__)


async def process_videos_task(
    ctx: dict,
    youtube_url: str,
    instagram_url: str,
    session_id: str,
    job_id: str,
) -> dict:
    """
    Background task: process two video URLs.

    This runs in the ARQ worker process, not the FastAPI process.
    Results are stored in Redis for the API to retrieve.
    """
    logger.info(
        "Worker starting video processing — job=%s, session=%s",
        job_id,
        session_id,
    )

    try:
        # Update status to processing
        await job_manager.update_status(job_id, JOB_STATUS_PROCESSING)

        # Run the full pipeline
        result = await video_processor.process_videos(
            youtube_url=youtube_url,
            instagram_url=instagram_url,
            session_id=session_id,
        )

        # Store result
        if result.get("status") == "completed":
            await job_manager.set_result(job_id, result)
        else:
            await job_manager.set_failed(
                job_id,
                result.get("message", "Processing failed"),
            )

        return result

    except Exception as e:
        logger.error("Worker task failed — job=%s: %s", job_id, e, exc_info=True)
        await job_manager.set_failed(job_id, str(e))
        return {"status": "failed", "message": str(e)}
