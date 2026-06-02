"""
Video processing API routes.

Endpoints:
- POST /api/videos/process — Start video processing (returns session_id + job_id)
- GET  /api/videos/{session_id}/status — Poll processing status
- GET  /api/videos/{session_id}/metadata — Get video metadata + engagement
"""

import uuid
import io
import asyncio
import requests
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse

from backend.api.schemas.video import (
    ProcessVideosRequest,
    ProcessVideosResponse,
    VideoStatusResponse,
    VideoMetadataResponse,
)
from backend.services.video_processor import video_processor
from backend.workers.job_manager import job_manager
from backend.cache.redis_cache import redis_cache, RedisCache
from backend.utils.constants import REDIS_PREFIX_SESSION, REDIS_PREFIX_JOB
from backend.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/videos", tags=["Videos"])


@router.post("/process", response_model=ProcessVideosResponse)
async def process_videos(
    request: ProcessVideosRequest,
    background_tasks: BackgroundTasks,
):
    """
    Start processing two video URLs.

    This endpoint:
    1. Creates a session ID
    2. Starts video processing as a background task
    3. Returns immediately with session_id for polling

    The frontend should poll GET /api/videos/{session_id}/status
    until status is "completed" or "failed".
    """
    session_id = str(uuid.uuid4())[:12]
    job_id = await job_manager.create_job(session_id)

    logger.info(
        "Processing request — session=%s, job=%s, yt=%s, ig=%s",
        session_id,
        job_id,
        request.youtube_url,
        request.instagram_url,
    )

    # Run processing in background (using FastAPI BackgroundTasks
    # instead of ARQ to avoid requiring a separate worker process for demo)
    async def _run_processing():
        from backend.utils.constants import JOB_STATUS_PROCESSING
        await job_manager.update_status(job_id, JOB_STATUS_PROCESSING)

        try:
            result = await video_processor.process_videos(
                youtube_url=request.youtube_url,
                instagram_url=request.instagram_url,
                session_id=session_id,
            )

            if result.get("status") == "completed":
                await job_manager.set_result(job_id, result)
            else:
                await job_manager.set_failed(
                    job_id, result.get("message", "Processing failed")
                )
        except Exception as e:
            await job_manager.set_failed(job_id, str(e))

    background_tasks.add_task(_run_processing)

    return ProcessVideosResponse(
        status="processing",
        session_id=session_id,
        job_id=job_id,
        message="Video processing started. Poll /api/videos/{session_id}/status for updates.",
    )


@router.get("/{session_id}/status", response_model=VideoStatusResponse)
async def get_processing_status(session_id: str):
    """
    Get the current processing status for a session.

    Status values: pending, processing, completed, failed
    """
    # Find job_id for this session (scan recent jobs)
    # In a production system, we'd store session→job mapping
    # For now, check the session cache for a result
    session_key = RedisCache.make_key(REDIS_PREFIX_SESSION, session_id)
    session_data = await redis_cache.get(session_key)

    if session_data:
        return VideoStatusResponse(
            session_id=session_id,
            status="completed",
            result=session_data,
        )

    # Check if there's a job for this session
    # Scan jobs (simple approach for demo)
    return VideoStatusResponse(
        session_id=session_id,
        status="processing",
        message="Video processing in progress...",
    )


@router.get("/{session_id}/metadata", response_model=VideoMetadataResponse)
async def get_video_metadata(session_id: str):
    """
    Get metadata and engagement metrics for processed videos.

    Only available after processing is complete.
    """
    session_key = RedisCache.make_key(REDIS_PREFIX_SESSION, session_id)
    session_data = await redis_cache.get(session_key)

    if not session_data:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. Process videos first.",
        )

    return VideoMetadataResponse(
        session_id=session_id,
        youtube={
            "metadata": session_data.get("youtube_metadata"),
            "engagement": session_data.get("youtube_engagement"),
        },
        instagram={
            "metadata": session_data.get("instagram_metadata"),
            "engagement": session_data.get("instagram_engagement"),
        },
    )


@router.get("/proxy-image")
async def proxy_image(url: str):
    """
    Proxy video thumbnail images from external CDNs (like Instagram/Facebook CDN)
    to bypass browser CORS/CORP and hotlinking blocking.
    """
    if not url:
        raise HTTPException(status_code=400, detail="Missing url parameter")
        
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
    }
    
    try:
        # Fetch the image in a thread pool to avoid blocking the event loop
        def fetch():
            return requests.get(url, headers=headers, timeout=10)
            
        r = await asyncio.to_thread(fetch)
        
        if r.status_code == 200:
            content_type = r.headers.get("content-type", "image/jpeg")
            return StreamingResponse(
                io.BytesIO(r.content),
                media_type=content_type,
                headers={
                    "Cache-Control": "public, max-age=86400",
                    "Access-Control-Allow-Origin": "*",
                }
            )
        else:
            logger.warning("Image proxy returned status %d for %s", r.status_code, url)
            raise HTTPException(status_code=r.status_code, detail="Failed to fetch image from source")
    except Exception as e:
        logger.error("Image proxy failed for %s: %s", url, e)
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")
