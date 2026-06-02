"""
Chat API route with SSE streaming.

Endpoints:
- POST /api/chat — Streaming chat (Server-Sent Events) with dynamic URL processing fallback
- POST /api/chat/sync — Synchronous chat endpoint
"""

import re
import json
import hashlib
from typing import List, Dict, Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend.api.schemas.chat import ChatRequest, ChatResponse
from backend.rag.chain import rag_chain
from backend.services.video_processor import video_processor
from backend.cache.redis_cache import redis_cache, RedisCache
from backend.utils.constants import REDIS_PREFIX_SESSION, VIDEO_A_LABEL, VIDEO_B_LABEL
from backend.utils.logger import get_logger
from backend.utils.url_parser import detect_platform, Platform, parse_url

logger = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])

# Regex to detect YouTube or Instagram video URLs
URL_PATTERN = re.compile(
    r'(https?://(?:www\.)?(?:youtube\.com/(?:watch\?v=|shorts/)[a-zA-Z0-9_-]+|youtu\.be/[a-zA-Z0-9_-]+|instagram\.com/(?:reel|reels|p)/[a-zA-Z0-9_-]+))',
    re.IGNORECASE
)


def extract_and_clean_urls(text: str) -> List[str]:
    """Extract YouTube and Instagram URLs from text, normalize them, and deduplicate."""
    found_urls = URL_PATTERN.findall(text)
    cleaned = []
    seen = set()
    for url in found_urls:
        platform, video_id = parse_url(url)
        if not video_id:
            continue
        
        # Reconstruct standard normalized URL
        if platform == Platform.YOUTUBE:
            clean = f"https://www.youtube.com/watch?v={video_id}"
        elif platform == Platform.INSTAGRAM:
            clean = f"https://www.instagram.com/reel/{video_id}"
        else:
            continue
            
        if clean not in seen:
            cleaned.append(clean)
            seen.add(clean)
    return cleaned


@router.post("")
async def chat_stream(request: ChatRequest):
    """
    Stream a RAG-powered chat response via Server-Sent Events.

    If the user's question contains video URLs, this endpoint will dynamically
    trigger uploader/transcript/metadata extraction on-the-fly, stream status
    updates, index the content in Qdrant/Redis, and then stream the RAG analysis.

    SSE Format:
    - data: {"type": "status", "content": "..."}   — live status updates
    - data: {"type": "token", "content": "..."}    — each LLM token
    - data: {"type": "done", "citations": [...]}    — final event with metadata and session ID
    - data: {"type": "error", "message": "..."}     — on failure
    """
    question = request.question
    session_id = request.session_id
    video_filter = request.video_filter

    # 1. Parse video URLs from prompt
    urls = extract_and_clean_urls(question)
    needs_processing = False

    if urls:
        # Generate deterministic session_id based on sorted clean URLs
        sorted_urls = sorted(urls)
        session_id = hashlib.md5("".join(sorted_urls).encode()).hexdigest()[:12]
        
        # Check if the session exists in Redis
        session_key = RedisCache.make_key(REDIS_PREFIX_SESSION, session_id)
        session_data = await redis_cache.get(session_key)
        if not session_data:
            needs_processing = True
            
        # Store this as the most recently active session
        await redis_cache.set("last_active_session_id", session_id)
    else:
        # Fallback if no URLs were provided in this prompt
        if not session_id:
            session_id = await redis_cache.get("last_active_session_id")
            
        if not session_id:
            raise HTTPException(
                status_code=400,
                detail="No active session found. Please include YouTube/Instagram URLs in your prompt to start a comparison.",
            )
            
        # Load existing session
        session_key = RedisCache.make_key(REDIS_PREFIX_SESSION, session_id)
        session_data = await redis_cache.get(session_key)
        if not session_data:
            raise HTTPException(
                status_code=404,
                detail=f"Session '{session_id}' not found. Please provide video URLs to start a new session.",
            )

    # 2. Clean question by replacing raw URLs with Video labels (e.g. Video A, Video B...)
    cleaned_question = question
    url_to_label = {}
    for idx, url in enumerate(urls):
        label = f"Video {chr(65 + idx)}"  # Video A, Video B, Video C...
        url_to_label[url] = label

    for raw_url in URL_PATTERN.findall(question):
        platform, video_id = parse_url(raw_url)
        if not video_id:
            continue
        if platform == Platform.YOUTUBE:
            clean = f"https://www.youtube.com/watch?v={video_id}"
        elif platform == Platform.INSTAGRAM:
            clean = f"https://www.instagram.com/reel/{video_id}"
        else:
            continue
            
        if clean in url_to_label:
            cleaned_question = cleaned_question.replace(raw_url, url_to_label[clean])

    async def event_generator():
        nonlocal session_data
        
        # 3. Dynamic on-the-fly video processing with live status streaming
        if needs_processing:
            logger.info("Triggering dynamic video processing for URLs: %s (session=%s)", urls, session_id)
            yield f"data: {json.dumps({'type': 'status', 'content': f'🔍 Auto-detected {len(urls)} video URL(s). Initializing comparison...'})}\n\n"
            
            try:
                yield f"data: {json.dumps({'type': 'status', 'content': '🎙️ Extracting transcripts & metadata (this may take 10-15 seconds)...'})}\n\n"
                
                # Execute video processor
                result = await video_processor.process_videos(urls=urls, session_id=session_id)
                
                if result.get("status") != "completed":
                    err_msg = result.get("message", "Video extraction failed")
                    yield f"data: {json.dumps({'type': 'error', 'message': f'Processing failed: {err_msg}'})}\n\n"
                    return
                
                # Fetch completed session metadata
                session_key = RedisCache.make_key(REDIS_PREFIX_SESSION, session_id)
                session_data = await redis_cache.get(session_key)
                
                yield f"data: {json.dumps({'type': 'status', 'content': '✨ Indexing complete. Generating comparison report...'})}\n\n"
                
            except Exception as e:
                logger.error("Dynamic processing failed: %s", e, exc_info=True)
                yield f"data: {json.dumps({'type': 'error', 'message': f'Dynamic processing failed: {str(e)}'})}\n\n"
                return

        # 4. Build context and filter video IDs
        metadata_context = video_processor.format_metadata_context(session_data)
        
        # Video filter options
        filter_video_ids = None
        if video_filter and "videos" in session_data:
            # Map video label filter (e.g. 'A') to the matching video ID
            target_label = f"Video {video_filter.upper()}"
            filter_video_ids = [
                v["video_id"] for v in session_data["videos"]
                if v.get("label") == target_label and v.get("video_id")
            ]

        # 5. Execute RAG stream
        logger.info("Executing RAG stream — session=%s, query='%s'", session_id, cleaned_question[:50])
        async for event in rag_chain.chat_stream(
            question=cleaned_question,
            session_id=session_id,
            metadata_context=metadata_context,
            video_ids=filter_video_ids,
        ):
            if event["type"] == "done":
                event["session_id"] = session_id
                event["session_data"] = session_data
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering
        },
    )


@router.post("/sync", response_model=ChatResponse)
async def chat_sync(request: ChatRequest):
    """
    Non-streaming chat endpoint (for testing/debugging).
    """
    question = request.question
    session_id = request.session_id

    # 1. Parse video URLs
    urls = extract_and_clean_urls(question)
    
    if urls:
        sorted_urls = sorted(urls)
        session_id = hashlib.md5("".join(sorted_urls).encode()).hexdigest()[:12]
        
        session_key = RedisCache.make_key(REDIS_PREFIX_SESSION, session_id)
        session_data = await redis_cache.get(session_key)
        if not session_data:
            logger.info("Sync chat: Triggering on-the-fly video processing for session=%s", session_id)
            result = await video_processor.process_videos(urls=urls, session_id=session_id)
            if result.get("status") != "completed":
                raise HTTPException(status_code=500, detail=f"Processing failed: {result.get('message')}")
            
            session_data = await redis_cache.get(session_key)
            
        await redis_cache.set("last_active_session_id", session_id)
    else:
        if not session_id:
            session_id = await redis_cache.get("last_active_session_id")
            
        if not session_id:
            raise HTTPException(status_code=400, detail="No active session found. Provide video URLs.")
            
        session_key = RedisCache.make_key(REDIS_PREFIX_SESSION, session_id)
        session_data = await redis_cache.get(session_key)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found.")

    # 2. Clean question
    cleaned_question = question
    url_to_label = {}
    for idx, url in enumerate(urls):
        label = f"Video {chr(65 + idx)}"
        url_to_label[url] = label

    for raw_url in URL_PATTERN.findall(question):
        platform, video_id = parse_url(raw_url)
        if not video_id:
            continue
        if platform == Platform.YOUTUBE:
            clean = f"https://www.youtube.com/watch?v={video_id}"
        elif platform == Platform.INSTAGRAM:
            clean = f"https://www.instagram.com/reel/{video_id}"
        else:
            continue
            
        if clean in url_to_label:
            cleaned_question = cleaned_question.replace(raw_url, url_to_label[clean])

    # 3. Format context & execute RAG
    metadata_context = video_processor.format_metadata_context(session_data)
    
    result = await rag_chain.chat(
        question=cleaned_question,
        session_id=session_id,
        metadata_context=metadata_context,
    )
    
    return ChatResponse(**result)
