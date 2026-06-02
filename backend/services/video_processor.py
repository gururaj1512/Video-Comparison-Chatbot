"""
Video processing orchestrator.

Coordinates the full pipeline for processing two video URLs:
1. Extract transcripts (YouTube + Instagram concurrently)
2. Extract metadata (concurrently with transcripts)
3. Chunk transcripts
4. Generate embeddings
5. Store in Qdrant vector DB
6. Cache metadata in Redis

Why an orchestrator service:
- Keeps individual services (transcript, metadata, chunker, etc.) focused and testable
- Single entry point for the API layer
- Handles concurrent execution of independent tasks
- Manages session lifecycle (create, process, cleanup)

Why asyncio.gather for concurrency:
- YouTube and Instagram extraction are independent — no reason to wait sequentially
- On a 2-core machine, this cuts wall-clock time nearly in half
- asyncio.gather propagates exceptions cleanly
"""

import uuid
import asyncio
import time
from typing import Dict, List, Optional

from backend.services.transcript_service import transcript_service
from backend.services.metadata_service import metadata_service, EngagementMetrics
from backend.rag.chunker import chunker
from backend.rag.embedding_service import embedding_service
from backend.rag.vector_store import vector_store
from backend.cache.redis_cache import redis_cache, RedisCache
from backend.config import settings
from backend.utils.logger import get_logger
from backend.utils.url_parser import Platform, detect_platform
from backend.utils.constants import (
    VIDEO_A_LABEL,
    VIDEO_B_LABEL,
    REDIS_PREFIX_SESSION,
)

logger = get_logger(__name__)


class VideoProcessor:
    """Orchestrates the full video processing pipeline."""

    async def process_videos(
        self,
        youtube_url: Optional[str] = None,
        instagram_url: Optional[str] = None,
        urls: Optional[List[str]] = None,
        session_id: Optional[str] = None,
    ) -> Dict:
        """
        Process multiple video URLs end-to-end.
        Supports both legacy parameters and a list of urls.
        """
        start_time = time.time()
        session_id = session_id or str(uuid.uuid4())[:12]

        if not urls:
            urls = []
            if youtube_url:
                urls.append(youtube_url)
            if instagram_url:
                urls.append(instagram_url)

        logger.info(
            "Starting video processing — session=%s, urls=%s",
            session_id,
            urls,
        )

        result = {
            "session_id": session_id,
            "status": "processing",
            "videos": [],
            "errors": [],
        }

        # Keep youtube and instagram dicts in top level for backward compatibility
        result["youtube"] = {}
        result["instagram"] = {}

        if not urls:
            result["status"] = "failed"
            result["message"] = "No video URLs provided"
            return result

        try:
            # ── Step 1: Extract transcripts + metadata concurrently ───
            tasks = []
            for url in urls:
                tasks.append(transcript_service.extract_transcript(url))
                tasks.append(metadata_service.extract_metadata(url))

            raw_results = await asyncio.gather(*tasks, return_exceptions=True)

            extracted_videos = []
            for i, url in enumerate(urls):
                label = f"Video {chr(65 + i)}"  # Video A, Video B, Video C...
                t_res = raw_results[i * 2]
                m_res = raw_results[i * 2 + 1]

                # Handle exceptions
                if isinstance(t_res, Exception):
                    error_msg = f"{label} transcript extraction failed: {t_res}"
                    logger.error(error_msg)
                    result["errors"].append(error_msg)
                    t_res = {"status": "error", "message": str(t_res)}
                elif t_res.get("status") != "success":
                    result["errors"].append(f"{label} transcript: {t_res.get('message', 'failed')}")

                if isinstance(m_res, Exception):
                    error_msg = f"{label} metadata extraction failed: {m_res}"
                    logger.error(error_msg)
                    result["errors"].append(error_msg)
                    m_res = {"status": "error", "message": str(m_res)}
                elif m_res.get("status") != "success":
                    result["errors"].append(f"{label} metadata: {m_res.get('message', 'failed')}")

                extracted_videos.append({
                    "url": url,
                    "label": label,
                    "transcript_res": t_res,
                    "metadata_res": m_res,
                })

            # Check if all transcript extractions failed
            any_transcript_ok = any(
                v["transcript_res"].get("status") == "success"
                for v in extracted_videos
            )
            if not any_transcript_ok:
                result["status"] = "failed"
                result["message"] = "All transcript extractions failed"
                return result

            # ── Step 2: Chunk transcripts ────────────────────────────
            all_chunks = []
            for video in extracted_videos:
                t_res = video["transcript_res"]
                if t_res.get("status") == "success":
                    video_id = t_res.get("video_id", "unknown")
                    platform = detect_platform(video["url"])
                    chunks = chunker.chunk_with_timestamps(
                        text=t_res["transcript"],
                        video_id=video_id,
                        source=platform.value,
                        transcript_entries=t_res.get("entries", []),
                        video_label=video["label"],
                    )
                    all_chunks.extend(chunks)
                    video["video_id"] = video_id
                    video["chunk_count"] = len(chunks)

                    # For backward compatibility
                    p_name = "youtube" if platform == Platform.YOUTUBE else "instagram"
                    result[p_name]["video_id"] = video_id
                    result[p_name]["chunk_count"] = len(chunks)
                    result[p_name]["transcript_length"] = len(t_res.get("transcript", ""))

            if not all_chunks:
                result["status"] = "failed"
                result["message"] = "No chunks generated from transcripts"
                return result

            # ── Step 3: Generate embeddings ──────────────────────────
            texts = [chunk.text for chunk in all_chunks]
            embeddings = await embedding_service.aembed_batch(texts)
            logger.info("Generated %d embeddings", len(embeddings))

            # ── Step 4: Store in Qdrant ──────────────────────────────
            vector_store.create_collection(session_id)
            payloads = [chunk.to_payload() for chunk in all_chunks]
            vector_store.upsert_chunks(session_id, embeddings, payloads)

            # ── Step 5: Cache metadata in Redis ──────────────────────
            session_meta = {
                "session_id": session_id,
                "urls": urls,
                "videos": [],
            }

            for video in extracted_videos:
                m_res = video["metadata_res"]
                platform = detect_platform(video["url"])
                
                video_entry = {
                    "url": video["url"],
                    "label": video["label"],
                    "video_id": video.get("video_id"),
                    "platform": platform.value,
                    "metadata": m_res.get("metadata") if m_res.get("status") == "success" else None,
                    "engagement": m_res.get("engagement") if m_res.get("status") == "success" else None,
                }
                session_meta["videos"].append(video_entry)
                result["videos"].append(video_entry)

                # Legacy mapping
                p_name = "youtube" if platform == Platform.YOUTUBE else "instagram"
                if m_res.get("status") == "success":
                    result[p_name]["metadata"] = m_res["metadata"]
                    result[p_name]["engagement"] = m_res["engagement"]
                else:
                    result[p_name]["metadata"] = None
                    result[p_name]["engagement"] = None

            # Legacy compatibility for session_meta keys
            # Store first youtube and instagram metadata to top-level legacy keys
            for v_entry in session_meta["videos"]:
                if v_entry["platform"] == Platform.YOUTUBE.value and "youtube_metadata" not in session_meta:
                    session_meta["youtube_url"] = v_entry["url"]
                    session_meta["youtube_metadata"] = v_entry["metadata"]
                    session_meta["youtube_engagement"] = v_entry["engagement"]
                elif v_entry["platform"] == Platform.INSTAGRAM.value and "instagram_metadata" not in session_meta:
                    session_meta["instagram_url"] = v_entry["url"]
                    session_meta["instagram_metadata"] = v_entry["metadata"]
                    session_meta["instagram_engagement"] = v_entry["engagement"]

            # Safe fallbacks if any legacy keys are still missing
            if "youtube_metadata" not in session_meta and len(session_meta["videos"]) > 0:
                v0 = session_meta["videos"][0]
                session_meta["youtube_url"] = v0["url"]
                session_meta["youtube_metadata"] = v0["metadata"]
                session_meta["youtube_engagement"] = v0["engagement"]
            if "instagram_metadata" not in session_meta and len(session_meta["videos"]) > 1:
                v1 = session_meta["videos"][1]
                session_meta["instagram_url"] = v1["url"]
                session_meta["instagram_metadata"] = v1["metadata"]
                session_meta["instagram_engagement"] = v1["engagement"]

            # Cache session metadata for chat
            session_cache_key = RedisCache.make_key(
                REDIS_PREFIX_SESSION, session_id
            )
            await redis_cache.set(
                session_cache_key, session_meta, ttl=settings.session_ttl
            )

            # ── Done ─────────────────────────────────────────────────
            elapsed = time.time() - start_time
            result["status"] = "completed"
            result["processing_time_seconds"] = round(elapsed, 2)
            result["total_chunks"] = len(all_chunks)

            logger.info(
                "Video processing completed — session=%s, chunks=%d, time=%.2fs",
                session_id,
                len(all_chunks),
                elapsed,
            )

            return result

        except Exception as e:
            logger.error(
                "Video processing failed for session %s: %s",
                session_id,
                e,
                exc_info=True,
            )
            result["status"] = "failed"
            result["message"] = str(e)
            return result

    def format_metadata_context(self, session_meta: Dict) -> str:
        """
        Format cached session metadata into a context string for the LLM.
        Supports both new videos list structure and legacy youtube/instagram keys.
        """
        parts = []
        videos = session_meta.get("videos")

        if not videos:
            videos = []
            for label, key_prefix in [
                (VIDEO_A_LABEL, "youtube"),
                (VIDEO_B_LABEL, "instagram"),
            ]:
                if f"{key_prefix}_metadata" in session_meta:
                    videos.append({
                        "label": label,
                        "platform": session_meta.get(f"{key_prefix}_metadata", {}).get("platform", "unknown"),
                        "metadata": session_meta.get(f"{key_prefix}_metadata"),
                        "engagement": session_meta.get(f"{key_prefix}_engagement"),
                    })

        for video in videos:
            label = video.get("label")
            platform = video.get("platform", "unknown")
            meta = video.get("metadata")
            engagement = video.get("engagement")

            if meta:
                parts.append(f"### {label} ({platform})")
                parts.append(f"- **Title**: {meta.get('title', 'N/A')}")
                parts.append(f"- **Creator**: {meta.get('creator_name', 'N/A')}")
                parts.append(f"- **Views**: {meta.get('view_count', 'N/A'):,}" if isinstance(meta.get('view_count'), int) else f"- **Views**: {meta.get('view_count', 'N/A')}")
                parts.append(f"- **Likes**: {meta.get('like_count', 'N/A'):,}" if isinstance(meta.get('like_count'), int) else f"- **Likes**: {meta.get('like_count', 'N/A')}")
                parts.append(f"- **Comments**: {meta.get('comment_count', 'N/A'):,}" if isinstance(meta.get('comment_count'), int) else f"- **Comments**: {meta.get('comment_count', 'N/A')}")
                parts.append(f"- **Duration**: {meta.get('duration_seconds', 0)}s")
                parts.append(f"- **Upload Date**: {meta.get('upload_date', 'N/A')}")
                parts.append(f"- **Hashtags**: {', '.join(meta.get('hashtags', [])) or 'None'}")

                if engagement:
                    parts.append(f"- **Engagement Rate**: {engagement.get('engagement_rate', 0)}%")
                    parts.append(f"- **Like Rate**: {engagement.get('like_rate', 0)}%")
                    parts.append(f"- **Comment Rate**: {engagement.get('comment_rate', 0)}%")
                    parts.append(f"- **Virality Score**: {engagement.get('virality_score', 0)}/100")

                parts.append("")

        return "\n".join(parts) if parts else "No metadata available."


# Singleton
video_processor = VideoProcessor()
