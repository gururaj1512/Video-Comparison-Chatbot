"""
Transcript extraction service.

Ported from research/1transcript_extraction.ipynb with production hardening:
- Async wrapper around synchronous yt-dlp / youtube-transcript-api
- Redis caching to avoid re-extracting same URLs
- Automatic cleanup of temporary audio files
- Structured error handling with fallback chain

Extraction Strategy:
  YouTube:  youtube-transcript-api (fast, free) → yt-dlp + faster-whisper (fallback)
  Instagram: yt-dlp audio download → faster-whisper transcription (only method)

Why youtube-transcript-api before yt-dlp+whisper:
- 10x faster (no download/transcription needed)
- Higher accuracy (uses YouTube's own captions)
- Falls back to Whisper only when captions are disabled/unavailable

Why faster-whisper over openai-whisper:
- 4x faster inference on CPU (CTranslate2 backend)
- Lower memory footprint
- Same accuracy (same model weights)
"""

import os
import asyncio
import shutil
from typing import Dict, List, Optional
from datetime import datetime

from backend.config import settings
from backend.cache.redis_cache import redis_cache, RedisCache
from backend.utils.logger import get_logger
from backend.utils.url_parser import detect_platform, extract_video_id, Platform
from backend.utils import constants

logger = get_logger(__name__)


class TranscriptService:
    """Extract transcripts from YouTube and Instagram videos."""

    def __init__(self):
        self._whisper_model = None  # Lazy-loaded

    # ── Public API ───────────────────────────────────────

    async def extract_transcript(self, url: str) -> Dict:
        """
        Extract transcript from any supported URL.
        Checks cache first, then extracts.

        Args:
            url: YouTube or Instagram video URL

        Returns:
            Dict with keys: status, transcript, entries, method,
                            video_id, platform, duration_seconds, etc.
        """
        platform = detect_platform(url)
        video_id = extract_video_id(url)

        if not video_id:
            return {
                "status": "error",
                "error_type": "InvalidURL",
                "message": f"Could not extract video ID from URL: {url}",
                "url": url,
            }

        # Check cache
        cache_key = RedisCache.make_key(
            constants.REDIS_PREFIX_TRANSCRIPT,
            RedisCache.hash_url(url),
        )
        cached = await redis_cache.get(cache_key)
        if cached:
            logger.info("Transcript cache HIT for %s", video_id)
            cached["from_cache"] = True
            return cached

        logger.info("Transcript cache MISS for %s — extracting", video_id)

        # Extract based on platform
        if platform == Platform.YOUTUBE:
            result = await self._extract_youtube(url, video_id)
        elif platform == Platform.INSTAGRAM:
            result = await self._extract_instagram(url, video_id)
        else:
            return {
                "status": "error",
                "error_type": "UnsupportedPlatform",
                "message": f"Unsupported platform for URL: {url}",
                "url": url,
            }

        # Cache successful results
        if result.get("status") == "success":
            result["from_cache"] = False
            await redis_cache.set(
                cache_key, result, ttl=settings.transcript_cache_ttl
            )

        return result

    # ── YouTube Extraction ───────────────────────────────

    async def _extract_youtube(self, url: str, video_id: str) -> Dict:
        """
        YouTube transcript extraction with fallback chain.
        1. Try youtube-transcript-api (fastest)
        2. Fallback to yt-dlp + faster-whisper
        """
        # Method 1: youtube-transcript-api
        result = await asyncio.to_thread(
            self._youtube_transcript_api, url, video_id
        )

        if result["status"] == "success":
            return result

        logger.warning(
            "youtube-transcript-api failed for %s: %s — trying whisper fallback",
            video_id,
            result.get("message", "unknown"),
        )

        # Method 2: yt-dlp + faster-whisper fallback
        return await asyncio.to_thread(
            self._whisper_fallback, url, video_id, Platform.YOUTUBE
        )

    def _youtube_transcript_api(self, url: str, video_id: str) -> Dict:
        """
        Direct transcript extraction via youtube-transcript-api.
        Ported from research/1transcript_extraction.ipynb cell 3.
        """
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            from youtube_transcript_api._errors import (
                TranscriptsDisabled,
                NoTranscriptFound,
                VideoUnavailable,
            )

            # Attempt to get available transcripts (handle version differences)
            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            except AttributeError:
                transcript_list = YouTubeTranscriptApi().list(video_id)
            except TypeError as exc:
                if "required positional argument" not in str(exc):
                    raise
                transcript_list = YouTubeTranscriptApi().list_transcripts(video_id)

            # Try English first, then fallback languages
            try:
                transcript = transcript_list.find_transcript(["en"])
            except NoTranscriptFound:
                transcript = transcript_list.find_transcript(
                    ["en", "es", "fr", "de", "hi"]
                )

            transcript_data = transcript.fetch()

            # Normalize entries (handle dict/attribute differences across versions)
            entries = []
            for entry in transcript_data:
                if isinstance(entry, dict):
                    entries.append(entry)
                else:
                    entries.append({
                        "text": getattr(entry, "text", ""),
                        "start": getattr(entry, "start", 0.0),
                        "duration": getattr(entry, "duration", 0.0),
                    })

            full_text = " ".join(
                e["text"] for e in entries if e.get("text")
            )

            duration_seconds = 0.0
            if entries:
                last = entries[-1]
                duration_seconds = last.get("start", 0.0) + last.get("duration", 0.0)

            return {
                "status": "success",
                "method": "youtube-transcript-api",
                "video_id": video_id,
                "platform": Platform.YOUTUBE.value,
                "transcript": full_text,
                "entries": entries,
                "entry_count": len(entries),
                "duration_seconds": duration_seconds,
                "language": transcript.language,
                "is_auto_generated": transcript.is_generated,
                "extracted_at": datetime.now().isoformat(),
            }

        except VideoUnavailable:
            return {
                "status": "error",
                "error_type": "VideoUnavailable",
                "message": f"Video {video_id} is not available (private/deleted)",
                "url": url,
            }
        except TranscriptsDisabled:
            return {
                "status": "fallback_required",
                "error_type": "TranscriptsDisabled",
                "message": "Transcripts are disabled for this video",
                "url": url,
            }
        except NoTranscriptFound:
            return {
                "status": "fallback_required",
                "error_type": "NoTranscriptFound",
                "message": "No transcripts available via API",
                "url": url,
            }
        except Exception as e:
            return {
                "status": "error",
                "error_type": type(e).__name__,
                "message": str(e),
                "url": url,
            }

    # ── Instagram Extraction ─────────────────────────────

    async def _extract_instagram(self, url: str, video_id: str) -> Dict:
        """
        Instagram transcript extraction via yt-dlp + faster-whisper.
        Instagram has no transcript API — audio download + ASR is the only method.
        """
        return await asyncio.to_thread(
            self._whisper_fallback, url, video_id, Platform.INSTAGRAM
        )

    # ── Whisper Fallback (shared by YouTube + Instagram) ─

    def _whisper_fallback(
        self, url: str, video_id: str, platform: Platform
    ) -> Dict:
        """
        Download audio with yt-dlp, transcribe with faster-whisper.
        Ported from research/1transcript_extraction.ipynb cells 5-7.
        """
        import yt_dlp

        # Ensure temp directory exists
        os.makedirs(settings.temp_dir, exist_ok=True)

        audio_file = None
        try:
            # Step 1: Download audio
            ydl_opts = {
                "format": "bestaudio/best",
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": "192",
                    }
                ],
                "outtmpl": os.path.join(settings.temp_dir, "%(id)s.%(ext)s"),
                "quiet": True,
                "no_warnings": True,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logger.info("Downloading audio from %s", url)
                info = ydl.extract_info(url, download=True)
                audio_file = os.path.join(
                    settings.temp_dir, f"{info['id']}.mp3"
                )

            if not os.path.exists(audio_file):
                return {
                    "status": "error",
                    "error_type": "DownloadFailed",
                    "message": "Audio file not found after download",
                    "url": url,
                }

            # Step 2: Transcribe with faster-whisper
            logger.info("Transcribing audio for %s with whisper-%s",
                        video_id, settings.whisper_model_size)

            if self._whisper_model is None:
                from faster_whisper import WhisperModel
                self._whisper_model = WhisperModel(
                    settings.whisper_model_size,
                    device="cpu",
                    compute_type="int8",
                )

            segments, info_whisper = self._whisper_model.transcribe(
                audio_file, language="en"
            )

            entries = []
            full_text_parts = []
            for segment in segments:
                entries.append({
                    "text": segment.text.strip(),
                    "start": segment.start,
                    "duration": segment.end - segment.start,
                })
                full_text_parts.append(segment.text.strip())

            full_text = " ".join(full_text_parts)

            return {
                "status": "success",
                "method": "yt-dlp + faster-whisper",
                "video_id": video_id,
                "platform": platform.value,
                "transcript": full_text,
                "entries": entries,
                "entry_count": len(entries),
                "duration_seconds": info_whisper.duration,
                "language": info_whisper.language,
                "is_auto_generated": True,
                "extracted_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error("Whisper fallback failed for %s: %s", url, e)
            return {
                "status": "error",
                "error_type": type(e).__name__,
                "message": str(e),
                "url": url,
            }
        finally:
            # Cleanup audio file
            if audio_file and os.path.exists(audio_file):
                try:
                    os.remove(audio_file)
                    logger.debug("Cleaned up audio file: %s", audio_file)
                except OSError:
                    pass


# Singleton
transcript_service = TranscriptService()
