"""
Metadata extraction and engagement analytics service.

Ported from research/2metadata_extraction.ipynb with production hardening:
- Pydantic v2 models
- Redis caching of metadata
- Async wrappers around synchronous yt-dlp calls
- Engagement metrics calculation (engagement rate, virality score, etc.)

Why yt-dlp for metadata:
- Single tool for both YouTube and Instagram
- Extracts everything: views, likes, comments, duration, hashtags, creator info
- No API keys required
- Handles rate limiting and anti-bot measures internally
"""

import re
import json
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass

from pydantic import BaseModel, Field, field_validator

from backend.config import settings
from backend.cache.redis_cache import redis_cache, RedisCache
from backend.utils.logger import get_logger
from backend.utils.url_parser import detect_platform, extract_video_id, Platform
from backend.utils import constants

logger = get_logger(__name__)


# ── Pydantic Models (upgraded from research notebook) ────


class VideoMetadata(BaseModel):
    """Structured metadata for any video platform."""

    platform: str = Field(..., description="youtube or instagram")
    video_id: str = Field(..., description="Unique video identifier")
    title: str = ""
    description: Optional[str] = None
    duration_seconds: float = Field(default=0, description="Video length in seconds")
    upload_date: Optional[str] = None  # ISO format

    # Engagement metrics (raw)
    view_count: Optional[int] = None
    like_count: Optional[int] = None
    comment_count: Optional[int] = None

    # Creator info
    creator_name: str = "Unknown"
    creator_id: Optional[str] = None
    follower_count: Optional[int] = None

    # Additional metadata
    hashtags: List[str] = Field(default_factory=list)
    thumbnail_url: Optional[str] = None
    video_url: str = ""

    is_live: bool = False
    is_age_restricted: bool = False
    extracted_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    @field_validator("duration_seconds")
    @classmethod
    def duration_non_negative(cls, v):
        """Duration must be non-negative."""
        if v < 0:
            return 0
        return v

    @field_validator("view_count", "like_count", "comment_count", mode="before")
    @classmethod
    def positive_counts(cls, v):
        """Treat negative counts as missing data."""
        if v is not None and v < 0:
            return None
        return v


class EngagementMetrics(BaseModel):
    """Calculated engagement analytics for a video."""

    video_id: str
    platform: str
    views: int = 0
    likes: int = 0
    comments: int = 0

    engagement_rate: float = 0.0      # (likes + comments) / views × 100
    like_rate: float = 0.0            # likes / views × 100
    comment_rate: float = 0.0         # comments / views × 100
    comment_to_like_ratio: float = 0.0
    virality_score: float = 0.0       # Custom 0-100 score

    @classmethod
    def from_metadata(cls, metadata: VideoMetadata) -> "EngagementMetrics":
        """
        Calculate engagement metrics from video metadata.
        Formula ported from research/2metadata_extraction.ipynb.
        """
        views = metadata.view_count or 0
        likes = metadata.like_count or 0
        comments = metadata.comment_count or 0

        engagement_rate = 0.0
        like_rate = 0.0
        comment_rate = 0.0
        comment_to_like_ratio = 0.0
        virality_score = 0.0

        if views > 0:
            engagement_rate = round(((likes + comments) / views) * 100, 2)
            like_rate = round((likes / views) * 100, 2)
            comment_rate = round((comments / views) * 100, 2)

        if likes > 0:
            comment_to_like_ratio = round(comments / likes, 2)

        # Virality score: normalized engagement + like rate, weighted
        if views > 0:
            normalized_engagement = min(engagement_rate / 5, 1.0)  # 5% is excellent
            normalized_likes = min(like_rate / 2, 1.0)             # 2% is excellent
            virality_score = round(
                (normalized_engagement * 0.6 + normalized_likes * 0.4) * 100, 2
            )

        return cls(
            video_id=metadata.video_id,
            platform=metadata.platform,
            views=views,
            likes=likes,
            comments=comments,
            engagement_rate=engagement_rate,
            like_rate=like_rate,
            comment_rate=comment_rate,
            comment_to_like_ratio=comment_to_like_ratio,
            virality_score=virality_score,
        )


# ── Service Class ────────────────────────────────────────

class MetadataService:
    """Extract metadata and compute engagement analytics."""

    # Public API
    async def extract_metadata(self, url: str) -> Dict:
        """
        Extract metadata from a video URL.
        Checks cache first, then extracts via yt-dlp.

        Args:
            url: YouTube or Instagram URL

        Returns:
            Dict with 'metadata' (VideoMetadata) and 'engagement' (EngagementMetrics)
        """
        platform = detect_platform(url)
        video_id = extract_video_id(url)

        if not video_id:
            return {
                "status": "error",
                "error_type": "InvalidURL",
                "message": f"Could not extract video ID from: {url}",
            }

        # Check cache
        cache_key = RedisCache.make_key(
            constants.REDIS_PREFIX_METADATA,
            RedisCache.hash_url(url),
        )
        cached = await redis_cache.get(cache_key)
        if cached:
            logger.info("Metadata cache HIT for %s", video_id)
            cached["from_cache"] = True
            return cached

        logger.info("Metadata cache MISS for %s — extracting", video_id)

        # Extract via yt-dlp (works for both platforms)
        raw_result = await asyncio.to_thread(self._extract_with_ytdlp, url, platform)

        if raw_result["status"] != "success":
            return raw_result

        # Build structured models
        metadata = raw_result["metadata"]
        engagement = EngagementMetrics.from_metadata(metadata)

        result = {
            "status": "success",
            "metadata": metadata.model_dump(),
            "engagement": engagement.model_dump(),
            "from_cache": False,
        }

        # Cache
        await redis_cache.set(
            cache_key, result, ttl=settings.metadata_cache_ttl
        )

        return result

    # ── Internal extraction ──────────────────────────────

    def _extract_with_ytdlp(self, url: str, platform: Platform) -> Dict:
        """
        Extract metadata using yt-dlp.
        Ported from research/2metadata_extraction.ipynb cells 3-4.
        """
        import yt_dlp

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logger.info("Extracting metadata from %s", url)
                info = ydl.extract_info(url, download=False)

            # Build VideoMetadata from yt-dlp info dict
            metadata = VideoMetadata(
                platform=platform.value,
                video_id=info.get("id", ""),
                title=info.get("title", ""),
                description=info.get("description", ""),
                duration_seconds=info.get("duration", 0) or 0,
                upload_date=self._parse_date(info.get("upload_date")),
                view_count=info.get("view_count"),
                like_count=info.get("like_count"),
                comment_count=info.get("comment_count"),
                creator_name=info.get("uploader", "Unknown"),
                creator_id=info.get("channel_id") or info.get("uploader_id"),
                follower_count=info.get("channel_follower_count"),
                hashtags=self._extract_hashtags(info.get("description", "")),
                thumbnail_url=info.get("thumbnail"),
                video_url=info.get("webpage_url", url),
                is_live=info.get("is_live", False) or False,
                is_age_restricted=(info.get("age_limit", 0) or 0) > 0,
            )

            # Fallback for Instagram views (yt-dlp frequently returns null view_count for Reels)
            if platform == Platform.INSTAGRAM and not metadata.view_count:
                fallback_views = self._fetch_instagram_views_fallback(metadata.video_id)
                if fallback_views:
                    logger.info("Successfully fetched fallback views for IG reel %s: %d", metadata.video_id, fallback_views)
                    metadata.view_count = fallback_views

            return {"status": "success", "metadata": metadata}

        except Exception as e:
            logger.error("Metadata extraction failed for %s: %s", url, e)
            return {
                "status": "error",
                "error_type": type(e).__name__,
                "message": str(e),
                "url": url,
            }

    def _fetch_instagram_views_fallback(self, shortcode: str) -> Optional[int]:
        """
        Fetch Instagram reel view count using unauthenticated web GraphQL query.
        """
        import requests

        url = "https://www.instagram.com/graphql/query/"
        variables = {
            "shortcode": shortcode,
            "child_comment_count": 3,
            "fetch_comment_count": 40,
            "parent_comment_count": 24,
            "has_threaded_comments": True
        }
        params = {
            "doc_id": "8845758582119845",
            "variables": json.dumps(variables)
        }
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'x-ig-app-id': '936619743392459',
            'Accept': '*/*',
            'Origin': 'https://www.instagram.com',
            'Referer': f'https://www.instagram.com/reel/{shortcode}/',
        }

        try:
            logger.info("Querying Instagram GraphQL view count fallback for reel %s...", shortcode)
            r = requests.get(url, params=params, headers=headers, timeout=10)
            if r.status_code == 200:
                res_data = r.json()
                media = res_data.get('data', {}).get('xdt_shortcode_media', {})
                if media:
                    # video_play_count is the actual play/view count visible on IG profile grid
                    plays = media.get('video_play_count')
                    views = media.get('video_view_count')
                    val = plays or views
                    if val is not None:
                        return int(val)
            else:
                logger.warning(
                    "Instagram views fallback returned status %d for %s: %s",
                    r.status_code, shortcode, r.text[:200]
                )
        except Exception as e:
            logger.error("Instagram views fallback failed for %s: %s", shortcode, e)
        return None

    @staticmethod
    def _parse_date(date_str: Optional[str]) -> Optional[str]:
        """Convert YYYYMMDD to ISO format."""
        if not date_str:
            return None
        try:
            return datetime.strptime(date_str, "%Y%m%d").isoformat()
        except (ValueError, TypeError):
            return date_str

    @staticmethod
    def _extract_hashtags(text: str) -> List[str]:
        """Extract hashtags from description text."""
        if not text:
            return []
        return re.findall(r"#\w+", text)


# Singleton
metadata_service = MetadataService()
