"""
Async Redis caching layer.

Why Redis over in-memory dict:
- Survives process restarts (persistence via appendonly)
- Shared across multiple uvicorn workers
- TTL-based expiration (no manual cleanup)
- Sub-millisecond reads for cached transcripts/metadata

Why redis.asyncio over synchronous redis-py:
- FastAPI is async-first; blocking Redis calls would starve the event loop
- aioredis was merged into redis-py 5.0+ as redis.asyncio
"""

import json
import hashlib
from typing import Any, Optional

import redis.asyncio as aioredis

from backend.config import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class RedisCache:
    """Async Redis cache with JSON serialization."""

    def __init__(self):
        self._client: Optional[aioredis.Redis] = None

    async def connect(self) -> None:
        """Initialize Redis connection pool."""
        try:
            self._client = aioredis.from_url(
                settings.redis_url,
                decode_responses=True,
                max_connections=20,
            )
            # Test connection
            await self._client.ping()
            logger.info("Redis connected successfully at %s", settings.redis_url)
        except Exception as e:
            logger.error("Failed to connect to Redis: %s", e)
            self._client = None

    async def disconnect(self) -> None:
        """Close Redis connection pool."""
        if self._client:
            await self._client.aclose()
            logger.info("Redis connection closed")

    @property
    def is_connected(self) -> bool:
        """Check if Redis is available."""
        return self._client is not None

    async def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache.

        Args:
            key: Cache key

        Returns:
            Deserialized value or None if not found / Redis unavailable
        """
        if not self._client:
            return None

        try:
            value = await self._client.get(key)
            if value is None:
                return None
            return json.loads(value)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("Redis GET error for key '%s': %s", key, e)
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Set a value in cache with optional TTL.

        Args:
            key: Cache key
            value: Value to cache (must be JSON-serializable)
            ttl: Time-to-live in seconds (None = no expiry)

        Returns:
            True if successful
        """
        if not self._client:
            return False

        try:
            serialized = json.dumps(value, default=str)
            if ttl:
                await self._client.setex(key, ttl, serialized)
            else:
                await self._client.set(key, serialized)
            return True
        except Exception as e:
            logger.warning("Redis SET error for key '%s': %s", key, e)
            return False

    async def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        if not self._client:
            return False

        try:
            await self._client.delete(key)
            return True
        except Exception as e:
            logger.warning("Redis DELETE error for key '%s': %s", key, e)
            return False

    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        if not self._client:
            return False

        try:
            return bool(await self._client.exists(key))
        except Exception:
            return False

    # ── List operations (for conversation memory) ────────

    async def lpush(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Push a value to the head of a list."""
        if not self._client:
            return False

        try:
            serialized = json.dumps(value, default=str)
            await self._client.lpush(key, serialized)
            if ttl:
                await self._client.expire(key, ttl)
            return True
        except Exception as e:
            logger.warning("Redis LPUSH error for key '%s': %s", key, e)
            return False

    async def lrange(self, key: str, start: int = 0, end: int = -1) -> list:
        """Get a range of elements from a list."""
        if not self._client:
            return []

        try:
            values = await self._client.lrange(key, start, end)
            return [json.loads(v) for v in values]
        except Exception as e:
            logger.warning("Redis LRANGE error for key '%s': %s", key, e)
            return []

    async def ltrim(self, key: str, start: int, end: int) -> bool:
        """Trim a list to the specified range."""
        if not self._client:
            return False

        try:
            await self._client.ltrim(key, start, end)
            return True
        except Exception as e:
            logger.warning("Redis LTRIM error for key '%s': %s", key, e)
            return False

    @staticmethod
    def make_key(*parts: str) -> str:
        """
        Build a namespaced cache key.

        Example: make_key("transcript", "abc123") -> "transcript:abc123"
        """
        return ":".join(parts)

    @staticmethod
    def hash_url(url: str) -> str:
        """
        Create a deterministic hash of a URL for use as cache key.

        Why hash instead of raw URL:
        - URLs can be very long (exceeding Redis key limits)
        - Normalizes different URL formats for the same video
        """
        return hashlib.sha256(url.strip().encode()).hexdigest()[:16]


# Singleton instance
redis_cache = RedisCache()
