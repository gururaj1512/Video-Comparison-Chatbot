"""
Redis-backed conversation memory for multi-turn chat.

Why Redis over in-memory:
- Survives server restarts (ARQ worker or uvicorn reload)
- Shared across multiple uvicorn workers
- TTL-based auto-expiration (sessions clean up automatically)
- Minimal memory footprint (only stores text, not embeddings)

Why store last N messages (not unlimited):
- LLM context window is finite (Llama 8B: 128K tokens, but prompt + context eats most of it)
- Older messages become less relevant for comparison queries
- 10 messages ≈ 2000 tokens of history — manageable
"""

from typing import List, Dict, Optional

from backend.cache.redis_cache import redis_cache, RedisCache
from backend.config import settings
from backend.utils.logger import get_logger
from backend.utils.constants import REDIS_PREFIX_MEMORY

logger = get_logger(__name__)

# Max messages to keep in history per session
MAX_HISTORY_MESSAGES = 20


class ConversationMemory:
    """Redis-backed conversation history for RAG chat sessions."""

    def _key(self, session_id: str) -> str:
        """Build Redis key for a session's memory."""
        return RedisCache.make_key(REDIS_PREFIX_MEMORY, session_id)

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        """
        Add a message to conversation history.

        Args:
            session_id: Chat session ID
            role: 'user' or 'assistant'
            content: Message text
        """
        key = self._key(session_id)
        message = {"role": role, "content": content}

        await redis_cache.lpush(key, message, ttl=settings.session_ttl)

        # Trim to max history length
        await redis_cache.ltrim(key, 0, MAX_HISTORY_MESSAGES - 1)

    async def get_history(
        self,
        session_id: str,
        limit: int = 10,
    ) -> List[Dict]:
        """
        Get recent conversation history.

        Args:
            session_id: Chat session ID
            limit: Max messages to return

        Returns:
            List of {"role": str, "content": str} dicts, oldest first
        """
        key = self._key(session_id)
        messages = await redis_cache.lrange(key, 0, limit - 1)

        # Redis LPUSH stores newest first; reverse for chronological order
        messages.reverse()
        return messages

    async def format_history(
        self,
        session_id: str,
        limit: int = 6,
    ) -> str:
        """
        Format conversation history as a string for the LLM prompt.

        Args:
            session_id: Chat session ID
            limit: Max messages to include

        Returns:
            Formatted string like "User: ...\nAssistant: ..."
        """
        messages = await self.get_history(session_id, limit=limit)

        if not messages:
            return "No previous conversation."

        parts = []
        for msg in messages:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            # Truncate long messages to save tokens
            if len(content) > 500:
                content = content[:500] + "..."
            parts.append(f"{role}: {content}")

        return "\n".join(parts)

    async def clear(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        key = self._key(session_id)
        await redis_cache.delete(key)
        logger.info("Cleared memory for session %s", session_id)


# Singleton
memory = ConversationMemory()
