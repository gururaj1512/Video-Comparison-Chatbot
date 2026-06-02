"""
Session management API routes.

Endpoints:
- GET    /api/sessions/{session_id} — Get full session state
- DELETE /api/sessions/{session_id} — Cleanup session data
- GET    /api/sessions/{session_id}/history — Get chat history
"""

from fastapi import APIRouter, HTTPException

from backend.api.schemas.chat import ChatHistoryResponse, ChatMessage
from backend.rag.memory import memory
from backend.rag.vector_store import vector_store
from backend.cache.redis_cache import redis_cache, RedisCache
from backend.utils.constants import REDIS_PREFIX_SESSION, REDIS_PREFIX_MEMORY
from backend.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/sessions", tags=["Sessions"])


@router.get("/{session_id}")
async def get_session(session_id: str):
    """Get full session state including metadata and chat history."""
    session_key = RedisCache.make_key(REDIS_PREFIX_SESSION, session_id)
    session_data = await redis_cache.get(session_key)

    if not session_data:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found.",
        )

    # Get chat history
    history = await memory.get_history(session_id, limit=50)

    return {
        "session_id": session_id,
        "status": "active",
        "metadata": session_data,
        "chat_history": history,
        "message_count": len(history),
    }


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """
    Delete all data for a session:
    - Vector store collection
    - Redis session cache
    - Conversation memory
    """
    # Delete Qdrant collection
    vector_store.delete_collection(session_id)

    # Delete Redis keys
    session_key = RedisCache.make_key(REDIS_PREFIX_SESSION, session_id)
    memory_key = RedisCache.make_key(REDIS_PREFIX_MEMORY, session_id)

    await redis_cache.delete(session_key)
    await redis_cache.delete(memory_key)

    logger.info("Deleted session %s", session_id)

    return {
        "status": "success",
        "message": f"Session '{session_id}' deleted.",
    }


@router.get("/{session_id}/history", response_model=ChatHistoryResponse)
async def get_chat_history(session_id: str):
    """Get conversation history for a session."""
    history = await memory.get_history(session_id, limit=50)

    return ChatHistoryResponse(
        session_id=session_id,
        messages=[ChatMessage(**msg) for msg in history],
    )
