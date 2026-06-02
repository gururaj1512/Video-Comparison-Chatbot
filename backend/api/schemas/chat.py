"""Chat request/response schemas."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request body for POST /api/chat."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User's question about the videos",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID from video processing. If omitted, will try to auto-detect from URLs in question or fallback to last active session.",
    )
    video_filter: Optional[str] = Field(
        default=None,
        description="Filter to specific video: 'A', 'B', etc.",
    )


class ChatMessage(BaseModel):
    """A single chat message."""

    role: str  # "user" or "assistant"
    content: str


class ChatResponse(BaseModel):
    """Non-streaming chat response."""

    status: str
    response: str = ""
    citations: List[Dict] = Field(default_factory=list)
    chunks_used: List[Dict] = Field(default_factory=list)


class ChatHistoryResponse(BaseModel):
    """Response for GET /api/sessions/{session_id}/history."""

    session_id: str
    messages: List[ChatMessage] = Field(default_factory=list)
