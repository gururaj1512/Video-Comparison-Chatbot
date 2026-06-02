"""Video processing request/response schemas."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator

from backend.utils.url_parser import validate_video_url, Platform


class ProcessVideosRequest(BaseModel):
    """Request body for POST /api/videos/process."""

    youtube_url: str = Field(
        ...,
        description="YouTube video URL",
        json_schema_extra={"examples": ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"]},
    )
    instagram_url: str = Field(
        ...,
        description="Instagram Reel URL",
        json_schema_extra={"examples": ["https://www.instagram.com/reels/DY6pGBLMfbe/"]},
    )

    @field_validator("youtube_url")
    @classmethod
    def validate_youtube(cls, v: str) -> str:
        if not validate_video_url(v, Platform.YOUTUBE):
            raise ValueError(
                f"Invalid YouTube URL: {v}. Expected format: https://www.youtube.com/watch?v=VIDEO_ID"
            )
        return v

    @field_validator("instagram_url")
    @classmethod
    def validate_instagram(cls, v: str) -> str:
        if not validate_video_url(v, Platform.INSTAGRAM):
            raise ValueError(
                f"Invalid Instagram URL: {v}. Expected format: https://www.instagram.com/reel/VIDEO_ID/"
            )
        return v


class ProcessVideosResponse(BaseModel):
    """Response for POST /api/videos/process."""

    status: str  # "processing", "completed", "failed"
    session_id: str
    job_id: Optional[str] = None
    message: Optional[str] = None


class VideoStatusResponse(BaseModel):
    """Response for GET /api/videos/{session_id}/status."""

    session_id: str
    status: str  # "pending", "processing", "completed", "failed"
    job_id: Optional[str] = None
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class VideoMetadataResponse(BaseModel):
    """Response for GET /api/videos/{session_id}/metadata."""

    session_id: str
    youtube: Optional[Dict[str, Any]] = None
    instagram: Optional[Dict[str, Any]] = None
