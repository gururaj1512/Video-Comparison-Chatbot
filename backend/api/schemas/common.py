"""Shared response models and common schemas."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class APIResponse(BaseModel):
    """Standard API response envelope."""
    status: str  # "success", "error", "processing"
    message: Optional[str] = None
    data: Optional[Any] = None


class ErrorResponse(BaseModel):
    """Error response."""
    status: str = "error"
    error_type: str
    message: str
    details: Optional[Dict] = None
