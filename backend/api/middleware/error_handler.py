"""Global error handling middleware."""

import traceback
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from backend.utils.logger import get_logger

logger = get_logger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Catch unhandled exceptions and return structured JSON errors."""

    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            logger.error(
                "Unhandled exception on %s %s: %s",
                request.method,
                request.url.path,
                e,
                exc_info=True,
            )
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "error_type": type(e).__name__,
                    "message": str(e),
                },
            )
