"""Request/response logging middleware."""

import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from backend.utils.logger import get_logger

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests with method, path, status, and latency."""

    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        elapsed = (time.time() - start) * 1000  # ms

        # Skip health check spam
        if request.url.path != "/api/health":
            logger.info(
                "%s %s → %d (%.1fms)",
                request.method,
                request.url.path,
                response.status_code,
                elapsed,
            )

        return response
