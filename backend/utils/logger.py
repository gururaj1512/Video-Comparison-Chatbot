"""
Structured logging configuration.

Why structured logging over print():
- Log levels (DEBUG/INFO/WARNING/ERROR) for filtering
- Consistent format with timestamps and module names
- Easy to pipe to log aggregators (ELK, Datadog) in production
- Thread-safe (print is not guaranteed thread-safe with concurrent I/O)
"""

import logging
import sys
from backend.config import settings


def get_logger(name: str) -> logging.Logger:
    """
    Create a configured logger instance.

    Args:
        name: Logger name (typically __name__ of the calling module)

    Returns:
        Configured Logger instance
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

        # Console handler with structured format
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Prevent duplicate logs from propagation
        logger.propagate = False

    return logger
