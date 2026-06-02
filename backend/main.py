"""
FastAPI application entry point.

Run with: uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
Or:       python -m backend.main

This module:
1. Creates the FastAPI app with metadata
2. Registers lifespan events (startup/shutdown)
3. Adds middleware (CORS, error handling, logging)
4. Includes all API route groups
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.cache.redis_cache import redis_cache
from backend.rag.embedding_service import embedding_service
from backend.rag.vector_store import vector_store
from backend.rag.chain import rag_chain
from backend.api.middleware.error_handler import ErrorHandlerMiddleware
from backend.api.middleware.logging_middleware import LoggingMiddleware
from backend.api.routes import health, videos, chat, sessions
from backend.utils.logger import get_logger

logger = get_logger(__name__)


# Lifespan (startup/shutdown)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan events.

    Startup: Initialize all services (Redis, Qdrant, embedding model, LLM)
    Shutdown: Clean up connections
    """
    logger.info("=" * 60)
    logger.info("Starting RAG Video Comparison Chatbot")
    logger.info("=" * 60)

    # 1. Connect to Redis
    await redis_cache.connect()

    # 2. Load embedding model (this takes 2-5 seconds)
    try:
        embedding_service.load_model()
    except Exception as e:
        logger.error("Failed to load embedding model: %s", e)

    # 3. Connect to Qdrant
    try:
        vector_store.connect()
    except Exception as e:
        logger.error("Failed to connect to Qdrant: %s", e)

    # 4. Initialize LLM
    try:
        rag_chain.init_llm()
    except Exception as e:
        logger.error("Failed to initialize LLM: %s", e)

    # Ensure temp directory exists
    os.makedirs(settings.temp_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("All services initialized. Server ready.")
    logger.info("Docs available at: http://%s:%d/docs", settings.app_host, settings.app_port)
    logger.info("=" * 60)

    yield  # App is running

    # Shutdown
    logger.info("Shutting down...")
    await redis_cache.disconnect()
    logger.info("Shutdown complete.")


# FastAPI App
app = FastAPI(
    title="RAG Video Comparison Chatbot",
    description=(
        "A full-stack RAG chatbot that compares YouTube and Instagram videos. "
        "Uses LangChain, Qdrant, BGE embeddings, and Groq-hosted Llama 3.1."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# Middleware

# Error handling (outermost)
app.add_middleware(ErrorHandlerMiddleware)

# Request logging
app.add_middleware(LoggingMiddleware)

# CORS (allow frontend dev server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Routes
app.include_router(health.router, prefix="/api")
app.include_router(videos.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(sessions.router, prefix="/api")


# Root
@app.get("/")
async def root():
    """Root endpoint — API info."""
    return {
        "name": "RAG Video Comparison Chatbot",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }


# CLI entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
    )
