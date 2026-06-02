"""
Embedding service using sentence-transformers (local, open-source).

Ported from research/3chunking_embedding.ipynb with production hardening:
- Singleton model loading (heavy model loaded once at startup)
- Thread-safe batch processing (model inference on thread pool)
- Normalized embeddings for cosine similarity

Why BAAI/bge-small-en-v1.5:
- #1 on MTEB leaderboard for its size class (33M params)
- 384-dimensional output (compact storage, fast search)
- Runs on CPU in <300ms per batch of 32
- $0 cost (no API calls)

Why NOT OpenAI text-embedding-3-small:
- $0.02 per million tokens → at scale, adds up
- Network latency per call
- Vendor lock-in and rate limits

Why normalize embeddings:
- Cosine similarity = dot product for unit vectors → faster search in Qdrant
- BGE model documentation recommends normalized embeddings
"""

import asyncio
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.config import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """Generate embeddings using a local SentenceTransformer model."""

    def __init__(self):
        self._model: SentenceTransformer = None
        self._dimension: int = settings.embedding_dimension

    def load_model(self) -> None:
        """
        Load the embedding model into memory.
        Called once during FastAPI lifespan startup.

        Why load at startup (not per-request):
        - Model loading takes 2-5 seconds
        - Model occupies ~130MB in RAM — loading per-request would thrash memory
        - Singleton pattern ensures one copy in memory
        """
        if self._model is not None:
            logger.info("Embedding model already loaded")
            return

        logger.info("Loading embedding model: %s", settings.embedding_model)
        self._model = SentenceTransformer(settings.embedding_model)

        # Detect dimension dynamically
        if hasattr(self._model, "get_embedding_dimension"):
            self._dimension = self._model.get_embedding_dimension()
        else:
            self._dimension = self._model.get_sentence_embedding_dimension()

        logger.info(
            "Embedding model loaded — dim=%d, max_seq_len=%d",
            self._dimension,
            self._model.max_seq_length,
        )

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        return self._dimension

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text (synchronous).

        Args:
            text: Text to embed

        Returns:
            Normalized embedding vector of shape (dimension,)
        """
        if self._model is None:
            raise RuntimeError("Embedding model not loaded. Call load_model() first.")

        return self._model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts (synchronous, batched).

        Args:
            texts: List of texts to embed
            batch_size: Batch size for GPU/CPU processing

        Returns:
            List of normalized embedding vectors
        """
        if self._model is None:
            raise RuntimeError("Embedding model not loaded. Call load_model() first.")

        if not texts:
            return []

        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )

        return [emb for emb in embeddings]

    async def aembed_text(self, text: str) -> np.ndarray:
        """Async wrapper — runs embedding on thread pool to avoid blocking event loop."""
        return await asyncio.to_thread(self.embed_text, text)

    async def aembed_batch(
        self, texts: List[str], batch_size: int = 32
    ) -> List[np.ndarray]:
        """Async wrapper for batch embedding."""
        return await asyncio.to_thread(self.embed_batch, texts, batch_size)


# Singleton
embedding_service = EmbeddingService()
