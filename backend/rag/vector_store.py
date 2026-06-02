"""
Qdrant vector store operations.

Ported from research/4vector_database.ipynb with production hardening:
- Support for both Qdrant Cloud (free tier) and local Docker
- Per-session collections for data isolation
- Batch upsert with proper payload serialization
- Metadata-filtered search (by video_id, source)

Why Qdrant over ChromaDB/Pinecone/pgvector:
- Qdrant Cloud has a generous free tier (1GB, 1 cluster)
- Native metadata filtering (critical for "show me only Video A" queries)
- Python-native client with async support
- Cosine similarity is built-in and optimized

Why per-session collections:
- Clean data isolation between different video comparison sessions
- Easy cleanup (delete collection = delete all data for a session)
- No cross-session data leakage in search results
"""

import uuid
from typing import List, Dict, Optional
from dataclasses import dataclass

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
)

from backend.config import settings
from backend.utils.logger import get_logger
from backend.utils.constants import COLLECTION_PREFIX

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """A chunk retrieved from vector search with relevance score."""

    point_id: int
    score: float
    text: str
    video_id: str
    video_label: str
    source: str
    chunk_id: int
    timestamp_start: float = 0.0
    timestamp_end: float = 0.0
    payload: Dict = None

    def __post_init__(self):
        if self.payload is None:
            self.payload = {}

    @property
    def citation(self) -> str:
        """Format as citation string for LLM output."""
        return f"[{self.video_label} - Chunk {self.chunk_id}]"


class VectorStore:
    """Qdrant vector database operations."""

    def __init__(self):
        self._client: Optional[QdrantClient] = None

    def connect(self) -> None:
        """
        Initialize Qdrant connection.
        Called once during FastAPI lifespan startup.
        """
        try:
            if settings.qdrant_api_key:
                # Qdrant Cloud
                logger.info("Connecting to Qdrant Cloud at %s", settings.qdrant_url)
                self._client = QdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key,
                )
            else:
                # Local Qdrant (Docker or in-memory)
                if "localhost" in settings.qdrant_url or "127.0.0.1" in settings.qdrant_url:
                    logger.info("Connecting to local Qdrant at %s", settings.qdrant_url)
                    self._client = QdrantClient(url=settings.qdrant_url)
                else:
                    logger.info("Using in-memory Qdrant (no URL/key configured)")
                    self._client = QdrantClient(":memory:")

            logger.info("Qdrant connected successfully")

        except Exception as e:
            logger.error("Failed to connect to Qdrant: %s", e)
            # Fallback to in-memory for development
            logger.info("Falling back to in-memory Qdrant")
            self._client = QdrantClient(":memory:")

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            raise RuntimeError("Qdrant not connected. Call connect() first.")
        return self._client

    # Collection Management

    def get_collection_name(self, session_id: str) -> str:
        """Build collection name from session ID."""
        return f"{COLLECTION_PREFIX}_{session_id}"

    def create_collection(self, session_id: str) -> bool:
        """Create a Qdrant collection for a session."""
        collection_name = self.get_collection_name(session_id)

        try:
            # Check if exists
            try:
                self.client.get_collection(collection_name)
                logger.info("Collection '%s' already exists", collection_name)
                return True
            except Exception:
                pass

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=settings.embedding_dimension,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(
                "Created collection '%s' (dim=%d, cosine)",
                collection_name,
                settings.embedding_dimension,
            )
            return True

        except Exception as e:
            logger.error("Failed to create collection '%s': %s", collection_name, e)
            return False

    def delete_collection(self, session_id: str) -> bool:
        """Delete a session's collection (cleanup)."""
        collection_name = self.get_collection_name(session_id)
        try:
            self.client.delete_collection(collection_name)
            logger.info("Deleted collection '%s'", collection_name)
            return True
        except Exception as e:
            logger.warning("Failed to delete collection '%s': %s", collection_name, e)
            return False

    # Upsert
    def upsert_chunks(
        self,
        session_id: str,
        embeddings: List[np.ndarray],
        payloads: List[Dict],
    ) -> bool:
        """
        Batch upsert embedded chunks into a session's collection.

        Args:
            session_id: Session identifier
            embeddings: List of embedding vectors
            payloads: List of payload dicts (from TextChunk.to_payload())

        Returns:
            True if successful
        """
        collection_name = self.get_collection_name(session_id)

        try:
            points = []
            for idx, (embedding, payload) in enumerate(zip(embeddings, payloads)):
                point = PointStruct(
                    id=idx + 1,  # Qdrant IDs are 1-indexed
                    vector=embedding.tolist(),
                    payload=payload,
                )
                points.append(point)

            # Batch upsert (Qdrant handles batching internally)
            self.client.upsert(
                collection_name=collection_name,
                points=points,
            )

            logger.info(
                "Upserted %d vectors to collection '%s'",
                len(points),
                collection_name,
            )
            return True

        except Exception as e:
            logger.error("Upsert failed for '%s': %s", collection_name, e)
            return False

    # Search
    def search(
        self,
        session_id: str,
        query_vector: np.ndarray,
        top_k: int = None,
        video_ids: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        score_threshold: float = None,
    ) -> List[RetrievedChunk]:
        """
        Similarity search with optional metadata filtering.

        Args:
            session_id: Session identifier
            query_vector: Query embedding
            top_k: Number of results (default from settings)
            video_ids: Filter by specific video IDs
            sources: Filter by platform ('youtube', 'instagram')
            score_threshold: Minimum similarity score

        Returns:
            List of RetrievedChunk objects sorted by relevance
        """
        collection_name = self.get_collection_name(session_id)
        top_k = top_k or settings.top_k
        score_threshold = score_threshold or settings.similarity_threshold

        # Build filter
        query_filter = self._build_filter(video_ids, sources)

        try:
            # Use the appropriate search method based on client version
            results = self.client.query_points(
                collection_name=collection_name,
                query=query_vector.tolist(),
                query_filter=query_filter,
                limit=top_k,
                score_threshold=score_threshold,
            )

            # Extract points from result
            points = self._extract_points(results)

            retrieved = []
            for point in points:
                payload = point.payload or {}
                retrieved.append(
                    RetrievedChunk(
                        point_id=point.id,
                        score=point.score,
                        text=payload.get("text", ""),
                        video_id=payload.get("video_id", ""),
                        video_label=payload.get("video_label", ""),
                        source=payload.get("source", ""),
                        chunk_id=payload.get("chunk_id", -1),
                        timestamp_start=payload.get("timestamp_start", 0.0),
                        timestamp_end=payload.get("timestamp_end", 0.0),
                        payload=payload,
                    )
                )

            logger.info(
                "Search returned %d results from '%s'",
                len(retrieved),
                collection_name,
            )
            return retrieved

        except Exception as e:
            logger.error("Search failed for '%s': %s", collection_name, e)
            return []

    # Helpers
    @staticmethod
    def _build_filter(
        video_ids: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
    ) -> Optional[Filter]:
        """Build Qdrant metadata filter from optional criteria."""
        conditions = []

        if video_ids:
            conditions.append(
                FieldCondition(
                    key="video_id",
                    match=MatchAny(any=video_ids),
                )
            )

        if sources:
            conditions.append(
                FieldCondition(
                    key="source",
                    match=MatchAny(any=sources),
                )
            )

        if not conditions:
            return None

        return Filter(must=conditions)

    @staticmethod
    def _extract_points(raw_results):
        """Extract points list from various Qdrant response formats."""
        if raw_results is None:
            return []
        if isinstance(raw_results, list):
            return raw_results
        if hasattr(raw_results, "points"):
            return raw_results.points
        if hasattr(raw_results, "result"):
            return raw_results.result
        return []


# Singleton
vector_store = VectorStore()
