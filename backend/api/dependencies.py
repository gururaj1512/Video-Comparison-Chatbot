"""
FastAPI dependency injection.

Provides shared instances to route handlers via Depends().
"""

from backend.cache.redis_cache import redis_cache, RedisCache
from backend.rag.embedding_service import embedding_service, EmbeddingService
from backend.rag.vector_store import vector_store, VectorStore
from backend.rag.chain import rag_chain, RAGChain


def get_redis() -> RedisCache:
    """Get Redis cache instance."""
    return redis_cache


def get_embedding_service() -> EmbeddingService:
    """Get embedding service instance."""
    return embedding_service


def get_vector_store() -> VectorStore:
    """Get vector store instance."""
    return vector_store


def get_rag_chain() -> RAGChain:
    """Get RAG chain instance."""
    return rag_chain
