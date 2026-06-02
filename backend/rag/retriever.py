"""
RAG retriever — combines embedding + vector search into a single call.

This is the bridge between user questions and relevant transcript chunks.
It embeds the query, searches Qdrant, and returns formatted results
ready for the LLM chain.

Why a separate retriever module:
- Decouples embedding logic from vector store logic
- Single entry point for "question → relevant chunks"
- Easy to add reranking, hybrid search, or query expansion later
"""

import asyncio
from typing import List, Optional

from backend.rag.embedding_service import embedding_service
from backend.rag.vector_store import vector_store, RetrievedChunk
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class Retriever:
    """Retrieve relevant transcript chunks for a user query."""

    async def retrieve(
        self,
        query: str,
        session_id: str,
        top_k: int = None,
        video_ids: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        """
        Full retrieval pipeline: embed query → search vectors → return chunks.

        Args:
            query: User's question
            session_id: Session ID for collection lookup
            top_k: Number of chunks to retrieve
            video_ids: Optional filter by video IDs
            sources: Optional filter by platform

        Returns:
            List of relevant chunks sorted by similarity score
        """
        # Step 1: Embed the query
        query_embedding = await embedding_service.aembed_text(query)

        # Step 2: Search Qdrant (sync, but fast <100ms)
        chunks = await asyncio.to_thread(
            vector_store.search,
            session_id=session_id,
            query_vector=query_embedding,
            top_k=top_k,
            video_ids=video_ids,
            sources=sources,
        )

        logger.info(
            "Retrieved %d chunks for query: '%s...' (session=%s)",
            len(chunks),
            query[:50],
            session_id,
        )

        return chunks

    def format_context(self, chunks: List[RetrievedChunk]) -> str:
        """
        Format retrieved chunks into a context string for the LLM prompt.

        Each chunk is labeled with its citation tag so the LLM knows
        how to reference it in the response.
        """
        if not chunks:
            return "No relevant transcript chunks found."

        context_parts = []
        for chunk in chunks:
            label = chunk.video_label or chunk.video_id
            ts = ""
            if chunk.timestamp_start > 0 or chunk.timestamp_end > 0:
                ts = f" (timestamp: {chunk.timestamp_start:.1f}s - {chunk.timestamp_end:.1f}s)"

            context_parts.append(
                f"[{label} - Chunk {chunk.chunk_id}]{ts}:\n{chunk.text}"
            )

        return "\n\n---\n\n".join(context_parts)


# Singleton
retriever = Retriever()
