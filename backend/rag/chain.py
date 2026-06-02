"""
LangChain RAG chain with streaming, citations, and conversation memory.

This is the core intelligence of the chatbot. It:
1. Retrieves relevant transcript chunks via the Retriever
2. Formats them with citation tags
3. Includes conversation history for multi-turn coherence
4. Streams the LLM response token-by-token via SSE
5. Extracts citations from the LLM output

Why LangChain over raw API calls:
- Provides streaming abstractions (.astream())
- Integrates with Groq's API via langchain-groq
- Memory and chain composition utilities

Why ChatGroq over ChatOllama:
- Groq free tier gives 300 tok/s inference (10-50x faster than local Ollama on CPU)
- No local GPU or 16GB RAM requirement
- 14,400 free req/day is sufficient for demo + moderate usage

Why temperature=0.3:
- Low enough for factual, data-driven responses
- High enough to avoid robotic/repetitive language
- Sweet spot for analytical comparison tasks
"""

import re
from typing import AsyncGenerator, Dict, List, Optional

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

from backend.config import settings
from backend.rag.retriever import retriever, Retriever
from backend.rag.memory import memory
from backend.rag.vector_store import RetrievedChunk
from backend.cache.redis_cache import redis_cache, RedisCache
from backend.utils.logger import get_logger
from backend.utils.constants import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, REDIS_PREFIX_SESSION

logger = get_logger(__name__)


class RAGChain:
    """
    LangChain-based RAG chain for video comparison chat.

    Flow per query:
    1. Retrieve relevant chunks from vector store
    2. Format context with citation tags
    3. Load conversation history from Redis
    4. Build prompt (system + context + history + question)
    5. Stream response from Groq LLM
    6. Extract and return citations
    7. Save messages to memory
    """

    def __init__(self):
        self._llm: Optional[ChatGroq] = None

    def init_llm(self) -> None:
        """
        Initialize the LLM client.
        Called during FastAPI lifespan startup.
        """
        if not settings.groq_api_key:
            logger.warning("GROQ_API_KEY not set — LLM will not work")
            return

        self._llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            streaming=True,
        )
        logger.info("LLM initialized: %s (temp=%.1f)", settings.llm_model, settings.llm_temperature)

    # Main Chat Method (Streaming)
    async def chat_stream(
        self,
        question: str,
        session_id: str,
        metadata_context: str = "",
        video_a_label: str = "Video A",
        video_b_label: str = "Video B",
        video_a_platform: str = "youtube",
        video_b_platform: str = "instagram",
        video_ids: Optional[List[str]] = None,
    ) -> AsyncGenerator[Dict, None]:
        """
        Stream a RAG-powered response.

        Yields dicts with either:
        - {"type": "token", "content": "..."} for each LLM token
        - {"type": "done", "citations": [...], "chunks_used": [...]} at the end
        - {"type": "error", "message": "..."} on failure

        Args:
            question: User's question
            session_id: Chat session ID
            metadata_context: Pre-formatted metadata string
            video_a/b_label: Display labels
            video_a/b_platform: Platform names
            video_ids: Optional video ID filter
        """
        if self._llm is None:
            yield {"type": "error", "message": "LLM not initialized. Set GROQ_API_KEY."}
            return

        try:
            # Step 1: Retrieve relevant chunks
            chunks = await retriever.retrieve(
                query=question,
                session_id=session_id,
                video_ids=video_ids,
            )

            # Step 2: Format context
            context = retriever.format_context(chunks)

            # Step 3: Get conversation history
            chat_history = await memory.format_history(session_id, limit=6)

            # Step 4: Fetch session details for dynamic prompt labeling
            session_key = RedisCache.make_key(REDIS_PREFIX_SESSION, session_id)
            session_data = await redis_cache.get(session_key)

            video_list_context_parts = []
            if session_data and "videos" in session_data:
                for v in session_data["videos"]:
                    title = v.get("metadata", {}).get("title") if v.get("metadata") else "N/A"
                    video_list_context_parts.append(
                        f"- **{v['label']}** ({v['platform']}): {title}"
                    )
            else:
                video_list_context_parts = [
                    f"- **{video_a_label}** ({video_a_platform})",
                    f"- **{video_b_label}** ({video_b_platform})"
                ]
            video_list_context = "\n".join(video_list_context_parts)

            # Step 5: Build messages
            system_msg = SYSTEM_PROMPT.format(
                video_list_context=video_list_context,
                metadata_context=metadata_context,
                context=context,
            )

            user_msg = USER_PROMPT_TEMPLATE.format(
                chat_history=chat_history,
                question=question,
            )

            messages = [
                SystemMessage(content=system_msg),
                HumanMessage(content=user_msg),
            ]

            # Step 5: Stream LLM response
            full_response = ""

            async for chunk in self._llm.astream(messages):
                token = chunk.content
                if token:
                    full_response += token
                    yield {"type": "token", "content": token}

            # Step 6: Extract citations from response
            citations = self._extract_citations(full_response)

            # Step 7: Save to memory
            await memory.add_message(session_id, "user", question)
            await memory.add_message(session_id, "assistant", full_response)

            # Step 8: Final event with metadata
            chunks_used = [
                {
                    "video_label": c.video_label,
                    "chunk_id": c.chunk_id,
                    "video_id": c.video_id,
                    "source": c.source,
                    "score": round(c.score, 4),
                    "text_preview": c.text[:100] + "..." if len(c.text) > 100 else c.text,
                    "timestamp_start": c.timestamp_start,
                    "timestamp_end": c.timestamp_end,
                }
                for c in chunks
            ]

            yield {
                "type": "done",
                "citations": citations,
                "chunks_used": chunks_used,
                "total_chunks_retrieved": len(chunks),
            }

        except Exception as e:
            logger.error("RAG chain error: %s", e, exc_info=True)
            yield {"type": "error", "message": f"Chat error: {str(e)}"}

    # Non-Streaming Chat (for testing)
    async def chat(
        self,
        question: str,
        session_id: str,
        metadata_context: str = "",
        **kwargs,
    ) -> Dict:
        """Non-streaming chat — collects all tokens and returns complete response."""
        full_response = ""
        final_event = {}

        async for event in self.chat_stream(
            question=question,
            session_id=session_id,
            metadata_context=metadata_context,
            **kwargs,
        ):
            if event["type"] == "token":
                full_response += event["content"]
            elif event["type"] == "done":
                final_event = event
            elif event["type"] == "error":
                return {"status": "error", "message": event["message"]}

        return {
            "status": "success",
            "response": full_response,
            "citations": final_event.get("citations", []),
            "chunks_used": final_event.get("chunks_used", []),
        }

    # Citation Extraction
    @staticmethod
    def _extract_citations(text: str) -> List[Dict]:
        """
        Extract citation references from LLM output.

        Looks for patterns like [Video A - Chunk 3], [Video B - Chunk 0], etc.

        Returns:
            List of {"video_label": str, "chunk_id": int} dicts
        """
        pattern = r"\[(Video [A-Z])\s*-\s*Chunk\s*(\d+)\]"
        matches = re.findall(pattern, text)

        citations = []
        seen = set()
        for video_label, chunk_id in matches:
            key = (video_label, int(chunk_id))
            if key not in seen:
                citations.append({
                    "video_label": video_label,
                    "chunk_id": int(chunk_id),
                })
                seen.add(key)

        return citations


# Singleton
rag_chain = RAGChain()
