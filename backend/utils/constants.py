"""
Application-wide constants.

Centralized here to avoid magic strings/numbers scattered across modules.
"""

# Video labels used in chat citations
VIDEO_A_LABEL = "Video A"
VIDEO_B_LABEL = "Video B"

# Qdrant collection prefix
COLLECTION_PREFIX = "session"

# Redis key prefixes
REDIS_PREFIX_TRANSCRIPT = "transcript"
REDIS_PREFIX_METADATA = "metadata"
REDIS_PREFIX_SESSION = "session"
REDIS_PREFIX_JOB = "job"
REDIS_PREFIX_MEMORY = "memory"

# Job statuses
JOB_STATUS_PENDING = "pending"
JOB_STATUS_PROCESSING = "processing"
JOB_STATUS_COMPLETED = "completed"
JOB_STATUS_FAILED = "failed"

# Whisper supported languages
SUPPORTED_LANGUAGES = ["en", "es", "fr", "de", "hi", "ja", "ko", "zh"]

# Max concurrent video processing
MAX_CONCURRENT_EXTRACTIONS = 2

# Default system prompt for RAG chain
SYSTEM_PROMPT = """You are an expert social media analyst and content strategist. You compare videos to help creators improve their content.

You have access to transcripts and metadata from the following videos:
{video_list_context}

## Video Metadata
{metadata_context}

## Rules
1. Base ALL answers on the provided transcript chunks and metadata. Do NOT make up information.
2. When referencing specific content, ALWAYS cite your source using the format [Video X - Chunk Y] where X is the video label (e.g. A, B, C, etc.).
3. Be specific and data-driven. Use actual numbers (engagement rates, views, likes) when available.
4. When comparing hooks, analyze the first few transcript chunks (Chunk 0, Chunk 1) which correspond to the opening seconds.
5. Provide actionable, constructive feedback when asked for suggestions.
6. If asked about data not in the provided context, say so clearly.

## Retrieved Transcript Chunks
{context}
"""

USER_PROMPT_TEMPLATE = """Conversation History:
{chat_history}

User Question: {question}

Provide a comprehensive, data-driven answer. Cite sources using [Video A/B - Chunk #] format."""
