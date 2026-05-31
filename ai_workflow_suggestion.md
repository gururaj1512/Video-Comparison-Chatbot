# RAG Video Comparison Chatbot - Development Plan

## Executive Summary

This document outlines a phased, modular development approach for building a production-grade RAG chatbot that compares YouTube and Instagram Reel videos. Development follows a bottom-up strategy: research & testing first, then core services, then APIs, and finally frontend integration.

**Total Estimated Timeline**: 6-8 weeks (with parallel development)

---

## Phase 1: Research & Prototyping (Week 1)

### Objectives
- Validate all core technologies work locally
- Test transcript extraction methods
- Validate metadata extraction
- Prototype embedding and retrieval

### Tasks

#### 1.1 Transcript Extraction Testing
**Files**: `research/transcript_testing.ipynb`

- [ ] Test youtube-transcript-api
  - Sample YouTube videos
  - Handle edge cases (live streams, age-restricted)
  - Measure latency
  - Document fallback strategy

- [ ] Test yt-dlp + faster-whisper fallback
  - Download Instagram Reel samples
  - Test transcription quality
  - Measure compute time
  - Compare accuracy with YouTube transcripts

**Outputs**:
- Working code snippets for both methods
- Performance benchmarks
- Failure scenarios and handling

#### 1.2 Metadata Extraction Testing
**Files**: `research/metadata_testing.ipynb`

- [ ] YouTube metadata extraction (yt-dlp)
  - Extract: views, likes, comments, creator, followers, upload_date, duration, hashtags
  - Test pagination for high-view videos
  - Test channel info retrieval

- [ ] Instagram metadata extraction (yt-dlp)
  - Test if yt-dlp can extract Instagram Reel metadata
  - Identify what data is publicly available
  - Plan scraping fallback if needed

- [ ] Data validation schema
  - Create Pydantic models for metadata
  - Handle missing fields gracefully

**Outputs**:
- Pydantic models for metadata
- yt-dlp configuration templates
- Metadata extraction utilities

#### 1.3 Engagement Analytics Testing
**Files**: `research/metadata_testing.ipynb`

- [ ] Implement engagement rate formula: `(likes + comments) / views * 100`
- [ ] Test with real video data
- [ ] Add additional metrics (comment:like ratio, view velocity estimates)

**Outputs**:
- Analytics calculation functions
- Test data with expected results

#### 1.4 Embedding & Vector Storage Testing
**Files**: `research/embedding_testing.ipynb`

- [ ] Test BAAI/bge-small-en-v1.5 locally
  - Install sentence-transformers
  - Verify embedding dimensions (384)
  - Measure embedding time per chunk
  - Test batch embedding

- [ ] Test Qdrant local setup
  - Docker setup and verification
  - Schema design with metadata
  - Basic CRUD operations
  - Metadata filtering

- [ ] Test alternative embeddings (e5-small, nomic-embed-text)
  - Compare embedding quality/speed
  - Document trade-offs

**Outputs**:
- Embedding service prototype
- Qdrant schema design
- Performance benchmarks (latency, throughput)

#### 1.5 LLM Setup Testing
**Files**: `research/rag_testing.ipynb`

- [ ] Install and test Ollama locally
  - Model download (llama3:8b)
  - Temperature and parameter tuning
  - Response quality assessment
  - Latency measurements

- [ ] Test LangChain integration with Ollama
  - LLM wrapper setup
  - Streaming capability verification
  - Token counting

**Outputs**:
- Ollama configuration guide
- LangChain LLM wrapper code
- Parameter tuning recommendations

#### 1.6 RAG Pipeline Prototype
**Files**: `research/rag_testing.ipynb`

- [ ] End-to-end prototype
  - Sample transcript chunking
  - Embedding generation
  - Vector storage
  - Retrieval with metadata filtering
  - RAG response generation with citations

- [ ] Quality evaluation
  - Test relevance of retrieved chunks
  - Verify citation accuracy
  - Measure response latency

**Outputs**:
- Working RAG pipeline code
- Chunking strategy analysis
- Retrieved chunk quality metrics

---

## Phase 2: Core Backend Services (Week 2)

### Objectives
- Build production-ready service modules
- Implement async architecture
- Establish error handling patterns
- Create configuration management

### Tasks

#### 2.1 Project Setup & Configuration
**Directory**: `backend/`

- [ ] Poetry/pip dependency management
  - Create `pyproject.toml` or `requirements.txt`
  - Separate dev and prod dependencies
  - Version pinning for reproducibility

- [ ] Environment configuration
  - Create `.env.example`
  - Implement config.py with pydantic-settings
  - Support Docker and local modes

- [ ] Logging setup
  - Structured logging with Python logging
  - Log levels for development/production

**Key Dependencies**:
```
fastapi==0.104.1
pydantic==2.5.0
langchain==0.1.0
langchain-community==0.0.1
sentence-transformers==2.2.2
qdrant-client==2.7.0
yt-dlp==2023.12.30
youtube-transcript-api==0.6.1
faster-whisper==0.10.0
python-dotenv==1.0.0
aiofiles==23.2.1
```

#### 2.2 Transcript Extraction Service
**Directory**: `backend/services/transcript_service.py`

```python
class TranscriptExtractor:
    - extract_youtube_transcript(url: str) -> Transcript
    - extract_instagram_transcript(url: str) -> Transcript
    - _fallback_whisper_extraction(url: str) -> Transcript
```

**Features**:
- Retry logic with exponential backoff
- Caching of extracted transcripts
- Metadata about extraction source (API vs Whisper)
- Error classification and recovery

**Tests**: Unit tests with mock URLs

#### 2.3 Metadata Extraction Service
**Directory**: `backend/services/metadata_service.py`

```python
class MetadataExtractor:
    - extract_youtube_metadata(url: str) -> VideoMetadata
    - extract_instagram_metadata(url: str) -> VideoMetadata
    - calculate_engagement_metrics(metadata: VideoMetadata) -> EngagementMetrics
```

**Pydantic Models**:
```python
class VideoMetadata(BaseModel):
    video_id: str
    platform: Literal["youtube", "instagram"]
    title: str
    creator: str
    views: int
    likes: int
    comments: int
    followers: Optional[int]
    upload_date: datetime
    duration_seconds: int
    hashtags: List[str]
    thumbnail_url: Optional[str]
    url: str

class EngagementMetrics(BaseModel):
    engagement_rate: float
    comment_like_ratio: float
    views_per_day: Optional[float]
```

**Tests**: Unit tests with real and mocked data

#### 2.4 Transcript Chunking Service
**Directory**: `backend/rag/chunker.py`

```python
class TranscriptChunker:
    - chunk_transcript(transcript: Transcript, metadata: VideoMetadata) -> List[TextChunk]
    - _semantic_chunking(transcript: str) -> List[str]
    - _overlap_chunks(chunks: List[str], overlap: int) -> List[str]
```

**Strategy**:
- Sentence/paragraph-based chunking (500-1000 tokens per chunk)
- Overlap between chunks (50-100 tokens)
- Preserve timestamp information if available
- Metadata per chunk:
  ```json
  {
    "video_id": "abc123",
    "chunk_id": 1,
    "source": "youtube",
    "timestamp_start": 0,
    "timestamp_end": 45,
    "text": "...",
    "embedding": [...]
  }
  ```

**Tests**: Chunking quality validation

#### 2.5 Embedding Service
**Directory**: `backend/embeddings/embedding_service.py`

```python
class EmbeddingService:
    - embed_text(text: str) -> np.ndarray
    - embed_batch(texts: List[str]) -> List[np.ndarray]
    - get_embedding_dimension() -> int
```

**Implementation**:
- Use sentence-transformers locally
- Batch processing for efficiency
- Optional GPU acceleration detection
- Caching of embeddings

**Tests**: Embedding consistency and quality

#### 2.6 Vector Store Service
**Directory**: `backend/vectorstore/qdrant_service.py`

```python
class QdrantVectorStore:
    - init_collection(collection_name: str)
    - upsert_chunks(chunks: List[TextChunk])
    - retrieve_relevant_chunks(
        query_embedding: np.ndarray,
        collection_name: str,
        limit: int = 5,
        filters: Optional[Dict] = None
      ) -> List[RetrievedChunk]
    - delete_collection(collection_name: str)
```

**Features**:
- Automatic collection creation
- Metadata filtering (by video_id, source, etc.)
- Similarity search with configurable threshold
- Connection pooling

**Tests**: CRUD operations and retrieval quality

#### 2.7 LLM Service
**Directory**: `backend/llm/llm_service.py`

```python
class LLMService:
    - generate(prompt: str, **kwargs) -> str
    - stream_generate(prompt: str, **kwargs) -> Iterator[str]
    - get_token_count(text: str) -> int
```

**Implementation**:
- Ollama integration via LangChain
- Streaming support
- Temperature and top_p configuration
- Graceful fallback if Ollama unavailable

**Tests**: Response quality and streaming

#### 2.8 RAG Chain Service
**Directory**: `backend/rag/rag_chain.py`

```python
class RAGChain:
    - retrieve_context(query: str, video_ids: List[str]) -> List[RetrievedChunk]
    - generate_response(query: str, context: List[RetrievedChunk]) -> str
    - stream_response(query: str, context: List[RetrievedChunk]) -> Iterator[str]
    - create_prompt(query: str, context: List[RetrievedChunk]) -> str
```

**Prompt Template**:
```
You are a helpful assistant comparing social media videos.

Video Context:
{context}

User Question: {question}

Provide a comprehensive answer based only on the video content. 
Cite your sources using [Video ID - Chunk #].
```

**Tests**: Response quality and citation accuracy

---

## Phase 3: FastAPI Backend (Week 2-3)

### Objectives
- Implement REST APIs
- Handle async/concurrent requests
- Implement streaming responses
- Error handling and validation

### Tasks

#### 3.1 API Structure
**Directory**: `backend/api/`

```
api/
├── __init__.py
├── main.py (FastAPI app)
├── routes/
│   ├── videos.py (POST /process-videos)
│   ├── chat.py (POST /chat, WebSocket /ws/chat)
│   └── health.py (GET /health)
├── schemas/
│   ├── video_request.py
│   ├── chat_request.py
│   └── responses.py
└── dependencies.py (shared dependencies)
```

#### 3.2 Video Processing Endpoint
**Route**: `POST /api/videos/process`

```python
@router.post("/process", response_model=VideoProcessResponse)
async def process_videos(
    request: ProcessVideosRequest,
    transcript_service: TranscriptExtractor = Depends(),
    metadata_service: MetadataExtractor = Depends(),
    chunker: TranscriptChunker = Depends(),
    embedding_service: EmbeddingService = Depends(),
    vector_store: QdrantVectorStore = Depends()
) -> VideoProcessResponse:
    """
    Process YouTube and Instagram Reel URLs.
    
    Flow:
    1. Extract transcripts (youtube-transcript-api + fallback)
    2. Extract metadata (yt-dlp)
    3. Chunk transcripts
    4. Generate embeddings
    5. Store in Qdrant
    6. Return metadata and collection_id
    """
```

**Request Schema**:
```python
class ProcessVideosRequest(BaseModel):
    youtube_url: str
    instagram_url: str
    
class VideoProcessResponse(BaseModel):
    youtube: VideoInfo
    instagram: VideoInfo
    collection_id: str
    processing_time_seconds: float
    status: Literal["success", "partial", "failed"]
```

**Implementation Details**:
- Async execution of independent tasks (youtube + instagram in parallel)
- Progress tracking (optional: WebSocket for progress updates)
- Error handling with partial success support
- Caching to avoid reprocessing same URLs

#### 3.3 Chat Endpoint (Streaming)
**Route**: `POST /api/chat` (streaming)

```python
@router.post("/chat")
async def chat(
    request: ChatRequest,
    rag_chain: RAGChain = Depends(),
    memory_service: ChatMemory = Depends()
) -> StreamingResponse:
    """
    Streaming chat endpoint.
    
    Flow:
    1. Retrieve relevant chunks (RAG retrieval)
    2. Build prompt with context
    3. Stream LLM response
    4. Store in memory
    """
```

**Request Schema**:
```python
class ChatRequest(BaseModel):
    question: str
    collection_id: str
    session_id: Optional[str] = None
    video_ids: Optional[List[str]] = None  # filter by specific videos
```

**Response**: Server-Sent Events (SSE) stream
```
data: "token1"
data: "token2"
data: {"done": true, "citations": [...]}
```

**Implementation**:
- Token-by-token streaming
- Citation tracking
- Session management
- Error recovery

#### 3.4 Chat WebSocket Endpoint (Optional Enhancement)
**Route**: `WebSocket /api/ws/chat/{session_id}`

Benefits:
- Bidirectional communication
- Better for conversational continuity
- Real-time progress updates
- Client disconnection handling

#### 3.5 Metadata Endpoint
**Route**: `GET /api/videos/{collection_id}/metadata`

```python
@router.get("/videos/{collection_id}/metadata")
async def get_metadata(
    collection_id: str,
    vector_store: QdrantVectorStore = Depends()
) -> Dict[str, VideoMetadata]:
    """Get metadata for videos in collection"""
```

#### 3.6 Health & Status Endpoints
**Route**: `GET /api/health`

```python
@router.get("/health")
async def health_check() -> HealthStatus:
    """
    Check status of all dependencies:
    - Ollama availability
    - Qdrant connection
    - Disk space
    """
```

#### 3.7 Error Handling Middleware
**Directory**: `backend/api/middleware/`

```python
class ErrorHandlingMiddleware:
    - Handle TranscriptExtractionError
    - Handle MetadataExtractionError
    - Handle EmbeddingError
    - Handle VectorStoreError
    - Handle LLMError
    - Handle ValidationError
    - Provide meaningful error messages
```

#### 3.8 Request/Response Logging
**Directory**: `backend/api/middleware/`

- Log all requests with timestamps
- Log response times
- Log errors with full stack traces
- Support for correlation IDs (for tracing)

---

## Phase 4: Frontend (Next.js) (Week 3-4)

### Objectives
- Build responsive UI
- Implement real-time streaming
- Show engagement analytics
- Side-by-side video comparison

### Tasks

#### 4.1 Project Setup
**Directory**: `frontend/`

```bash
npx create-next-app@latest frontend \
  --typescript \
  --tailwind \
  --eslint
```

**Dependencies**:
```json
{
  "next": "14.0.0",
  "react": "18.2.0",
  "typescript": "5.3.0",
  "tailwindcss": "3.3.0",
  "@radix-ui/react-dialog": "latest",
  "shadcn/ui": "latest",
  "lucide-react": "latest",
  "axios": "latest"
}
```

#### 4.2 Layout & Navigation
**Directory**: `frontend/components/`

```
components/
├── Layout/
│   ├── Header.tsx
│   ├── Sidebar.tsx
│   └── Footer.tsx
├── VideoCards/
│   ├── VideoCard.tsx
│   ├── MetadataDisplay.tsx
│   └── EngagementMetrics.tsx
├── Chat/
│   ├── ChatInterface.tsx
│   ├── ChatMessage.tsx
│   ├── ChatInput.tsx
│   └── MessageStream.tsx
├── Forms/
│   └── VideoUploadForm.tsx
└── Common/
    ├── LoadingSpinner.tsx
    ├── ErrorBoundary.tsx
    └── Toast.tsx
```

#### 4.3 Pages
**Directory**: `frontend/app/`

```
app/
├── page.tsx (Home / Dashboard)
├── videos/
│   └── [collection_id]/
│       └── page.tsx (Video Comparison Page)
├── compare/
│   └── page.tsx (URL Input Page)
├── layout.tsx (Root Layout)
└── api/ (Optional: proxy endpoints)
```

#### 4.4 Home Page
**File**: `frontend/app/page.tsx`

Features:
- Welcome section
- Quick start guide
- Recent comparisons (localStorage)
- CTA to start new comparison

#### 4.5 URL Input Page
**File**: `frontend/app/compare/page.tsx`

Features:
- YouTube URL input
- Instagram Reel URL input
- Validation
- Submit button
- Progress indicator
- Error handling

#### 4.6 Comparison Page
**File**: `frontend/app/videos/[collection_id]/page.tsx`

Layout (side-by-side on desktop, stacked on mobile):

```
┌────────────────────────────────────────┐
│            Header + Navigation         │
├──────────────────┬──────────────────────┤
│   YouTube Video  │  Instagram Reel     │
│   Card           │  Card               │
│   - Metadata     │  - Metadata         │
│   - Engagement   │  - Engagement       │
├──────────────────┴──────────────────────┤
│          Chat Interface                │
│  ┌─────────────────────────────────┐  │
│  │ Message 1 [Video A - Chunk 3]   │  │
│  │ Message 2 [Video B - Chunk 5]   │  │
│  ├─────────────────────────────────┤  │
│  │ Type your question...           │  │
│  │ [Send]                          │  │
│  └─────────────────────────────────┘  │
└────────────────────────────────────────┘
```

**Components**:
- MetadataDisplay: views, likes, comments, creator, followers, duration
- EngagementMetrics: card showing engagement rates
- ChatInterface: conversation view with streaming
- LoadingState: during video processing

#### 4.7 Chat Interface Component
**File**: `frontend/components/Chat/ChatInterface.tsx`

Features:
- Message history
- Streaming responses (token-by-token)
- Citation highlighting
- Input field with auto-focus
- Send button + Enter key support
- Loading indicator during response
- Error message display

**Streaming Implementation**:
```typescript
async function* streamChat(question: string) {
  const response = await fetch('/api/chat', {
    method: 'POST',
    body: JSON.stringify({ question, collection_id })
  });
  
  const reader = response.body?.getReader();
  while (true) {
    const { done, value } = await reader!.read();
    if (done) break;
    yield new TextDecoder().decode(value);
  }
}
```

#### 4.8 State Management
**Directory**: `frontend/hooks/`

```typescript
// hooks/useChat.ts
export function useChat(collectionId: string) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  
  const sendMessage = async (question: string) => {
    // Stream response
  };
  
  return { messages, sendMessage, isLoading };
}

// hooks/useVideoProcessing.ts
export function useVideoProcessing() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  
  const processVideos = async (youtubeUrl: string, instagramUrl: string) => {
    // Process and return collection_id
  };
  
  return { processVideos, isProcessing, progress };
}
```

#### 4.9 Styling & Responsive Design
**Directory**: `frontend/styles/`

- Use Tailwind CSS
- Mobile-first approach
- Dark mode support (optional)
- Component-level CSS with Tailwind utilities

#### 4.10 Error Boundaries & Loading States
**Directory**: `frontend/components/Common/`

```typescript
// ErrorBoundary.tsx - Catch unexpected errors
// LoadingSpinner.tsx - Consistent loading indicator
// Toast.tsx - Error/success messages
```

---

## Phase 5: Integration & Testing (Week 4-5)

### Objectives
- End-to-end testing
- Performance optimization
- Error handling refinement

### Tasks

#### 5.1 Integration Testing
**Directory**: `backend/tests/`

```python
# tests/integration/test_end_to_end.py
- Test full flow: upload -> process -> chat -> response
- Use real sample videos
- Verify citation accuracy
- Check response quality
```

#### 5.2 Performance Testing
**Directory**: `backend/tests/`

```python
# tests/performance/test_latency.py
- Measure transcript extraction time
- Measure embedding time
- Measure retrieval latency
- Measure LLM inference time
- Measure E2E latency

Target Benchmarks:
- Transcript extraction: < 30s (YouTube), < 60s (Instagram)
- Metadata extraction: < 5s
- Embedding generation: < 2s per 1000 tokens
- Vector retrieval: < 100ms
- LLM inference: < 2s first token, then streaming
- E2E response: < 5-10s
```

#### 5.3 Frontend Testing
**Directory**: `frontend/__tests__/`

```typescript
// Component tests with React Testing Library
// Streaming implementation tests
// Form validation tests
// Error handling tests
```

#### 5.4 API Contract Testing
- Test all endpoints with various inputs
- Verify response schemas
- Test error scenarios
- Load testing with ab or k6

---

## Phase 6: Deployment & Containerization (Week 5-6)

### Objectives
- Docker setup
- Docker Compose orchestration
- Production-ready configuration

### Tasks

#### 6.1 Docker Setup
**Directory**: `docker/`

```dockerfile
# docker/Dockerfile.backend
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY backend/ .
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0"]

# docker/Dockerfile.frontend
FROM node:18-alpine
WORKDIR /app
COPY frontend/ .
RUN npm install && npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

#### 6.2 Docker Compose Orchestration
**File**: `docker-compose.yml`

```yaml
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
    volumes:
      - ollama_data:/root/.ollama

  backend:
    build:
      context: .
      dockerfile: docker/Dockerfile.backend
    ports:
      - "8000:8000"
    depends_on:
      - qdrant
      - ollama
    environment:
      - QDRANT_URL=http://qdrant:6333
      - OLLAMA_BASE_URL=http://ollama:11434
    volumes:
      - ./backend:/app

  frontend:
    build:
      context: .
      dockerfile: docker/Dockerfile.frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000/api

volumes:
  qdrant_data:
  ollama_data:
```

#### 6.3 Environment Configuration
**Files**:
- `.env.example` - Template
- `.env.local` - Local development
- `.env.production` - Production config

#### 6.4 Startup Script
**File**: `start.sh`

```bash
#!/bin/bash
# 1. Check dependencies
# 2. Pull required Docker images
# 3. Start docker-compose
# 4. Verify all services healthy
# 5. Provide setup instructions
```

---

## Phase 7: Documentation & Delivery (Week 6-8)

### Objectives
- Complete documentation
- Demo video
- Architecture explanation
- Scalability reasoning

### Tasks

#### 7.1 README
**File**: `README.md`

Sections:
1. Project overview
2. Quick start (local setup)
3. Docker setup
4. API documentation
5. Architecture diagram
6. Technology choices and rationale
7. Performance benchmarks
8. Known limitations
9. Future improvements
10. Contributing guidelines

#### 7.2 Architecture Documentation
**File**: `docs/ARCHITECTURE.md`

Sections:
1. System diagram (tools like Excalidraw)
2. Data flow (videos → transcripts → chunks → embeddings → vectors)
3. RAG pipeline explanation
4. Component responsibilities
5. Error handling strategy
6. Scalability considerations

#### 7.3 API Documentation
**File**: `docs/API.md`

- OpenAPI/Swagger spec (auto-generated from FastAPI)
- Request/response examples
- Error codes and messages
- Rate limiting (if applicable)

#### 7.4 Deployment Guide
**File**: `docs/DEPLOYMENT.md`

- Local setup
- Docker Compose
- Production considerations
- Monitoring and logging
- Troubleshooting

#### 7.5 Demo Video (Loom)
**Requirements**:
- 10-15 minute walkthrough
- Show:
  1. Home page
  2. URL input form
  3. Processing flow
  4. Comparison view
  5. Chat interface with streaming
  6. Citation examples
  7. Code walkthrough (backend structure)

#### 7.6 GitHub Repository Setup
**Structure**:
```
.gitignore (Python, Node, env files)
.github/
  └── workflows/
      ├── tests.yml (CI/CD)
      └── lint.yml
LICENSE
README.md
CONTRIBUTING.md
docs/
  ├── ARCHITECTURE.md
  ├── API.md
  ├── DEPLOYMENT.md
  └── SCALABILITY.md
```

#### 7.7 Scalability Analysis Document
**File**: `docs/SCALABILITY.md`

Discuss:
1. Current bottlenecks
2. Optimization strategies:
   - **Caching**: Redis for transcript/metadata cache
   - **Batch Processing**: Queue system (Celery) for embeddings
   - **Database Sharding**: Qdrant collections per creator
   - **CDN**: Cache video thumbnails
   - **Async/Concurrency**: Current implementation
3. Estimated throughput:
   - 1000 creators/day = ~42 per hour
   - With async processing: easily achievable
4. Cost optimization:
   - All free/open-source
   - Estimated cloud cost if needed
   - Self-hosting vs managed services

#### 7.8 Cost Analysis
**File**: `docs/COST_ANALYSIS.md`

**Current Stack (Self-Hosted)**:
- Server: $5-20/month (VPS)
- No API costs
- Storage: $0 (local)
- Bandwidth: Minimal

**Alternative (Managed)**:
- Qdrant Cloud: $25-200/month
- Hugging Face Inference: $0-100/month
- Ollama replacement (API): $0.01-0.10 per 1K tokens

---

## Development Timeline (Detailed)

### Week 1: Research & Prototyping
- **Mon-Tue**: Transcript extraction testing
- **Wed**: Metadata extraction testing
- **Thu**: Embedding testing
- **Fri**: RAG pipeline prototype

### Week 2: Core Services
- **Mon-Tue**: Project setup + Transcript/Metadata services
- **Wed**: Chunking + Embedding service
- **Thu**: Vector store + LLM service
- **Fri**: RAG chain service

### Week 3: Backend APIs + Frontend Setup
- **Mon-Tue**: FastAPI endpoints (process, chat)
- **Wed**: Frontend project + components setup
- **Thu**: Video card + metadata display
- **Fri**: Chat interface + streaming

### Week 4: Integration & Polish
- **Mon**: Frontend integration with backend
- **Tue**: Error handling refinement
- **Wed**: Performance optimization
- **Thu**: Testing (unit + integration)
- **Fri**: Bug fixes

### Week 5: Docker & Deployment
- **Mon-Tue**: Dockerfile + docker-compose
- **Wed**: Environment configuration
- **Thu**: Local deployment testing
- **Fri**: Deployment documentation

### Week 6: Final Documentation
- **Mon**: README + API docs
- **Tue**: Architecture documentation
- **Wed**: Scalability analysis
- **Thu**: GitHub setup + cleanup
- **Fri**: Demo video recording

---

## Success Metrics & Validation

### Phase 1 (Research)
✅ All core technologies proven to work locally
✅ Transcript extraction success rate > 95%
✅ Embedding generation < 2s per 1000 tokens
✅ RAG retrieval accuracy > 80%

### Phase 2-3 (Backend)
✅ All endpoints tested and documented
✅ Error handling covers >90% of failure cases
✅ E2E latency < 10s (from API call to response)
✅ Streaming works smoothly

### Phase 4 (Frontend)
✅ Responsive on mobile, tablet, desktop
✅ Streaming displays token-by-token
✅ Citations visible and accurate
✅ Forms validate inputs properly

### Phase 5 (Integration)
✅ Full end-to-end flow works
✅ No data loss
✅ Performance benchmarks met
✅ Error recovery graceful

### Phase 6-7 (Deployment & Docs)
✅ Docker setup reproducible
✅ Startup within 2 minutes
✅ All documentation clear and complete
✅ Demo video professional

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| Instagram transcript extraction fails | Medium | High | Implement yt-dlp + Whisper fallback early |
| Embedding quality poor | Low | Medium | Test multiple models in Phase 1 |
| Ollama unavailable | Low | High | Mock LLM for testing, fallback to API if needed |
| Vector retrieval too slow | Low | Medium | Test Qdrant performance, optimize queries |
| Streaming breaks on frontend | Medium | Medium | Test streaming early with SSE mock server |
| Chunking loses context | Medium | Medium | Implement overlap strategy, test quality |

---

## Parallel Work Streams

**Frontend & Backend can proceed in parallel** once Phase 2 is complete:
- Backend: Finish services + APIs
- Frontend: Build components + integrate

**Recommendations for parallel work**:
1. Use API mocks in frontend while backend development ongoing
2. Implement contract testing to catch integration issues early
3. Regular syncs on API schema changes
4. Use feature flags for incomplete features

---

## Key Decision Points

### ✅ Resolved Decisions
1. **Framework**: FastAPI + Next.js ✓
2. **Embeddings**: BAAI/bge-small ✓
3. **Vector DB**: Qdrant ✓
4. **LLM**: Ollama + Llama3 ✓
5. **Transcript**: youtube-transcript-api + yt-dlp + Whisper ✓

### ⏳ To Decide
1. Real-time progress updates (WebSocket vs polling)?
2. Session persistence (Redis vs in-memory)?
3. User authentication (MVP without it)?
4. Advanced RAG features (reranking, multi-hop)?

---

## Resource Requirements

### Hardware (Local Development)
- CPU: 4+ cores (for Ollama + embeddings)
- RAM: 16GB+ (Ollama model + vectors)
- Disk: 50GB+ (models + data)
- GPU: Optional (speeds up embeddings/LLM)

### Software
- Docker + Docker Compose
- Python 3.11+
- Node.js 18+
- Git

### Time Commitment
- **Solo Developer**: 8 weeks (part-time) or 4 weeks (full-time)
- **2 Developers**: 4-5 weeks (parallel frontend/backend)
- **3+ Developers**: 2-3 weeks (highly parallel)

---

## Appendix: File Structure Reference

```
project/
├── README.md
├── CONTRIBUTING.md
├── docker-compose.yml
├── .env.example
├── .gitignore
│
├── backend/
│   ├── pyproject.toml
│   ├── requirements.txt
│   ├── config.py
│   ├── api/
│   │   ├── main.py
│   │   ├── routes/
│   │   │   ├── videos.py
│   │   │   ├── chat.py
│   │   │   └── health.py
│   │   ├── schemas/
│   │   │   ├── video_request.py
│   │   │   ├── chat_request.py
│   │   │   └── responses.py
│   │   └── middleware/
│   │       └── error_handling.py
│   ├── services/
│   │   ├── transcript_service.py
│   │   └── metadata_service.py
│   ├── rag/
│   │   ├── chunker.py
│   │   └── rag_chain.py
│   ├── embeddings/
│   │   └── embedding_service.py
│   ├── vectorstore/
│   │   └── qdrant_service.py
│   ├── llm/
│   │   └── llm_service.py
│   ├── utils/
│   │   └── logger.py
│   └── tests/
│       ├── unit/
│       ├── integration/
│       └── performance/
│
├── frontend/
│   ├── package.json
│   ├── tsconfig.json
│   ├── tailwind.config.js
│   ├── next.config.js
│   ├── app/
│   │   ├── page.tsx
│   │   ├── layout.tsx
│   │   ├── compare/
│   │   │   └── page.tsx
│   │   └── videos/
│   │       └── [collection_id]/
│   │           └── page.tsx
│   ├── components/
│   │   ├── Layout/
│   │   ├── VideoCards/
│   │   ├── Chat/
│   │   ├── Forms/
│   │   └── Common/
│   ├── hooks/
│   │   ├── useChat.ts
│   │   └── useVideoProcessing.ts
│   ├── lib/
│   │   ├── api.ts
│   │   └── utils.ts
│   └── __tests__/
│
├── research/
│   ├── transcript_testing.ipynb
│   ├── metadata_testing.ipynb
│   ├── embedding_testing.ipynb
│   └── rag_testing.ipynb
│
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── docker-entrypoint.sh
│
└── docs/
    ├── ARCHITECTURE.md
    ├── API.md
    ├── DEPLOYMENT.md
    ├── SCALABILITY.md
    └── COST_ANALYSIS.md
```

---

## Next Steps After Approval

1. **Immediate**: Set up project structure + git repository
2. **Week 1**: Begin Phase 1 research in Jupyter notebooks
3. **Weekly**: Progress reviews and adjustments
4. **Feedback loops**: Test assumptions early and often

This plan is **flexible**: prioritize based on feedback, adjust timelines, and pivot if needed.

---

**Plan Created**: 2026-05-31  
**Status**: Ready for Review & Approval