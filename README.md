# CreatorLens — RAG Video Comparison Chatbot

CreatorLens is a full-stack RAG chatbot that lets you paste YouTube and Instagram Reel links into a chat prompt and get instant, data-driven comparisons. Engagement rates, hooks, content strategies, virality scores & all grounded in actual transcripts and metadata, with inline citations you can click to jump to the source.

I made this as a deep-dive learning project to understand the RAG pipeline end-to-end from transcript extraction all the way through chunking, embedding, vector search, and streaming LLM responses. Also prototyped everything in Jupyter notebooks first (`research/`), validated each piece worked independently, and then ported it into a clean production codebase.

---

## What I built

You type something like:

> Compare these two videos: https://www.youtube.com/watch?v=X5nMfrRrCgo & https://www.instagram.com/p/DY1ue54Ijwt/

And my system:
1. **Detects the URLs** in your prompt (regex, nothing fancy — I didn't want to over-engineer URL parsing)
2. **Extracts transcripts** — I try the YouTube captions API first because it's 10x faster. If that fails (disabled captions, age-restricted), I fall back to downloading audio + Whisper. Instagram always goes through Whisper since there's no caption API for Reels.
3. **Pulls metadata** — views, likes, comments, duration, hashtags, creator info, follower count. I use yt-dlp for both platforms because it handles everything with zero API keys.
4. **Calculates engagement metrics** — Calculate through given custom formula for engagement rate, like rate, comment rate, and a virality score (0-100). More on that formula later.
5. **Chunks the transcripts** — Sentence-boundary splitting, 256 words per chunk, 50-word overlap. Explained why below.
6. **Embeds them** — BGE-small-en-v1.5, runs locally on my CPU
7. **Indexes into Qdrant** — one collection per session, cosine similarity
8. **Caches everything in Redis** — metadata, transcripts, session state, chat history
9. **Streams a RAG-powered answer** via SSE — the LLM sees the retrieved chunks + metadata and answers with `[Video A - Chunk 3]` style citations

After that first prompt, you can ask follow-ups without pasting URLs again:
- *"Compare the hooks in the first 5 seconds"*
- *"What's the engagement rate of each?"*
- *"Suggest improvements for Video B based on what worked in A"*

I designed it to handle N videos, not just two. Paste 3 or 4 links and they get labeled Video A, B, C, D automatically.

---

## Tech Stack & Why I Chose Each Piece

| Layer | Choice | Why I picked it |
|---|---|---|
| **LLM** | Llama 3.1 8B via Groq | Groq's free tier gives ~300 tok/s. That's 10-50x faster than running Ollama locally on my CPU-only machine. 14,400 free requests/day is way more than we need currently for prototype. |
| **Embeddings** | BAAI/bge-small-en-v1.5 (local) | $0 cost, no API calls, no vendor lock-in. This model is #1 on MTEB for its size class (33M params), outputs 384-dim vectors, and runs on CPU in <300ms per batch. Good enough for conversational English transcripts. |
| **Vector DB** | Qdrant Cloud (free tier) | Explained in detail below. |
| **Cache** | Redis 7 (Docker) | We needed something that survives server restarts during development. `uvicorn --reload` kills in-memory state. Redis gives me sub-ms reads, TTL-based auto-expiration, and list operations for chat history. |
| **Transcripts** | youtube-transcript-api → yt-dlp + faster-whisper | I use the caption API first (fast, accurate, uses YouTube's own captions). Falls back to Whisper only when needed. I picked faster-whisper over openai-whisper because it's 4x faster on CPU & same model weights, CTranslate2 backend. |
| **Metadata** | yt-dlp | One tool for both YouTube and Instagram. No API keys. Thought about using the official YouTube Data API, but it has quota limits and doesn't work for Instagram. yt-dlp just works. |
| **Backend** | FastAPI + Uvicorn | Async-first, which matters because transcript extraction and embedding are I/O-bound. SSE streaming via `StreamingResponse`. Auto-generated API docs at `/docs`. |
| **Frontend** | Vite + React + TypeScript | Mostly vibecoded. |
| **Orchestration** | LangChain (minimal) | I'm only using it for the Groq LLM wrapper (`ChatGroq`) and its `.astream()` method. I intentionally didn't use LangChain's retrieval chains or agents — I built my own retrieval pipeline because I wanted to understand every step. |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     React Frontend (Vite)                   │
│  ┌──────────────────┐  ┌──────────────────────────────────┐ │
│  │  Video Cards     │  │  Streaming Chat Panel (SSE)      │ │
│  │  - Metadata      │  │  - Markdown rendering            │ │
│  │  - Engagement    │  │  - Citation badges               │ │
│  │  - Embedded video│  │  - Suggestion chips              │ │
│  └──────────────────┘  └──────────────────────────────────┘ │
└─────────────────────┬───────────────────────────────────────┘
                      │ /api/* (Vite proxy)
┌─────────────────────▼───────────────────────────────────────-┐
│                    FastAPI Backend                           │
│                                                              │
│  POST /api/chat (SSE)                                        │
│    │                                                         │
│    ├─ 1. Extract URLs from prompt (regex)                    │
│    ├─ 2. Generate deterministic session_id (MD5 of URLs)     │
│    ├─ 3. If new session → trigger processing pipeline:       │
│    │     ├─ Transcript extraction (concurrent per video)     │
│    │     ├─ Metadata extraction (concurrent per video)       │
│    │     ├─ Chunking (sentence-based, 256 words, 50 overlap) │
│    │     ├─ Embedding (BGE-small, batch)                     │
│    │     ├─ Qdrant upsert (per-session collection)           │
│    │     └─ Redis cache (session metadata + transcripts)     │
│    │                                                         │
│    ├─ 4. Retrieve relevant chunks (embed query → Qdrant)     │
│    ├─ 5. Build prompt (system + metadata + chunks + history) │
│    └─ 6. Stream LLM response (Groq, token-by-token SSE)      │
│                                                              │
│  GET  /api/sessions/{id}     — Load session + chat history   │
│  DELETE /api/sessions/{id}   — Clean up session data         │
│  GET  /api/health            — Service health check          │
└──────────┬──────────────┬────────────────────────────────────┘
           │              │
    ┌──────▼──────┐ ┌─────▼──────┐
    │  Qdrant     │ │  Redis     │
    │  (vectors)  │ │  (cache)   │
    └─────────────┘ └────────────┘
```

---

## Trade-offs I Thought About

### Why I went with Qdrant over Chroma/Pinecone/pgvector

Considered Chroma (no free hosted tier), Pinecone (clunky metadata filtering, need to define indexes upfront), and pgvector (didn't want Postgres just for vectors when Redis handles everything else). Qdrant Cloud's free tier (1GB) solved hosting, and its native metadata filtering lets me filter by `video_id` or `source` on any field without schema changes.

The killer feature: **per-session collection isolation**. Each comparison session gets its own collection, zero cross-session leakage, and cleanup is a single `delete_collection()` call instead of batch-deleting by metadata.

### Why I chose 256-word chunks with 50-word overlap

Tested in `research/3chunking_embedding.ipynb`. BGE-small has a 512-token max input, 256 words ≈ 340 tokens, safe margin. 512 words caused truncation; 128 words fragmented ideas across chunks. 50-word overlap (~20%) prevents boundary information loss, tested 25 (too little) and 100 (diminishing returns). Sentence-based splitting over fixed-size keeps complete thoughts together, so citations actually make sense.

### Why I picked BGE-small over OpenAI embeddings

Honestly, the main reason is cost. OpenAI charges $0.02 per million tokens. For a project where I was re-embedding constantly during development (no persistent embedding cache yet), that would've added up. BGE-small is $0, runs locally, and I don't need to worry about network latency or vendor deprecation.

The trade-off I'm knowingly accepting: BGE-small (384-dim) is less accurate than `text-embedding-3-large` (3072-dim) on nuanced semantic queries. But for transcript comparison & conversational English, not legal documents, it's been good enough in my testing. If I ever need better retrieval quality, I can swap models by changing one environment variable.

### Why I'm using Groq instead of running Ollama locally

Speed. Groq pushes ~300 tokens/second on their free tier. Running Llama 3.1 8B locally on my M4 MacBook via Ollama gives me ~20-30 tok/s.

The trade-off: I'm dependent on Groq's free tier (30 req/min, 14,400 req/day). For a demo project that's fine. If I ever need to scale this, I'd switch to a paid Groq plan or self-host behind vLLM.

### Why I put chat memory in Redis instead of keeping it in-memory

Three things led to this:

1. **`uvicorn --reload` kept killing my state.** During development, every code change restarts the server. In-memory dicts just evaporate. Redis persists via `appendonly yes`.
2. **I wanted bounded history.** I keep the last 20 messages per session with `LPUSH` + `LTRIM`. The LLM only sees the last 6 turns (truncated to 500 chars each). Older messages become irrelevant for video comparison anyway.
3. **TTL auto-cleanup.** Sessions expire after 24 hours. I didn't want to write cron jobs or manual garbage collection logic.

### Why faster-whisper over openai-whisper

Same model weights, 4x faster on CPU. It uses a CTranslate2 backend with int8 quantization, so the memory footprint is lower too. For Instagram Reels (typically 15-90 seconds of audio), transcription takes 3-8 seconds instead of 15-30. That difference matters when users are staring at a loading spinner.

---

## What Breaks at Scale

I built this for demo. Here's what I know would break if real traffic hit it:

### ~100 concurrent users

**Whisper becomes the bottleneck.** Each Instagram transcription blocks a thread for 3-8 seconds. I'm using `asyncio.to_thread()` to offload to the thread pool, but Python's default pool is 40 threads. 100 concurrent Instagram transcriptions would queue up badly.

**What I'd do:** Move transcription to a background job queue. I already have `arq` in my `requirements.txt`. I just haven't wired it up because it wasn't needed for a single-user demo.

### ~1,000 concurrent users

**Qdrant collections explode.** I create one collection per session. 1,000 sessions = 1,000 collections. Math-wise it fits (~75KB per session × 1,000 = 75MB, well under the 1GB free tier), but collection management overhead becomes noticeable.

**Redis memory pressure starts.** Each session caches ~50KB of metadata, transcripts, and chat history. 1,000 sessions = 50MB. My Docker config caps Redis at 256MB with LRU eviction. Active sessions would start getting evicted.

**What I'd do:** Switch to a shared Qdrant collection with `session_id` as a metadata filter. Loses the clean per-session isolation but scales way better. Compress cached data in Redis. Reduce TTLs.

### ~10,000 concurrent users

**Everything single-instance breaks.** My single Uvicorn process can't hold 10K concurrent SSE connections. Redis's 20-connection pool becomes a bottleneck. Groq's rate limits get hit instantly. The embedding model (130MB, single instance) can't keep up.

**What I'd need:** A complete re-architecture — Kubernetes with HPA, managed Redis, paid Qdrant cluster with sharding, self-hosted LLM behind a load balancer, and a proper distributed job queue.

I'm not there and I know it.

---

## Things That Are Genuinely Fragile (Analyzed by AI)

I want to be upfront about what's held together with duct tape:

- **Instagram metadata extraction** depends on yt-dlp's ability to scrape Instagram, which breaks every few weeks when Meta changes their frontend. I added a GraphQL fallback for view counts, but that uses an undocumented endpoint (`doc_id=8845758582119845`) — it could stop working any day.
- **YouTube transcript API** fails for age-restricted videos and videos with disabled captions. My Whisper fallback handles this, but adds 10-15 seconds of latency.
- **No authentication.** Anyone with the URL can use it. For multi-user, I'd need auth + per-user session isolation.
- **No rate limiting** on the API. A single user could spam `/api/chat` and eat my entire Groq quota in minutes.
- **Embedding model loads on every server restart.** Takes 2-5 seconds and downloads from HuggingFace Hub the first time. In production, I'd bake the model into the Docker image.
- **The virality score is a formula I made up.** `(normalized_engagement × 0.6 + normalized_likes × 0.4) × 100`. It's useful for relative comparison between two videos, but it's not a scientifically validated metric. Don't cite it in a paper.

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- Docker (for Redis)
- ffmpeg (for audio extraction)

### 1. Clone and configure

```bash
git clone https://github.com/gururaj1512/Video-Comparison-Chatbot.git
cd Video-Comparison-Chatbot
cp .env.example .env
```

Edit `.env` with your API keys:
- **`GROQ_API_KEY`** — Get a free key at [console.groq.com/keys](https://console.groq.com/keys)
- **`QDRANT_URL`** + **`QDRANT_API_KEY`** — Get a free cluster at [cloud.qdrant.io](https://cloud.qdrant.io) (or leave blank for in-memory mode)

### 2. Start Redis

```bash
docker compose up redis -d
```

### 3. Install and run backend

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

First startup downloads the BGE embedding model (~130MB) and the Whisper model (~140MB). Takes about 30-60 seconds — I know, I wish it was faster.

### 4. Install and run frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173). Vite proxies `/api/*` to the FastAPI backend automatically.

### Or use the start script I wrote

```bash
chmod +x start.sh
./start.sh
```

It checks for `.env`, starts Redis via Docker, creates a venv, installs deps, and launches the server.

---

## Environment Variables

| Variable | Default | What I use it for |
|---|---|---|
| `GROQ_API_KEY` | — | **Required.** My Groq API key for LLM inference |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL (I use Qdrant Cloud) |
| `QDRANT_API_KEY` | — | Qdrant API key (needed for cloud) |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | HuggingFace model for embeddings |
| `CHUNK_SIZE` | `256` | Words per transcript chunk |
| `CHUNK_OVERLAP` | `50` | Word overlap between chunks |
| `TOP_K` | `5` | How many chunks I retrieve per query |
| `SIMILARITY_THRESHOLD` | `0.3` | Minimum cosine similarity to include a chunk |
| `WHISPER_MODEL_SIZE` | `base` | Whisper model size (tiny/base/small/medium/large) |
| `LLM_MODEL` | `llama-3.1-8b-instant` | Which Groq model to use |
| `LLM_TEMPERATURE` | `0.3` | Low for factual answers, not so low it sounds robotic |
| `SESSION_TTL` | `86400` | How long sessions live (24h) |

---

## API Endpoints

| Method | Endpoint | What it does |
|---|---|---|
| `POST` | `/api/chat` | SSE streaming chat — auto-detects URLs, processes videos on-the-fly, streams the RAG response |
| `POST` | `/api/chat/sync` | Same thing but non-streaming (I use this for debugging) |
| `POST` | `/api/videos/process` | Manually trigger video processing without a chat question |
| `GET` | `/api/sessions/{id}` | Get session metadata + chat history |
| `DELETE` | `/api/sessions/{id}` | Delete session data (Qdrant collection + Redis keys) |
| `GET` | `/api/health` | Health check for all services |

Full interactive docs at `http://localhost:8000/docs` when running.

---

## Honest Limitations

- **English only.** I pinned Whisper to `language="en"` and the LLM system prompt is English-only. Supporting other languages would need a multilingual embedding model too.
- **No auth.** This is a single-user demo. I didn't build login because it wasn't the point of this project.
- **Instagram scraping is flaky.** It breaks every few weeks. I've built fallbacks, but I can't guarantee they'll work tomorrow.
- **First request is slow.** Processing two videos from scratch takes 10-30 seconds (download + transcribe + embed + index). After that, follow-up questions are fast (<2 seconds).
- **The virality score is my invention.** It's useful for comparing two videos against each other, but it's not a published metric. I made up the formula based on what seemed reasonable.

---
