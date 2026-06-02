"""
Transcript chunking service.

Ported from research/3chunking_embedding.ipynb with enhancements:
- Sentence-based chunking preserves semantic boundaries
- Timestamp mapping links chunks back to video segments
- Configurable chunk_size and overlap via settings

Why sentence-based chunking over fixed-size:
- Fixed-size splits mid-sentence → broken citations and lost meaning
- Sentence-based keeps complete thoughts together
- Overlap prevents information loss at boundaries

Why 256 words per chunk:
- BGE-small has a 512-token max input — 256 words ≈ 340 tokens (safe margin)
- Large enough to capture a coherent idea (15-30 seconds of speech)
- Small enough for precise retrieval (avoids diluting relevance)

Why 50-word overlap (~20%):
- Standard in RAG literature for preventing boundary information loss
- Minimal storage overhead (only ~20% redundancy)
- Ensures key phrases spanning chunk boundaries appear in at least one chunk
"""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from backend.config import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TextChunk:
    """A chunk of transcript with metadata for vector storage."""

    chunk_id: int
    video_id: str
    source: str                     # 'youtube' or 'instagram'
    text: str
    video_label: str = ""           # 'Video A' or 'Video B'
    timestamp_start: float = 0.0    # seconds
    timestamp_end: float = 0.0
    chunk_size: int = 0             # word count

    def __post_init__(self):
        self.chunk_size = len(self.text.split())

    def to_payload(self) -> Dict:
        """Convert to dict for Qdrant payload (stored alongside the vector)."""
        return {
            "chunk_id": self.chunk_id,
            "video_id": self.video_id,
            "source": self.source,
            "video_label": self.video_label,
            "text": self.text,
            "timestamp_start": self.timestamp_start,
            "timestamp_end": self.timestamp_end,
            "chunk_size": self.chunk_size,
        }


class TranscriptChunker:
    """
    Chunks transcripts into overlapping segments for embedding.

    Strategies:
    - chunk_by_sentences: Splits by sentence boundaries (default, recommended)
    - chunk_with_timestamps: Preserves timestamp info from transcript entries
    """

    def __init__(
        self,
        chunk_size: int = None,
        overlap: int = None,
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.overlap = overlap or settings.chunk_overlap

    def chunk_by_sentences(
        self,
        text: str,
        video_id: str,
        source: str,
        video_label: str = "",
    ) -> List[TextChunk]:
        """
        Chunk text by sentence boundaries with overlap.
        Primary chunking strategy — preserves semantic coherence.
        """
        if not text or not text.strip():
            return []

        # Split by sentence-ending punctuation
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks: List[TextChunk] = []
        chunk_id = 0
        current_sentences: List[str] = []
        current_word_count = 0

        for sentence in sentences:
            words_in_sentence = len(sentence.split())

            if (
                current_word_count + words_in_sentence > self.chunk_size
                and current_sentences
            ):
                # Save current chunk
                chunk_text = " ".join(current_sentences)
                chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        video_id=video_id,
                        source=source,
                        video_label=video_label,
                        text=chunk_text,
                    )
                )
                chunk_id += 1

                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(
                    current_sentences, self.overlap
                )
                current_sentences = overlap_sentences + [sentence]
                current_word_count = sum(
                    len(s.split()) for s in current_sentences
                )
            else:
                current_sentences.append(sentence)
                current_word_count += words_in_sentence

        # Final chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    video_id=video_id,
                    source=source,
                    video_label=video_label,
                    text=chunk_text,
                )
            )

        logger.info(
            "Chunked transcript for %s into %d chunks (avg %d words)",
            video_id,
            len(chunks),
            sum(c.chunk_size for c in chunks) // max(len(chunks), 1),
        )
        return chunks

    def chunk_with_timestamps(
        self,
        text: str,
        video_id: str,
        source: str,
        transcript_entries: List[Dict],
        video_label: str = "",
    ) -> List[TextChunk]:
        """
        Chunk by word count while preserving timestamp information.
        Used when transcript entries have start/duration fields.
        """
        if not text or not text.strip():
            return []

        # Build word → timestamp mapping
        word_timestamps = self._build_word_timestamp_map(transcript_entries)
        words = text.split()

        chunks: List[TextChunk] = []
        chunk_id = 0

        step = max(self.chunk_size - self.overlap, 1)
        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.chunk_size]
            if not chunk_words:
                continue

            chunk_text = " ".join(chunk_words)
            start_idx = i
            end_idx = min(i + self.chunk_size, len(words)) - 1

            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    video_id=video_id,
                    source=source,
                    video_label=video_label,
                    text=chunk_text,
                    timestamp_start=word_timestamps.get(start_idx, 0.0),
                    timestamp_end=word_timestamps.get(end_idx, 0.0),
                )
            )
            chunk_id += 1

        logger.info(
            "Chunked transcript for %s into %d chunks with timestamps",
            video_id,
            len(chunks),
        )
        return chunks

    # Helpers
    def _get_overlap_sentences(
        self, sentences: List[str], overlap_words: int
    ) -> List[str]:
        """Get trailing sentences totaling ~overlap_words."""
        total = 0
        overlap: List[str] = []
        for sentence in reversed(sentences):
            wc = len(sentence.split())
            if total + wc <= overlap_words:
                overlap.insert(0, sentence)
                total += wc
            else:
                break
        return overlap

    @staticmethod
    def _build_word_timestamp_map(entries: List[Dict]) -> Dict[int, float]:
        """Create word-index → timestamp mapping from transcript entries."""
        mapping: Dict[int, float] = {}
        word_count = 0
        for entry in entries:
            text = entry.get("text", "")
            timestamp = entry.get("start", 0.0)
            for _ in text.split():
                mapping[word_count] = timestamp
                word_count += 1
        return mapping


# Singleton (stateless, so sharing is safe)
chunker = TranscriptChunker()
