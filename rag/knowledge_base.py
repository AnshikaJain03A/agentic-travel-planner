"""
RAG knowledge base built on ChromaDB.

Pipeline:
    documents (.md/.txt)  ->  chunking  ->  embeddings  ->  Chroma collection
    query                 ->  embeddings ->  similarity search -> context

The :class:`TravelKnowledgeBase` is the single entry point used by agents to
retrieve grounding context for a destination.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from core.config import settings
from core.logging_config import get_logger
from rag.embeddings import TravelEmbeddingFunction

logger = get_logger(__name__)

_COLLECTION_NAME = "travel_knowledge"


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    """Split ``text`` into overlapping, sentence-aware chunks.

    We first normalise whitespace, then greedily pack sentences into chunks of
    roughly ``chunk_size`` characters with ``overlap`` characters carried over
    to preserve context across boundaries.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current)
            # Start a new chunk, carrying over an overlap tail for continuity.
            tail = current[-overlap:] if overlap and current else ""
            current = f"{tail} {sentence}".strip()

    if current:
        chunks.append(current)
    return chunks


class TravelKnowledgeBase:
    """A persistent Chroma-backed vector store of travel knowledge."""

    def __init__(self) -> None:
        import chromadb

        self._embedding_fn = TravelEmbeddingFunction()
        self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Knowledge base ready (%d chunks) at %s",
            self.count(),
            settings.chroma_persist_dir,
        )

    # ------------------------------------------------------------------
    def count(self) -> int:
        try:
            return self._collection.count()
        except Exception:  # noqa: BLE001
            return 0

    # ------------------------------------------------------------------
    def add_document(self, doc_id: str, text: str, source: str = "") -> int:
        """Chunk a document and upsert its chunks. Returns number of chunks."""
        chunks = chunk_text(text)
        if not chunks:
            return 0

        ids = [f"{doc_id}::chunk-{i}" for i in range(len(chunks))]
        metadatas = [{"source": source or doc_id, "chunk": i} for i in range(len(chunks))]
        # Embed explicitly so behaviour is identical across ChromaDB versions
        # (different releases call the embedding function differently).
        embeddings = self._embedding_fn(chunks)
        self._collection.upsert(
            ids=ids, documents=chunks, metadatas=metadatas, embeddings=embeddings
        )
        logger.info("Ingested %d chunks from %s", len(chunks), source or doc_id)
        return len(chunks)

    # ------------------------------------------------------------------
    def ingest_directory(self, directory: str | None = None) -> int:
        """Ingest every ``.md`` / ``.txt`` file in ``directory``.

        Returns the total number of chunks ingested.
        """
        docs_dir = Path(directory or settings.rag_docs_dir)
        if not docs_dir.exists():
            logger.warning("RAG docs directory %s does not exist", docs_dir)
            return 0

        total = 0
        for path in sorted(docs_dir.glob("**/*")):
            if path.suffix.lower() not in {".md", ".txt"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            total += self.add_document(doc_id=path.stem, text=text, source=path.name)
        logger.info("Directory ingest complete: %d total chunks", total)
        return total

    # ------------------------------------------------------------------
    def retrieve(self, query: str, k: int = 4) -> List[str]:
        """Return up to ``k`` relevant chunks for ``query``."""
        if self.count() == 0:
            return []
        try:
            query_embedding = self._embedding_fn([query])
            result = self._collection.query(query_embeddings=query_embedding, n_results=k)
            documents = result.get("documents") or [[]]
            return documents[0]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Retrieval failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    def retrieve_context(self, query: str, k: int = 4) -> str:
        """Return retrieved chunks joined into a single prompt-ready string."""
        chunks = self.retrieve(query, k=k)
        if not chunks:
            return ""
        return "\n\n".join(f"- {chunk}" for chunk in chunks)


_kb_singleton: TravelKnowledgeBase | None = None


def get_knowledge_base() -> TravelKnowledgeBase:
    """Lazily construct and return the shared knowledge base singleton.

    On first construction, auto-ingests the bundled documents if the
    collection is empty so the system is useful out of the box.
    """
    global _kb_singleton
    if _kb_singleton is None:
        _kb_singleton = TravelKnowledgeBase()
        if _kb_singleton.count() == 0:
            _kb_singleton.ingest_directory()
    return _kb_singleton
