"""
Embedding functions for the RAG knowledge base.

We expose a single :class:`TravelEmbeddingFunction` that conforms to ChromaDB's
``EmbeddingFunction`` protocol. When a valid OpenAI key is configured it uses
``text-embedding-3-small``; otherwise it transparently falls back to a
deterministic, dependency-free hashing embedding so the application (and its
tests) run fully offline.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import List

from core.config import settings
from core.logging_config import get_logger

logger = get_logger(__name__)

# Dimensionality of the local fallback embedding. Independent of OpenAI's.
_LOCAL_DIM = 256
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _hash_embedding(text: str, dim: int = _LOCAL_DIM) -> List[float]:
    """Deterministic bag-of-words hashing embedding, L2-normalised.

    Not semantically powerful, but stable and free - perfect for local dev,
    CI and demos without API credentials.
    """
    vector = [0.0] * dim
    tokens = _TOKEN_RE.findall(text.lower())
    for token in tokens:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


class TravelEmbeddingFunction:
    """ChromaDB-compatible embedding function.

    Backend priority:
      1. Google embeddings - when the active provider is Google/Gemini and a key is set.
      2. OpenAI embeddings - when an OpenAI key is available.
      3. Deterministic local hashing fallback - fully offline.
    """

    def __init__(self) -> None:
        self._backend = "local"
        self._client = None
        provider = settings.llm_provider.lower()

        # 1. Prefer Google embeddings when running on Gemini.
        if provider in ("google", "gemini") and settings.has_google_key:
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings

                self._client = GoogleGenerativeAIEmbeddings(
                    model=settings.google_embedding_model,
                    google_api_key=settings.google_api_key,
                )
                self._backend = "google"
                logger.info("RAG embeddings: using Google %s", settings.google_embedding_model)
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("Google embeddings unavailable (%s); trying next backend", exc)

        # 2. OpenAI embeddings.
        if settings.has_openai_key:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=settings.openai_api_key)
                self._backend = "openai"
                logger.info("RAG embeddings: using OpenAI %s", settings.openai_embedding_model)
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("OpenAI embeddings unavailable (%s); using local fallback", exc)

        # 3. Local fallback.
        logger.info("RAG embeddings: using deterministic local fallback")

    # ChromaDB calls the function with a list of documents.
    def __call__(self, input: List[str]) -> List[List[float]]:  # noqa: A002 - chroma API name
        try:
            if self._backend == "google" and self._client is not None:
                return self._client.embed_documents(list(input))
            if self._backend == "openai" and self._client is not None:
                response = self._client.embeddings.create(
                    model=settings.openai_embedding_model,
                    input=input,
                )
                return [item.embedding for item in response.data]
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s embedding call failed (%s); falling back", self._backend, exc)
        return [_hash_embedding(text) for text in input]

    # Chroma >=0.4.16 requires a stable name for cached collections.
    @staticmethod
    def name() -> str:
        return "travel-embedding-fn"
