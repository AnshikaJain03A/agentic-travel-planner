"""
CLI entry point for (re)ingesting the travel knowledge base.

Usage::

    python -m rag.ingest                 # ingest the bundled documents dir
    python -m rag.ingest path/to/docs    # ingest a custom directory
"""

from __future__ import annotations

import sys

from core.logging_config import configure_logging, get_logger
from rag.knowledge_base import TravelKnowledgeBase

logger = get_logger(__name__)


def main() -> None:
    configure_logging()
    directory = sys.argv[1] if len(sys.argv) > 1 else None
    kb = TravelKnowledgeBase()
    total = kb.ingest_directory(directory)
    logger.info("Ingestion finished: %d chunks now in the knowledge base", kb.count())
    print(f"Ingested {total} chunks. Collection size: {kb.count()}")


if __name__ == "__main__":
    main()
