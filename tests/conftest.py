"""
Pytest configuration and shared fixtures.

Crucially, this module sets environment variables BEFORE any application module
(and therefore the cached ``Settings`` singleton) is imported, forcing the test
suite into deterministic offline "mock" mode with isolated, temporary storage.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# ---- Force offline / isolated configuration (must precede project imports) ----
_TMP = Path(tempfile.mkdtemp(prefix="travel_planner_tests_"))
os.environ["OPENAI_API_KEY"] = ""
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["LLM_PROVIDER"] = "openai"
os.environ["ALLOW_MOCK_FALLBACK"] = "true"
os.environ["SQLITE_DB_PATH"] = str(_TMP / "test.db")
os.environ["CHROMA_PERSIST_DIR"] = str(_TMP / "chroma")
# Point RAG at the real bundled documents directory.
os.environ["RAG_DOCS_DIR"] = str(Path(__file__).resolve().parents[1] / "rag" / "documents")
os.environ["LOG_LEVEL"] = "WARNING"

import pytest  # noqa: E402

from schemas.models import Interest, TravelStyle, TripRequest  # noqa: E402


@pytest.fixture
def sample_request() -> TripRequest:
    return TripRequest(
        destination="Goa",
        days=4,
        budget=25000,
        travelers=2,
        travel_style=TravelStyle.BUDGET,
        interests=[Interest.BEACHES, Interest.FOOD, Interest.NIGHTLIFE],
        departure_city="Delhi",
        user_id="test-user",
    )
