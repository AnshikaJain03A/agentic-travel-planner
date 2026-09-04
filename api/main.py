"""
FastAPI application exposing the travel planner.

Endpoints:
    POST /plan-trip          - run the multi-agent workflow and return a plan
    GET  /trip-history       - list previously planned trips (optionally by user)
    GET  /user-preferences   - aggregated memory/preferences for a user
    GET  /trip/{trip_id}     - fetch a single saved plan
    GET  /health             - liveness + LLM/RAG status

Run with::

    uvicorn api.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from core.logging_config import configure_logging, get_logger
from database.db import init_db
from graph.workflow import get_planner
from memory.memory_manager import memory_manager
from rag.knowledge_base import get_knowledge_base
from schemas.models import (
    TravelPlan,
    TripHistoryItem,
    TripRequest,
    UserPreferences,
)
from services.llm import get_llm

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up persistence and the knowledge base at boot."""
    init_db()
    try:
        kb = get_knowledge_base()
        logger.info("Knowledge base ready with %d chunks", kb.count())
    except Exception:  # noqa: BLE001
        logger.exception("Knowledge base failed to initialise (continuing)")
    logger.info("API started. LLM available: %s", get_llm().is_available())
    yield


app = FastAPI(
    title="Agentic AI Travel Planner",
    description="Multi-agent travel planning powered by LangGraph, LangChain and RAG.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", tags=["system"])
def health() -> dict:
    """Liveness probe + backend capability summary."""
    llm = get_llm()
    try:
        kb_count = get_knowledge_base().count()
    except Exception:  # noqa: BLE001
        kb_count = 0
    return {
        "status": "ok",
        "llm_provider": llm.provider,
        "llm_model": llm.model_name,
        "llm_available": llm.is_available(),
        "rag_chunks": kb_count,
        "mock_fallback_enabled": settings.allow_mock_fallback,
    }


@app.post("/plan-trip", response_model=TravelPlan, tags=["planning"])
def plan_trip(request: TripRequest) -> TravelPlan:
    """Run the multi-agent workflow and return the aggregated travel plan."""
    logger.info("POST /plan-trip destination=%s user=%s", request.destination, request.user_id)
    try:
        return get_planner().plan(request)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Planning failed")
        raise HTTPException(status_code=500, detail=f"Planning failed: {exc}") from exc


@app.get("/trip-history", response_model=List[TripHistoryItem], tags=["planning"])
def trip_history(
    user_id: Optional[str] = Query(default=None, description="Filter by user id"),
    limit: int = Query(default=50, ge=1, le=200),
) -> List[TripHistoryItem]:
    """Return recent trip history, optionally filtered by user."""
    return memory_manager.get_history(user_id=user_id, limit=limit)


@app.get("/user-preferences", response_model=UserPreferences, tags=["planning"])
def user_preferences(
    user_id: str = Query(..., description="The user id to look up"),
) -> UserPreferences:
    """Return aggregated, learned preferences for a user."""
    return memory_manager.get_preferences(user_id)


@app.get("/trip/{trip_id}", response_model=TravelPlan, tags=["planning"])
def get_trip(trip_id: str) -> TravelPlan:
    """Fetch a single previously generated plan by id."""
    plan = memory_manager.get_trip(trip_id)
    if plan is None:
        raise HTTPException(status_code=404, detail="Trip not found")
    return plan
