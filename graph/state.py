"""
Shared LangGraph workflow state.

A single ``PlannerState`` TypedDict flows through every node. Each agent reads
the request (plus upstream outputs) and writes its own structured slice. Using
``total=False`` keeps nodes free to return partial updates that LangGraph merges.
"""

from __future__ import annotations

from typing import List, TypedDict

from schemas.models import (
    ActivityOutput,
    BudgetOutput,
    HotelOutput,
    ItineraryOutput,
    ResearchOutput,
    SafetyOutput,
    TravelPlan,
    TripRequest,
)


class PlannerState(TypedDict, total=False):
    # ---- Inputs / context ----
    request: TripRequest
    trip_id: str
    memory_context: str

    # ---- Per-agent outputs ----
    research: ResearchOutput
    budget: BudgetOutput
    hotels: HotelOutput
    activities: ActivityOutput
    itinerary: ItineraryOutput
    safety: SafetyOutput

    # ---- Aggregation ----
    final_plan: TravelPlan
    warnings: List[str]
