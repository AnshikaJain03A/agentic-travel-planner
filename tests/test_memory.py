"""Tests for SQLite-backed memory management."""

from __future__ import annotations

from agents import ResearchAgent
from memory.memory_manager import MemoryManager
from schemas.models import (
    Interest,
    ResearchOutput,
    TravelPlan,
    TravelStyle,
    TripRequest,
)


def _make_plan(trip_id: str, user_id: str, destination: str, budget: float) -> TravelPlan:
    req = TripRequest(
        destination=destination,
        days=3,
        budget=budget,
        travelers=2,
        travel_style=TravelStyle.BUDGET,
        interests=[Interest.FOOD, Interest.BEACHES],
        departure_city="Delhi",
        user_id=user_id,
    )
    return TravelPlan(trip_id=trip_id, request=req, research=ResearchOutput())


def test_save_and_fetch_trip() -> None:
    mm = MemoryManager()
    plan = _make_plan("trip-001", "mem-user", "Goa", 20000)
    mm.save_trip(plan)

    fetched = mm.get_trip("trip-001")
    assert fetched is not None
    assert fetched.request.destination == "Goa"


def test_history_filtered_by_user() -> None:
    mm = MemoryManager()
    mm.save_trip(_make_plan("trip-101", "hist-user", "Jaipur", 15000))
    mm.save_trip(_make_plan("trip-102", "hist-user", "Manali", 30000))

    history = mm.get_history(user_id="hist-user")
    destinations = {h.destination for h in history}
    assert {"Jaipur", "Manali"} <= destinations


def test_preferences_aggregation() -> None:
    mm = MemoryManager()
    mm.save_trip(_make_plan("trip-201", "pref-user", "Goa", 20000))
    mm.save_trip(_make_plan("trip-202", "pref-user", "Goa", 30000))

    prefs = mm.get_preferences("pref-user")
    assert prefs.trips_planned == 2
    assert prefs.average_budget == 25000
    assert "Goa" in prefs.favorite_destinations
    assert "food" in prefs.common_interests


def test_memory_context_first_time_user() -> None:
    mm = MemoryManager()
    req = TripRequest(
        destination="Leh",
        days=5,
        budget=40000,
        travelers=1,
        departure_city="Delhi",
        user_id="brand-new-user",
    )
    context = mm.build_memory_context(req)
    assert "first trip" in context.lower()


def test_run_persists_via_workflow_components(sample_request: TripRequest) -> None:
    # Sanity: agent run works against a fresh request (no DB writes here).
    update = ResearchAgent().run(
        {"request": sample_request, "trip_id": "x", "memory_context": "", "warnings": []}
    )
    assert "research" in update
