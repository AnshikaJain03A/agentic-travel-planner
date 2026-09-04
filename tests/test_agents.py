"""Tests that every agent's mock output conforms to its schema."""

from __future__ import annotations

from agents import (
    ActivityAgent,
    BudgetAgent,
    HotelAgent,
    ItineraryAgent,
    ResearchAgent,
    SafetyAgent,
)
from graph.state import PlannerState
from schemas.models import (
    ActivityOutput,
    BudgetOutput,
    HotelOutput,
    ItineraryOutput,
    ResearchOutput,
    SafetyOutput,
    TripRequest,
)


def _state(req: TripRequest) -> PlannerState:
    return {"request": req, "trip_id": "t1", "memory_context": "", "warnings": []}


def test_research_agent_mock(sample_request: TripRequest) -> None:
    out = ResearchAgent().mock_output(_state(sample_request))
    assert isinstance(out, ResearchOutput)
    assert out.top_attractions and out.hidden_gems


def test_budget_agent_mock(sample_request: TripRequest) -> None:
    out = BudgetAgent().mock_output(_state(sample_request))
    assert isinstance(out, BudgetOutput)
    assert out.total_estimated_cost.startswith("INR")


def test_hotel_agent_mock(sample_request: TripRequest) -> None:
    out = HotelAgent().mock_output(_state(sample_request))
    assert isinstance(out, HotelOutput)
    assert len(out.recommended_hotels) >= 1


def test_activity_agent_mock(sample_request: TripRequest) -> None:
    out = ActivityAgent().mock_output(_state(sample_request))
    assert isinstance(out, ActivityOutput)
    assert out.activities and out.restaurants


def test_itinerary_agent_mock(sample_request: TripRequest) -> None:
    state = _state(sample_request)
    state["research"] = ResearchAgent().mock_output(state)
    out = ItineraryAgent().mock_output(state)
    assert isinstance(out, ItineraryOutput)
    # One day plan per requested day.
    assert len(out.days) == sample_request.days


def test_safety_agent_mock(sample_request: TripRequest) -> None:
    out = SafetyAgent().mock_output(_state(sample_request))
    assert isinstance(out, SafetyOutput)
    assert out.safety_tips and out.emergency_contacts


def test_agent_run_returns_state_update(sample_request: TripRequest) -> None:
    """In offline mode, ``run`` should yield a valid partial update + warning."""
    update = ResearchAgent().run(_state(sample_request))
    assert "research" in update
    assert isinstance(update["research"], ResearchOutput)
    assert any("mock" in w for w in update["warnings"])
