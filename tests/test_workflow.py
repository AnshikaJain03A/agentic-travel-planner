"""End-to-end test of the LangGraph workflow in offline mock mode."""

from __future__ import annotations

from graph.workflow import TravelPlannerGraph
from schemas.models import TravelPlan, TripRequest


def test_full_plan_offline(sample_request: TripRequest) -> None:
    planner = TravelPlannerGraph()
    plan = planner.plan(sample_request, persist=False)

    assert isinstance(plan, TravelPlan)
    assert plan.trip_id
    # Every section should be populated by the agents (mock mode).
    assert plan.research.top_attractions
    assert plan.budget.total_estimated_cost
    assert plan.hotels.recommended_hotels
    assert plan.activities.activities
    assert len(plan.itinerary.days) == sample_request.days
    assert plan.safety.safety_tips
    assert plan.overview


def test_plan_persists_to_history(sample_request: TripRequest) -> None:
    planner = TravelPlannerGraph()
    plan = planner.plan(sample_request, persist=True)

    history = planner._memory.get_history(user_id=sample_request.user_id)
    assert any(h.trip_id == plan.trip_id for h in history)
