"""
LangGraph workflow definition.

Wires the specialist agents into a sequential graph:

    research -> budget -> hotel -> activity -> itinerary -> safety -> orchestrator

Each node receives the shared :class:`PlannerState` and returns a partial
update. The orchestrator produces the final aggregated :class:`TravelPlan`.

``TravelPlannerGraph`` is a thin, reusable facade with a ``plan`` method that the
API and tests call.
"""

from __future__ import annotations

import uuid
from typing import Optional

from langgraph.graph import END, StateGraph

from agents import (
    ActivityAgent,
    BudgetAgent,
    HotelAgent,
    ItineraryAgent,
    OrchestratorAgent,
    ResearchAgent,
    SafetyAgent,
)
from core.logging_config import get_logger
from graph.state import PlannerState
from memory.memory_manager import MemoryManager, memory_manager
from schemas.models import TravelPlan, TripRequest

logger = get_logger(__name__)


def build_graph():
    """Construct and compile the LangGraph state machine."""
    research = ResearchAgent()
    budget = BudgetAgent()
    hotel = HotelAgent()
    activity = ActivityAgent()
    itinerary = ItineraryAgent()
    safety = SafetyAgent()
    orchestrator = OrchestratorAgent()

    graph = StateGraph(PlannerState)

    # Register each agent's ``run`` method as a node.
    graph.add_node("research", research.run)
    graph.add_node("budget", budget.run)
    graph.add_node("hotel", hotel.run)
    graph.add_node("activity", activity.run)
    graph.add_node("itinerary", itinerary.run)
    graph.add_node("safety", safety.run)
    graph.add_node("orchestrator", orchestrator.run)

    # Sequential edges (matching the documented workflow).
    graph.set_entry_point("research")
    graph.add_edge("research", "budget")
    graph.add_edge("budget", "hotel")
    graph.add_edge("hotel", "activity")
    graph.add_edge("activity", "itinerary")
    graph.add_edge("itinerary", "safety")
    graph.add_edge("safety", "orchestrator")
    graph.add_edge("orchestrator", END)

    return graph.compile()


class TravelPlannerGraph:
    """High-level facade around the compiled LangGraph workflow."""

    def __init__(self, memory: Optional[MemoryManager] = None) -> None:
        self._app = build_graph()
        self._memory = memory or memory_manager

    def plan(self, request: TripRequest, persist: bool = True) -> TravelPlan:
        """Run the full multi-agent workflow for ``request``.

        Args:
            request: the validated trip request.
            persist: whether to save the resulting plan to long-term memory.

        Returns:
            The aggregated :class:`TravelPlan`.
        """
        trip_id = uuid.uuid4().hex
        memory_context = self._memory.build_memory_context(request)
        logger.info("Starting plan %s for %s", trip_id, request.destination)

        initial_state: PlannerState = {
            "request": request,
            "trip_id": trip_id,
            "memory_context": memory_context,
            "warnings": [],
        }

        final_state = self._app.invoke(initial_state)
        plan: TravelPlan = final_state["final_plan"]

        if persist:
            try:
                self._memory.save_trip(plan)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to persist trip %s", trip_id)

        logger.info("Completed plan %s (%d warnings)", trip_id, len(plan.warnings))
        return plan


_graph_singleton: Optional[TravelPlannerGraph] = None


def get_planner() -> TravelPlannerGraph:
    """Return the shared compiled planner graph."""
    global _graph_singleton
    if _graph_singleton is None:
        _graph_singleton = TravelPlannerGraph()
    return _graph_singleton
