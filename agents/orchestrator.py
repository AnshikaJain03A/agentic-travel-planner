"""
Orchestrator Agent.

The orchestrator is the final LangGraph node. It does not produce a new
specialist slice; instead it:
  * synthesises a friendly trip overview (via the LLM, with a mock fallback), and
  * aggregates every agent's structured output into a single :class:`TravelPlan`.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agents.formatting import request_brief
from agents.prompts import ORCHESTRATOR_PROMPT
from core.config import settings
from core.logging_config import get_logger
from graph.state import PlannerState
from schemas.models import (
    ActivityOutput,
    BudgetOutput,
    HotelOutput,
    ItineraryOutput,
    ResearchOutput,
    SafetyOutput,
    TravelPlan,
)
from services.llm import get_llm


class _Overview(BaseModel):
    """Tiny structured wrapper for the synthesised overview text."""

    overview: str = Field(default="")


class OrchestratorAgent:
    name = "orchestrator"

    def __init__(self) -> None:
        self.llm = get_llm()
        self.logger = get_logger("agent.orchestrator")

    # ------------------------------------------------------------------
    def _build_overview(self, state: PlannerState) -> str:
        req = state["request"]
        research = state.get("research", ResearchOutput())
        budget = state.get("budget", BudgetOutput())

        if not self.llm.is_available():
            return self._mock_overview(state)

        human = (
            f"{request_brief(req)}\n\n"
            f"Destination summary: {research.destination_summary}\n"
            f"Total estimated cost: {budget.total_estimated_cost} "
            f"(within budget: {budget.within_budget}).\n"
            f"Top attractions: {', '.join(research.top_attractions[:5])}.\n"
            "Write the trip overview."
        )
        try:
            result = self.llm.structured(_Overview, ORCHESTRATOR_PROMPT, human)
            return result.overview or self._mock_overview(state)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Overview synthesis failed (%s); using fallback", exc)
            if not settings.allow_mock_fallback:
                raise
            return self._mock_overview(state)

    def _mock_overview(self, state: PlannerState) -> str:
        req = state["request"]
        budget = state.get("budget", BudgetOutput())
        interests = ", ".join(i.value for i in req.interests) or "exploration"
        fit = "comfortably within" if budget.within_budget else "slightly above"
        return (
            f"Here is your {req.days}-day {req.travel_style.value} trip to "
            f"{req.destination} for {req.travelers} traveller(s), built around "
            f"{interests}. The estimated cost of {budget.total_estimated_cost} sits "
            f"{fit} your INR {req.budget:,.0f} budget. Expect a balanced mix of must-see "
            f"highlights, local flavour and relaxation - enjoy the trip!"
        )

    # ------------------------------------------------------------------
    def run(self, state: PlannerState) -> dict:
        self.logger.info("Running orchestrator: aggregating final plan")
        req = state["request"]
        overview = self._build_overview(state)

        plan = TravelPlan(
            trip_id=state["trip_id"],
            request=req,
            overview=overview,
            research=state.get("research", ResearchOutput()),
            budget=state.get("budget", BudgetOutput()),
            hotels=state.get("hotels", HotelOutput()),
            activities=state.get("activities", ActivityOutput()),
            itinerary=state.get("itinerary", ItineraryOutput()),
            safety=state.get("safety", SafetyOutput()),
            warnings=state.get("warnings", []),
        )
        return {"final_plan": plan}
