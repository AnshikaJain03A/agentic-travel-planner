"""Budget Planning Agent - estimates a realistic cost breakdown."""

from __future__ import annotations

from typing import Optional, Type

from pydantic import BaseModel

from agents.base import BaseAgent
from agents.formatting import request_brief, with_context
from agents.prompts import BUDGET_AGENT_PROMPT
from graph.state import PlannerState
from schemas.models import BudgetOutput


class BudgetAgent(BaseAgent):
    name = "budget"
    state_key = "budget"

    @property
    def output_model(self) -> Type[BaseModel]:
        return BudgetOutput

    @property
    def system_prompt(self) -> str:
        return BUDGET_AGENT_PROMPT

    def retrieval_query(self, state: PlannerState) -> Optional[str]:
        req = state["request"]
        return f"budget planning costs transport accommodation food activities {req.destination}"

    def build_prompt(self, state: PlannerState, rag_context: str) -> str:
        req = state["request"]
        research = state.get("research")
        extra = "Estimate the full-group, full-trip budget breakdown."
        if research is not None:
            extra += f"\nKnown top attractions: {', '.join(research.top_attractions[:5])}."
        return with_context(
            request_brief(req), state.get("memory_context", ""), rag_context, extra
        )

    def mock_output(self, state: PlannerState) -> BaseModel:
        req = state["request"]
        budget = req.budget
        # Rule-of-thumb allocation used as deterministic offline estimate.
        transport = budget * 0.30
        accommodation = budget * 0.30
        food = budget * 0.20
        activities = budget * 0.12
        misc = budget * 0.08
        total = transport + accommodation + food + activities + misc

        def fmt(v: float) -> str:
            return f"INR {v:,.0f}"

        return BudgetOutput(
            transport=fmt(transport),
            accommodation=fmt(accommodation),
            food=fmt(food),
            activities=fmt(activities),
            miscellaneous=fmt(misc),
            total_estimated_cost=fmt(total),
            within_budget=total <= budget * 1.02,
            notes=[
                "Allocation follows a balanced 30/30/20/12/8 split.",
                "Travel in shoulder season to reduce transport and stay costs.",
                "A ~8% buffer is included for miscellaneous expenses.",
            ],
        )
