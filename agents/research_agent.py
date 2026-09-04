"""Travel Research Agent - gathers destination knowledge and tips."""

from __future__ import annotations

from typing import Optional, Type

from pydantic import BaseModel

from agents.base import BaseAgent
from agents.formatting import request_brief, with_context
from agents.prompts import RESEARCH_AGENT_PROMPT
from graph.state import PlannerState
from schemas.models import ResearchOutput


class ResearchAgent(BaseAgent):
    name = "research"
    state_key = "research"

    @property
    def output_model(self) -> Type[BaseModel]:
        return ResearchOutput

    @property
    def system_prompt(self) -> str:
        return RESEARCH_AGENT_PROMPT

    def retrieval_query(self, state: PlannerState) -> Optional[str]:
        req = state["request"]
        interests = " ".join(i.value for i in req.interests)
        return f"{req.destination} attractions hidden gems travel tips {interests}"

    def build_prompt(self, state: PlannerState, rag_context: str) -> str:
        req = state["request"]
        return with_context(
            request_brief(req),
            state.get("memory_context", ""),
            rag_context,
            extra="Research this destination and return the structured fields.",
        )

    def mock_output(self, state: PlannerState) -> BaseModel:
        req = state["request"]
        d = req.destination
        interest_list = [i.value for i in req.interests] or ["sightseeing"]
        return ResearchOutput(
            destination_summary=(
                f"{d} is a popular destination well-suited to a {req.travel_style.value} "
                f"trip focused on {', '.join(interest_list)}."
            ),
            top_attractions=[
                f"{d} City Centre",
                f"{d} Main Beach / Landmark",
                f"Historic Quarter of {d}",
                f"{d} Viewpoint",
                f"Local Market in {d}",
                f"{d} Cultural Museum",
            ],
            hidden_gems=[
                f"Quiet neighbourhood cafe in {d}",
                f"Sunset spot away from the crowds in {d}",
                f"Family-run eatery loved by locals in {d}",
                f"Lesser-known trail near {d}",
            ],
            seasonal_recommendations=[
                "Visit outdoor attractions in the morning to avoid afternoon heat.",
                "Carry a light layer for cooler evenings.",
                "Book popular experiences in advance during peak season.",
            ],
            travel_tips=[
                f"Use local transport or a rented scooter to get around {d} affordably.",
                "Carry some cash for small vendors and markets.",
                "Stay hydrated and use sunscreen.",
                "Learn a few local phrases - it goes a long way.",
            ],
        )
