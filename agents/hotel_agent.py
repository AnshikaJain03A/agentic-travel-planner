"""Hotel Recommendation Agent - matches stays to budget and style."""

from __future__ import annotations

from typing import Optional, Type

from pydantic import BaseModel

from agents.base import BaseAgent
from agents.formatting import request_brief, with_context
from agents.prompts import HOTEL_AGENT_PROMPT
from graph.state import PlannerState
from schemas.models import Hotel, HotelOutput, TravelStyle


class HotelAgent(BaseAgent):
    name = "hotel"
    state_key = "hotels"

    @property
    def output_model(self) -> Type[BaseModel]:
        return HotelOutput

    @property
    def system_prompt(self) -> str:
        return HOTEL_AGENT_PROMPT

    def retrieval_query(self, state: PlannerState) -> Optional[str]:
        req = state["request"]
        return f"accommodation hotels stay {req.travel_style.value} {req.destination}"

    def build_prompt(self, state: PlannerState, rag_context: str) -> str:
        req = state["request"]
        budget = state.get("budget")
        extra = "Recommend stays matching the per-night budget and style."
        if budget is not None:
            extra += f"\nPlanned accommodation budget: {budget.accommodation} total for {req.days} night(s)."
        return with_context(
            request_brief(req), state.get("memory_context", ""), rag_context, extra
        )

    def mock_output(self, state: PlannerState) -> BaseModel:
        req = state["request"]
        nights = max(req.days - 1, 1)
        per_night = (req.budget * 0.30) / nights
        style = req.travel_style

        def band(multiplier: float) -> str:
            low = per_night * multiplier * 0.85
            high = per_night * multiplier * 1.15
            return f"INR {low:,.0f} - {high:,.0f}"

        style_label = {
            TravelStyle.LUXURY: "Luxury Resort",
            TravelStyle.BUDGET: "Budget Guesthouse",
            TravelStyle.FAMILY: "Family Hotel",
            TravelStyle.ADVENTURE: "Adventure Camp",
            TravelStyle.SOLO: "Hostel",
        }.get(style, "Hotel")

        return HotelOutput(
            recommended_hotels=[
                Hotel(
                    name=f"{req.destination} {style_label} Central",
                    area="City Centre",
                    price_per_night=band(1.0),
                    rating=4.2,
                    why_recommended=f"Central, well-rated and matches a {style.value} budget.",
                ),
                Hotel(
                    name=f"{req.destination} Bay View {style_label}",
                    area="Near main attractions",
                    price_per_night=band(1.15),
                    rating=4.4,
                    why_recommended="Great location close to the top sights and dining.",
                ),
                Hotel(
                    name=f"{req.destination} Cozy {style_label}",
                    area="Quiet neighbourhood",
                    price_per_night=band(0.8),
                    rating=4.0,
                    why_recommended="Best value option, calm area, friendly hosts.",
                ),
            ]
        )
