"""Itinerary Agent - builds an optimised, day-by-day plan."""

from __future__ import annotations

from typing import Type

from pydantic import BaseModel

from agents.base import BaseAgent
from agents.formatting import request_brief, with_context
from agents.prompts import ITINERARY_AGENT_PROMPT
from graph.state import PlannerState
from schemas.models import DayPlan, ItineraryEntry, ItineraryOutput


class ItineraryAgent(BaseAgent):
    name = "itinerary"
    state_key = "itinerary"

    @property
    def output_model(self) -> Type[BaseModel]:
        return ItineraryOutput

    @property
    def system_prompt(self) -> str:
        return ITINERARY_AGENT_PROMPT

    def build_prompt(self, state: PlannerState, rag_context: str) -> str:
        req = state["request"]
        research = state.get("research")
        activities = state.get("activities")
        hotels = state.get("hotels")

        details = []
        if research is not None:
            details.append("Top attractions: " + ", ".join(research.top_attractions[:8]))
            if research.hidden_gems:
                details.append("Hidden gems: " + ", ".join(research.hidden_gems[:4]))
        if activities is not None and activities.activities:
            details.append(
                "Suggested activities: "
                + ", ".join(a.name for a in activities.activities[:8])
            )
        if hotels is not None and hotels.recommended_hotels:
            details.append(f"Base hotel area: {hotels.recommended_hotels[0].area}")

        extra = (
            f"Build a {req.days}-day itinerary. Group nearby places together to "
            "minimise travel. Include arrival on day 1 and departure on the last day.\n"
            + "\n".join(details)
        )
        return with_context(
            request_brief(req), state.get("memory_context", ""), rag_context, extra
        )

    def mock_output(self, state: PlannerState) -> BaseModel:
        req = state["request"]
        research = state.get("research")
        attractions = (research.top_attractions if research else []) or [
            f"{req.destination} highlight"
        ]

        days: list[DayPlan] = []
        for day_num in range(1, req.days + 1):
            # Rotate through attractions to spread them across days.
            idx = (day_num - 1) % len(attractions)
            spot = attractions[idx]
            entries = [
                ItineraryEntry(
                    time="Morning",
                    activity=("Arrive and check in, then explore" if day_num == 1 else f"Visit {spot}"),
                    location=req.destination if day_num == 1 else spot,
                    notes="Start early to beat the crowds.",
                ),
                ItineraryEntry(
                    time="Afternoon",
                    activity="Lunch at a local restaurant, then nearby sightseeing",
                    location=spot,
                    notes="Keep activities clustered to reduce travel time.",
                ),
                ItineraryEntry(
                    time="Evening",
                    activity=("Departure" if day_num == req.days else "Relax, dinner and local stroll"),
                    location=req.destination,
                    notes="Wind down and enjoy the local atmosphere.",
                ),
            ]
            days.append(
                DayPlan(day=day_num, title=f"Day {day_num}: {spot}", entries=entries)
            )

        return ItineraryOutput(
            days=days,
            route_summary=(
                "Days are organised so that nearby attractions are visited together, "
                "minimising back-and-forth travel across the destination."
            ),
        )
