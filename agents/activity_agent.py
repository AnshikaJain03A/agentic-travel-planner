"""Activity Recommendation Agent - activities, restaurants, local experiences."""

from __future__ import annotations

from typing import Optional, Type

from pydantic import BaseModel

from agents.base import BaseAgent
from agents.formatting import request_brief, with_context
from agents.prompts import ACTIVITY_AGENT_PROMPT
from graph.state import PlannerState
from schemas.models import ActivityItem, ActivityOutput, Restaurant


class ActivityAgent(BaseAgent):
    name = "activity"
    state_key = "activities"

    @property
    def output_model(self) -> Type[BaseModel]:
        return ActivityOutput

    @property
    def system_prompt(self) -> str:
        return ACTIVITY_AGENT_PROMPT

    def retrieval_query(self, state: PlannerState) -> Optional[str]:
        req = state["request"]
        interests = " ".join(i.value for i in req.interests)
        return f"things to do activities restaurants experiences {interests} {req.destination}"

    def build_prompt(self, state: PlannerState, rag_context: str) -> str:
        req = state["request"]
        research = state.get("research")
        extra = "Suggest activities, restaurants and local experiences."
        if research is not None and research.hidden_gems:
            extra += f"\nConsider these hidden gems: {', '.join(research.hidden_gems[:4])}."
        return with_context(
            request_brief(req), state.get("memory_context", ""), rag_context, extra
        )

    def mock_output(self, state: PlannerState) -> BaseModel:
        req = state["request"]
        interests = [i.value for i in req.interests] or ["sightseeing"]
        activities = [
            ActivityItem(
                name=f"{interest.title()} experience in {req.destination}",
                category=interest,
                estimated_cost="INR 500 - 1,500",
                duration="2-3 hours",
            )
            for interest in interests[:5]
        ]
        activities.append(
            ActivityItem(
                name=f"Guided walking tour of {req.destination}",
                category="culture",
                estimated_cost="INR 800",
                duration="3 hours",
            )
        )
        return ActivityOutput(
            activities=activities,
            restaurants=[
                Restaurant(
                    name=f"{req.destination} Local Kitchen",
                    cuisine="Regional",
                    price_range="INR 400 - 800 per person",
                    must_try="Signature local thali",
                ),
                Restaurant(
                    name=f"{req.destination} Seaside Cafe",
                    cuisine="Continental & seafood",
                    price_range="INR 600 - 1,200 per person",
                    must_try="Fresh catch of the day",
                ),
                Restaurant(
                    name=f"{req.destination} Street Food Lane",
                    cuisine="Street food",
                    price_range="INR 150 - 400 per person",
                    must_try="Assorted local snacks",
                ),
            ],
            local_experiences=[
                f"Explore the local market in {req.destination}",
                f"Catch sunset at a scenic viewpoint in {req.destination}",
                f"Join a hands-on local cooking or craft workshop in {req.destination}",
            ],
        )
