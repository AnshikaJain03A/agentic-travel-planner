"""Travel Safety Agent - safety, regulations, emergencies and weather."""

from __future__ import annotations

from typing import Optional, Type

from pydantic import BaseModel

from agents.base import BaseAgent
from agents.formatting import request_brief, with_context
from agents.prompts import SAFETY_AGENT_PROMPT
from graph.state import PlannerState
from schemas.models import SafetyOutput


class SafetyAgent(BaseAgent):
    name = "safety"
    state_key = "safety"

    @property
    def output_model(self) -> Type[BaseModel]:
        return SafetyOutput

    @property
    def system_prompt(self) -> str:
        return SAFETY_AGENT_PROMPT

    def retrieval_query(self, state: PlannerState) -> Optional[str]:
        req = state["request"]
        return f"safety tips regulations emergency weather {req.destination}"

    def build_prompt(self, state: PlannerState, rag_context: str) -> str:
        req = state["request"]
        return with_context(
            request_brief(req),
            state.get("memory_context", ""),
            rag_context,
            extra="Provide safety, regulation, emergency and weather guidance.",
        )

    def mock_output(self, state: PlannerState) -> BaseModel:
        req = state["request"]
        return SafetyOutput(
            safety_tips=[
                "Keep digital and physical copies of your ID and tickets.",
                "Avoid displaying expensive items; stay aware in crowded areas.",
                f"Use reputable transport options when moving around {req.destination}.",
                "Share your live location/itinerary with someone you trust.",
            ],
            local_regulations=[
                "Respect dress codes at religious and cultural sites.",
                "Always carry a valid photo ID.",
                "Follow local rules on alcohol, smoking and public conduct.",
            ],
            emergency_contacts=[
                "All-India emergency: 112",
                "Ambulance: 108",
                "Police: 100",
                "Tourist helpline: 1363",
            ],
            weather_advice=[
                "Check the forecast before outdoor activities.",
                "Carry sunscreen, a hat and water for hot days.",
                "Pack a light rain layer in case of showers.",
            ],
        )
