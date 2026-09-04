"""
Pydantic data contracts shared across agents, the LangGraph workflow,
the FastAPI layer and the Streamlit frontend.

Keeping every structured payload in one place means the agents emit
exactly what the API returns and the UI consumes, with validation for free.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class TravelStyle(str, Enum):
    LUXURY = "luxury"
    BUDGET = "budget"
    FAMILY = "family"
    ADVENTURE = "adventure"
    SOLO = "solo"


class Interest(str, Enum):
    FOOD = "food"
    HISTORY = "history"
    NIGHTLIFE = "nightlife"
    BEACHES = "beaches"
    SHOPPING = "shopping"
    CULTURE = "culture"
    NATURE = "nature"
    PHOTOGRAPHY = "photography"


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------
class TripRequest(BaseModel):
    """User-supplied travel planning request."""

    destination: str = Field(..., min_length=2, examples=["Goa"])
    days: int = Field(..., ge=1, le=30, examples=[4])
    budget: float = Field(..., gt=0, description="Total budget in INR", examples=[25000])
    travelers: int = Field(..., ge=1, le=20, examples=[2])
    travel_style: TravelStyle = Field(default=TravelStyle.BUDGET)
    interests: List[Interest] = Field(default_factory=list, examples=[["beaches", "food", "nightlife"]])
    departure_city: str = Field(..., min_length=2, examples=["Delhi"])
    user_id: str = Field(default="anonymous", description="Stable identifier used for memory")

    @field_validator("destination", "departure_city")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()

    @field_validator("interests")
    @classmethod
    def _dedupe(cls, value: List[Interest]) -> List[Interest]:
        seen: list[Interest] = []
        for item in value:
            if item not in seen:
                seen.append(item)
        return seen

    @property
    def per_person_budget(self) -> float:
        return self.budget / max(self.travelers, 1)


# ---------------------------------------------------------------------------
# Per-agent structured outputs
# ---------------------------------------------------------------------------
class ResearchOutput(BaseModel):
    """Output of the Travel Research Agent."""

    destination_summary: str = Field(default="")
    top_attractions: List[str] = Field(default_factory=list)
    hidden_gems: List[str] = Field(default_factory=list)
    seasonal_recommendations: List[str] = Field(default_factory=list)
    travel_tips: List[str] = Field(default_factory=list)


class BudgetOutput(BaseModel):
    """Output of the Budget Planning Agent. Costs are human-readable strings."""

    transport: str = Field(default="")
    accommodation: str = Field(default="")
    food: str = Field(default="")
    activities: str = Field(default="")
    miscellaneous: str = Field(default="")
    total_estimated_cost: str = Field(default="")
    notes: List[str] = Field(default_factory=list)
    within_budget: bool = Field(default=True)


class Hotel(BaseModel):
    name: str
    area: str = Field(default="")
    price_per_night: str = Field(default="")
    rating: Optional[float] = Field(default=None)
    why_recommended: str = Field(default="")


class HotelOutput(BaseModel):
    """Output of the Hotel Recommendation Agent."""

    recommended_hotels: List[Hotel] = Field(default_factory=list)


class Restaurant(BaseModel):
    name: str
    cuisine: str = Field(default="")
    price_range: str = Field(default="")
    must_try: str = Field(default="")


class ActivityItem(BaseModel):
    name: str
    category: str = Field(default="")
    estimated_cost: str = Field(default="")
    duration: str = Field(default="")


class ActivityOutput(BaseModel):
    """Output of the Activity Recommendation Agent."""

    activities: List[ActivityItem] = Field(default_factory=list)
    restaurants: List[Restaurant] = Field(default_factory=list)
    local_experiences: List[str] = Field(default_factory=list)


class ItineraryEntry(BaseModel):
    time: str = Field(default="", description="e.g. 'Morning', '09:00'")
    activity: str = Field(default="")
    location: str = Field(default="")
    notes: str = Field(default="")


class DayPlan(BaseModel):
    day: int
    title: str = Field(default="")
    entries: List[ItineraryEntry] = Field(default_factory=list)


class ItineraryOutput(BaseModel):
    """Output of the Itinerary Agent."""

    days: List[DayPlan] = Field(default_factory=list)
    route_summary: str = Field(default="")


class SafetyOutput(BaseModel):
    """Output of the Travel Safety Agent."""

    safety_tips: List[str] = Field(default_factory=list)
    local_regulations: List[str] = Field(default_factory=list)
    emergency_contacts: List[str] = Field(default_factory=list)
    weather_advice: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Aggregated final plan
# ---------------------------------------------------------------------------
class TravelPlan(BaseModel):
    """Final aggregated plan produced by the Orchestrator Agent."""

    trip_id: str
    request: TripRequest
    overview: str = Field(default="")
    research: ResearchOutput = Field(default_factory=ResearchOutput)
    budget: BudgetOutput = Field(default_factory=BudgetOutput)
    hotels: HotelOutput = Field(default_factory=HotelOutput)
    activities: ActivityOutput = Field(default_factory=ActivityOutput)
    itinerary: ItineraryOutput = Field(default_factory=ItineraryOutput)
    safety: SafetyOutput = Field(default_factory=SafetyOutput)
    warnings: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Memory / persistence DTOs
# ---------------------------------------------------------------------------
class UserPreferences(BaseModel):
    user_id: str
    favorite_destinations: List[str] = Field(default_factory=list)
    preferred_styles: List[str] = Field(default_factory=list)
    common_interests: List[str] = Field(default_factory=list)
    average_budget: Optional[float] = Field(default=None)
    trips_planned: int = Field(default=0)


class TripHistoryItem(BaseModel):
    trip_id: str
    user_id: str
    destination: str
    days: int
    budget: float
    travelers: int
    travel_style: str
    created_at: datetime
