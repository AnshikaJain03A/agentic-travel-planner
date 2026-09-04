"""Small helpers for assembling request context into prompts."""

from __future__ import annotations

from schemas.models import TripRequest


def request_brief(request: TripRequest) -> str:
    """Return a compact, prompt-friendly description of the trip request."""
    interests = ", ".join(i.value for i in request.interests) or "general sightseeing"
    return (
        f"Destination: {request.destination}\n"
        f"Departure city: {request.departure_city}\n"
        f"Trip length: {request.days} day(s)\n"
        f"Travellers: {request.travelers}\n"
        f"Total budget: INR {request.budget:,.0f} "
        f"(~INR {request.per_person_budget:,.0f} per person)\n"
        f"Travel style: {request.travel_style.value}\n"
        f"Interests: {interests}"
    )


def with_context(brief: str, memory_context: str, rag_context: str, extra: str = "") -> str:
    """Combine the request brief with memory + RAG grounding into one message."""
    sections = [brief]
    if memory_context:
        sections.append(f"\nUser memory / personalisation:\n{memory_context}")
    if rag_context:
        sections.append(f"\nGrounding knowledge (use if relevant):\n{rag_context}")
    if extra:
        sections.append(f"\n{extra}")
    return "\n".join(sections)
