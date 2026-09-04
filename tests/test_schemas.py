"""Validation tests for the Pydantic data contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas.models import Interest, TravelStyle, TripRequest


def test_trip_request_valid(sample_request: TripRequest) -> None:
    assert sample_request.destination == "Goa"
    assert sample_request.per_person_budget == 12500


def test_trip_request_strips_whitespace() -> None:
    req = TripRequest(
        destination="  Goa  ",
        days=2,
        budget=10000,
        travelers=1,
        departure_city="  Delhi ",
    )
    assert req.destination == "Goa"
    assert req.departure_city == "Delhi"


def test_trip_request_dedupes_interests() -> None:
    req = TripRequest(
        destination="Goa",
        days=2,
        budget=10000,
        travelers=1,
        departure_city="Delhi",
        interests=[Interest.FOOD, Interest.FOOD, Interest.BEACHES],
    )
    assert req.interests == [Interest.FOOD, Interest.BEACHES]


@pytest.mark.parametrize("days", [0, -1, 100])
def test_trip_request_invalid_days(days: int) -> None:
    with pytest.raises(ValidationError):
        TripRequest(
            destination="Goa",
            days=days,
            budget=10000,
            travelers=1,
            departure_city="Delhi",
        )


def test_trip_request_invalid_budget() -> None:
    with pytest.raises(ValidationError):
        TripRequest(
            destination="Goa",
            days=2,
            budget=0,
            travelers=1,
            departure_city="Delhi",
        )


def test_travel_style_enum() -> None:
    assert TravelStyle("luxury") == TravelStyle.LUXURY
