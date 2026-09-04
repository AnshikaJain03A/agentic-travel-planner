"""API tests using FastAPI's TestClient (offline mock mode)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "llm_available" in body


def test_plan_trip_endpoint() -> None:
    payload = {
        "destination": "Goa",
        "days": 3,
        "budget": 25000,
        "travelers": 2,
        "travel_style": "budget",
        "interests": ["beaches", "food"],
        "departure_city": "Delhi",
        "user_id": "api-user",
    }
    resp = client.post("/plan-trip", json=payload)
    assert resp.status_code == 200
    plan = resp.json()
    assert plan["request"]["destination"] == "Goa"
    assert len(plan["itinerary"]["days"]) == 3
    assert plan["budget"]["total_estimated_cost"]


def test_plan_trip_validation_error() -> None:
    resp = client.post("/plan-trip", json={"destination": "Goa"})
    assert resp.status_code == 422


def test_history_and_preferences_endpoints() -> None:
    payload = {
        "destination": "Jaipur",
        "days": 2,
        "budget": 18000,
        "travelers": 2,
        "travel_style": "family",
        "interests": ["history", "culture"],
        "departure_city": "Mumbai",
        "user_id": "api-user-2",
    }
    client.post("/plan-trip", json=payload)

    hist = client.get("/trip-history", params={"user_id": "api-user-2"})
    assert hist.status_code == 200
    assert len(hist.json()) >= 1

    prefs = client.get("/user-preferences", params={"user_id": "api-user-2"})
    assert prefs.status_code == 200
    assert prefs.json()["trips_planned"] >= 1


def test_get_trip_not_found() -> None:
    resp = client.get("/trip/does-not-exist")
    assert resp.status_code == 404
