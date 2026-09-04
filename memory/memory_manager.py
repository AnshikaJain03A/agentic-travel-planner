"""
Long-term memory management backed by SQLite.

Responsibilities:
* Persist every generated trip plan (history).
* Maintain a rolling, aggregated view of each user's preferences:
  favourite destinations, preferred travel styles, common interests and
  average budget. These are surfaced back to the agents so that subsequent
  plans feel personalised.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from typing import List, Optional

from core.logging_config import get_logger
from database.db import get_connection, init_db
from schemas.models import (
    TravelPlan,
    TripHistoryItem,
    TripRequest,
    UserPreferences,
)

logger = get_logger(__name__)


class MemoryManager:
    """High-level CRUD over the persistence layer for user memory."""

    def __init__(self) -> None:
        init_db()

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------
    def _ensure_user(self, conn, user_id: str) -> None:
        conn.execute(
            "INSERT OR IGNORE INTO users (user_id) VALUES (?)",
            (user_id,),
        )

    def get_preferences(self, user_id: str) -> UserPreferences:
        """Return the aggregated preferences for ``user_id`` (empty if new)."""
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()

        if row is None:
            return UserPreferences(user_id=user_id)

        return UserPreferences(
            user_id=user_id,
            favorite_destinations=json.loads(row["favorite_destinations"]),
            preferred_styles=json.loads(row["preferred_styles"]),
            common_interests=json.loads(row["common_interests"]),
            average_budget=row["average_budget"],
            trips_planned=row["trips_planned"],
        )

    # ------------------------------------------------------------------
    # Trips
    # ------------------------------------------------------------------
    def save_trip(self, plan: TravelPlan) -> None:
        """Persist a full plan and refresh the user's aggregated preferences."""
        req = plan.request
        with get_connection() as conn:
            self._ensure_user(conn, req.user_id)
            conn.execute(
                """
                INSERT OR REPLACE INTO trips (
                    trip_id, user_id, destination, departure_city, days,
                    budget, travelers, travel_style, interests, plan_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    plan.trip_id,
                    req.user_id,
                    req.destination,
                    req.departure_city,
                    req.days,
                    req.budget,
                    req.travelers,
                    req.travel_style.value,
                    json.dumps([i.value for i in req.interests]),
                    plan.model_dump_json(),
                    plan.created_at.isoformat(),
                ),
            )
            self._recompute_preferences(conn, req.user_id)
        logger.info("Saved trip %s for user %s", plan.trip_id, req.user_id)

    def get_trip(self, trip_id: str) -> Optional[TravelPlan]:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT plan_json FROM trips WHERE trip_id = ?", (trip_id,)
            ).fetchone()
        if row is None:
            return None
        return TravelPlan.model_validate_json(row["plan_json"])

    def get_history(self, user_id: Optional[str] = None, limit: int = 50) -> List[TripHistoryItem]:
        """Return recent trips, optionally filtered by user."""
        query = (
            "SELECT trip_id, user_id, destination, days, budget, travelers, "
            "travel_style, created_at FROM trips"
        )
        params: list = []
        if user_id:
            query += " WHERE user_id = ?"
            params.append(user_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with get_connection() as conn:
            rows = conn.execute(query, params).fetchall()

        return [
            TripHistoryItem(
                trip_id=r["trip_id"],
                user_id=r["user_id"],
                destination=r["destination"],
                days=r["days"],
                budget=r["budget"],
                travelers=r["travelers"],
                travel_style=r["travel_style"],
                created_at=datetime.fromisoformat(r["created_at"]),
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Preference aggregation
    # ------------------------------------------------------------------
    def _recompute_preferences(self, conn, user_id: str) -> None:
        """Recalculate aggregate preferences from the user's trip history."""
        rows = conn.execute(
            "SELECT destination, budget, travel_style, interests FROM trips WHERE user_id = ?",
            (user_id,),
        ).fetchall()

        if not rows:
            return

        dest_counter: Counter[str] = Counter()
        style_counter: Counter[str] = Counter()
        interest_counter: Counter[str] = Counter()
        budgets: list[float] = []

        for r in rows:
            dest_counter[r["destination"]] += 1
            style_counter[r["travel_style"]] += 1
            budgets.append(float(r["budget"]))
            for interest in json.loads(r["interests"]):
                interest_counter[interest] += 1

        favorite_destinations = [d for d, _ in dest_counter.most_common(5)]
        preferred_styles = [s for s, _ in style_counter.most_common(3)]
        common_interests = [i for i, _ in interest_counter.most_common(5)]
        average_budget = round(sum(budgets) / len(budgets), 2)

        conn.execute(
            """
            UPDATE users SET
                trips_planned = ?,
                average_budget = ?,
                favorite_destinations = ?,
                preferred_styles = ?,
                common_interests = ?
            WHERE user_id = ?
            """,
            (
                len(rows),
                average_budget,
                json.dumps(favorite_destinations),
                json.dumps(preferred_styles),
                json.dumps(common_interests),
                user_id,
            ),
        )

    # ------------------------------------------------------------------
    # Prompt helper
    # ------------------------------------------------------------------
    def build_memory_context(self, request: TripRequest) -> str:
        """Produce a short natural-language summary of the user's history.

        Injected into agent prompts so plans become progressively personalised.
        """
        prefs = self.get_preferences(request.user_id)
        if prefs.trips_planned == 0:
            return "This is the user's first trip with us; no prior history available."

        parts = [f"The user has planned {prefs.trips_planned} trip(s) before."]
        if prefs.favorite_destinations:
            parts.append("Favourite destinations: " + ", ".join(prefs.favorite_destinations) + ".")
        if prefs.preferred_styles:
            parts.append("Usual travel styles: " + ", ".join(prefs.preferred_styles) + ".")
        if prefs.common_interests:
            parts.append("Frequent interests: " + ", ".join(prefs.common_interests) + ".")
        if prefs.average_budget:
            parts.append(f"Typical budget around INR {prefs.average_budget:,.0f}.")
        return " ".join(parts)


# Module-level singleton for convenience.
memory_manager = MemoryManager()
