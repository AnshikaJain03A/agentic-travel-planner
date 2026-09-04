"""
Streamlit frontend for the Agentic AI Travel Planner.

Talks to the FastAPI backend over HTTP. Provides:
  * a sidebar form for trip parameters,
  * a rich main view of the generated plan (overview, itinerary, budget,
    hotels, activities, safety), and
  * tabs for trip history and learned user preferences.

Run with::

    streamlit run frontend/streamlit_app.py
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 180  # multi-agent planning can take a while

TRAVEL_STYLES = ["budget", "luxury", "family", "adventure", "solo"]
INTERESTS = [
    "food", "history", "nightlife", "beaches",
    "shopping", "culture", "nature", "photography",
]

st.set_page_config(page_title="AI Travel Planner", page_icon="🧭", layout="wide")


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def api_get(path: str, params: dict | None = None):
    resp = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def api_post(path: str, payload: dict):
    resp = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def backend_status() -> dict | None:
    try:
        return api_get("/health")
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_plan(plan: dict) -> None:
    req = plan.get("request", {})
    st.success(
        f"Trip plan for **{req.get('destination')}** · {req.get('days')} days · "
        f"{req.get('travelers')} traveller(s) · {req.get('travel_style')} style"
    )

    if plan.get("overview"):
        st.markdown(f"#### Overview\n{plan['overview']}")

    if plan.get("warnings"):
        with st.expander("Notes / warnings", expanded=False):
            for w in plan["warnings"]:
                st.info(w)

    tabs = st.tabs(
        ["Itinerary", "Budget", "Hotels", "Activities", "Research", "Safety"]
    )

    # ---- Itinerary ----
    with tabs[0]:
        itinerary = plan.get("itinerary", {})
        if itinerary.get("route_summary"):
            st.caption(itinerary["route_summary"])
        for day in itinerary.get("days", []):
            st.markdown(f"**Day {day.get('day')}: {day.get('title', '')}**")
            for e in day.get("entries", []):
                line = f"- **{e.get('time','')}** — {e.get('activity','')}"
                if e.get("location"):
                    line += f" _( {e['location']} )_"
                st.markdown(line)
                if e.get("notes"):
                    st.caption(f"  {e['notes']}")
            st.divider()

    # ---- Budget ----
    with tabs[1]:
        budget = plan.get("budget", {})
        cols = st.columns(3)
        cols[0].metric("Transport", budget.get("transport", "-"))
        cols[1].metric("Accommodation", budget.get("accommodation", "-"))
        cols[2].metric("Food", budget.get("food", "-"))
        cols2 = st.columns(3)
        cols2[0].metric("Activities", budget.get("activities", "-"))
        cols2[1].metric("Miscellaneous", budget.get("miscellaneous", "-"))
        cols2[2].metric("Total", budget.get("total_estimated_cost", "-"))
        if budget.get("within_budget"):
            st.success("Estimated to be within budget.")
        else:
            st.warning("Estimate exceeds the stated budget.")
        for note in budget.get("notes", []):
            st.markdown(f"- {note}")

    # ---- Hotels ----
    with tabs[2]:
        for h in plan.get("hotels", {}).get("recommended_hotels", []):
            rating = f" · ⭐ {h['rating']}" if h.get("rating") else ""
            st.markdown(f"**{h.get('name')}** — {h.get('area','')}{rating}")
            st.caption(f"{h.get('price_per_night','')} · {h.get('why_recommended','')}")

    # ---- Activities ----
    with tabs[3]:
        activities = plan.get("activities", {})
        st.markdown("**Activities**")
        for a in activities.get("activities", []):
            st.markdown(
                f"- **{a.get('name')}** ({a.get('category','')}) — "
                f"{a.get('estimated_cost','')}, {a.get('duration','')}"
            )
        st.markdown("**Restaurants**")
        for r in activities.get("restaurants", []):
            st.markdown(
                f"- **{r.get('name')}** ({r.get('cuisine','')}) — "
                f"{r.get('price_range','')}; try _{r.get('must_try','')}_"
            )
        if activities.get("local_experiences"):
            st.markdown("**Local experiences**")
            for exp in activities["local_experiences"]:
                st.markdown(f"- {exp}")

    # ---- Research ----
    with tabs[4]:
        research = plan.get("research", {})
        if research.get("destination_summary"):
            st.markdown(research["destination_summary"])
        _bullet_section("Top attractions", research.get("top_attractions", []))
        _bullet_section("Hidden gems", research.get("hidden_gems", []))
        _bullet_section("Seasonal recommendations", research.get("seasonal_recommendations", []))
        _bullet_section("Travel tips", research.get("travel_tips", []))

    # ---- Safety ----
    with tabs[5]:
        safety = plan.get("safety", {})
        _bullet_section("Safety tips", safety.get("safety_tips", []))
        _bullet_section("Local regulations", safety.get("local_regulations", []))
        _bullet_section("Emergency contacts", safety.get("emergency_contacts", []))
        _bullet_section("Weather advice", safety.get("weather_advice", []))


def _bullet_section(title: str, items: list) -> None:
    if not items:
        return
    st.markdown(f"**{title}**")
    for item in items:
        st.markdown(f"- {item}")


# ---------------------------------------------------------------------------
# Sidebar (inputs)
# ---------------------------------------------------------------------------
st.title("🧭 Agentic AI Travel Planner")
st.caption("Seven AI agents collaborate to craft your personalised itinerary.")

status = backend_status()
with st.sidebar:
    st.header("Plan your trip")
    if status is None:
        st.error(f"Backend not reachable at {API_BASE_URL}. Start the API first.")
    else:
        llm_state = "LLM connected" if status.get("llm_available") else "Offline mock mode"
        st.caption(f"Backend: {llm_state} · {status.get('llm_model','')}")

    user_id = st.text_input("User ID", value="demo-user")
    destination = st.text_input("Destination", value="Goa")
    departure_city = st.text_input("Departure city", value="Delhi")
    days = st.slider("Number of days", 1, 21, 4)
    travelers = st.number_input("Number of travelers", 1, 20, 2)
    budget = st.number_input("Budget (INR)", min_value=1000, value=25000, step=1000)
    travel_style = st.selectbox("Travel style", TRAVEL_STYLES)
    interests = st.multiselect("Interests", INTERESTS, default=["beaches", "food", "nightlife"])
    generate = st.button("Generate plan", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Main view
# ---------------------------------------------------------------------------
view_tabs = st.tabs(["Plan", "History", "Preferences"])

with view_tabs[0]:
    if generate:
        payload = {
            "destination": destination,
            "days": int(days),
            "budget": float(budget),
            "travelers": int(travelers),
            "travel_style": travel_style,
            "interests": interests,
            "departure_city": departure_city,
            "user_id": user_id,
        }
        with st.spinner("Agents are planning your trip... this can take a moment."):
            try:
                plan = api_post("/plan-trip", payload)
                st.session_state["last_plan"] = plan
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to generate plan: {exc}")

    if "last_plan" in st.session_state:
        render_plan(st.session_state["last_plan"])
    else:
        st.info("Fill in the sidebar and click **Generate plan** to begin.")

with view_tabs[1]:
    st.subheader("Trip history")
    if st.button("Refresh history"):
        st.session_state.pop("history", None)
    try:
        history = st.session_state.get("history") or api_get(
            "/trip-history", params={"user_id": user_id}
        )
        st.session_state["history"] = history
        if not history:
            st.info("No trips planned yet for this user.")
        for item in history:
            st.markdown(
                f"- **{item['destination']}** · {item['days']} days · "
                f"INR {item['budget']:,.0f} · {item['travel_style']} "
                f"· `{item['trip_id'][:8]}`"
            )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load history: {exc}")

with view_tabs[2]:
    st.subheader("Learned preferences")
    try:
        prefs = api_get("/user-preferences", params={"user_id": user_id})
        col1, col2 = st.columns(2)
        col1.metric("Trips planned", prefs.get("trips_planned", 0))
        if prefs.get("average_budget"):
            col2.metric("Average budget", f"INR {prefs['average_budget']:,.0f}")
        _bullet_section("Favourite destinations", prefs.get("favorite_destinations", []))
        _bullet_section("Preferred styles", prefs.get("preferred_styles", []))
        _bullet_section("Common interests", prefs.get("common_interests", []))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load preferences: {exc}")
