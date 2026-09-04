-- ============================================================
--  Travel Planner - SQLite schema
--  Stores user preferences, trip history and full saved plans.
-- ============================================================

PRAGMA foreign_keys = ON;

-- One row per known user. Aggregated memory lives here.
CREATE TABLE IF NOT EXISTS users (
    user_id            TEXT PRIMARY KEY,
    created_at         TEXT NOT NULL DEFAULT (datetime('now')),
    trips_planned      INTEGER NOT NULL DEFAULT 0,
    average_budget     REAL,
    -- JSON encoded aggregate lists.
    favorite_destinations TEXT NOT NULL DEFAULT '[]',
    preferred_styles      TEXT NOT NULL DEFAULT '[]',
    common_interests      TEXT NOT NULL DEFAULT '[]'
);

-- One row per planned trip (history + the full plan payload as JSON).
CREATE TABLE IF NOT EXISTS trips (
    trip_id        TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL,
    destination    TEXT NOT NULL,
    departure_city TEXT NOT NULL,
    days           INTEGER NOT NULL,
    budget         REAL NOT NULL,
    travelers      INTEGER NOT NULL,
    travel_style   TEXT NOT NULL,
    interests      TEXT NOT NULL DEFAULT '[]',  -- JSON array
    plan_json      TEXT NOT NULL,               -- full TravelPlan as JSON
    created_at     TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_trips_user_id  ON trips (user_id);
CREATE INDEX IF NOT EXISTS idx_trips_created  ON trips (created_at);
