# 🧭 Agentic AI Travel Planner

A production-grade, **multi-agent** travel planning assistant. Seven specialised
AI agents collaborate through a **LangGraph** workflow to turn a few preferences
into a complete, personalised itinerary — with budget breakdown, hotel and
activity recommendations, day-by-day plans and safety guidance.

Built with **FastAPI**, **LangGraph + LangChain**, **ChromaDB** (RAG),
**SQLite** (long-term memory) and a **Streamlit** frontend. Configurable to use
**OpenAI GPT-4o** or **Anthropic Claude Sonnet**, and runs fully offline in a
deterministic *mock mode* when no API key is provided (great for dev/CI/demos).

---

## ✨ Features

- **7 collaborating agents**: Research, Budget, Hotel, Activity, Itinerary,
  Safety and an Orchestrator that aggregates everything.
- **LangGraph workflow** with a shared, typed state object.
- **RAG**: ChromaDB vector store with document ingestion, sentence-aware
  chunking and similarity retrieval (OpenAI embeddings, local fallback).
- **Long-term memory**: SQLite stores trip history and learns each user's
  favourite destinations, styles, interests and budget patterns.
- **Provider-agnostic LLM layer** (OpenAI / Anthropic) with retries.
- **Graceful offline mode**: every agent has a deterministic mock so the system
  always produces a usable plan even without API keys.
- **Typed end-to-end** with Pydantic; the API, agents and UI share one schema.
- **Docker-ready**: `Dockerfile` + `docker-compose.yml` run API + UI together.
- **Tested**: unit + integration tests covering schemas, RAG, agents, memory,
  the full workflow and the API.

---

## 🏗️ Architecture

```
                ┌──────────────────────────────────────────────┐
                │                Streamlit UI                   │
                │   (sidebar inputs · plan · history · prefs)   │
                └───────────────────────┬──────────────────────┘
                                        │ HTTP (requests)
                                        ▼
                ┌──────────────────────────────────────────────┐
                │                 FastAPI API                   │
                │  /plan-trip  /trip-history  /user-preferences │
                └───────────────────────┬──────────────────────┘
                                        │
                                        ▼
                ┌──────────────────────────────────────────────┐
                │            LangGraph Workflow                 │
                │                                               │
                │  research → budget → hotel → activity →       │
                │  itinerary → safety → orchestrator → PLAN     │
                └───────┬───────────────┬──────────────────────┘
                        │               │
          ┌─────────────▼──┐     ┌──────▼───────────┐
          │  RAG (Chroma)  │     │  Memory (SQLite) │
          │  travel KB     │     │  history + prefs │
          └────────────────┘     └──────────────────┘
                        ▲
              ┌─────────┴──────────┐
              │  LLM (OpenAI /     │
              │  Anthropic)        │
              └────────────────────┘
```

### LangGraph workflow

```
User Input
   ↓
Research Agent      → top attractions, hidden gems, tips (RAG-grounded)
   ↓
Budget Agent        → transport / stay / food / activities breakdown
   ↓
Hotel Agent         → stays matched to budget + style
   ↓
Activity Agent      → activities, restaurants, local experiences
   ↓
Itinerary Agent     → optimised day-by-day plan
   ↓
Safety Agent        → safety, regulations, emergencies, weather
   ↓
Orchestrator Agent  → synthesised overview + aggregated Final Travel Plan
```

---

## 📁 Project structure

```
travel-planner/
├── agents/            # The 7 agents + prompts + base class
│   ├── base.py            # shared agent lifecycle (RAG → LLM → mock fallback)
│   ├── prompts.py         # all system prompts
│   ├── research_agent.py
│   ├── budget_agent.py
│   ├── hotel_agent.py
│   ├── activity_agent.py
│   ├── itinerary_agent.py
│   ├── safety_agent.py
│   └── orchestrator.py
├── graph/             # LangGraph state + compiled workflow
│   ├── state.py
│   └── workflow.py
├── services/          # LLM provider abstraction
│   └── llm.py
├── schemas/           # Pydantic data contracts (shared everywhere)
│   └── models.py
├── memory/            # SQLite-backed long-term memory
│   └── memory_manager.py
├── rag/               # ChromaDB knowledge base + ingestion
│   ├── embeddings.py
│   ├── knowledge_base.py
│   ├── ingest.py
│   └── documents/         # sample travel guides (auto-ingested)
├── database/          # SQLite connection + schema
│   ├── db.py
│   └── schema.sql
├── api/               # FastAPI app
│   └── main.py
├── frontend/          # Streamlit UI
│   └── streamlit_app.py
├── core/              # config + logging
│   ├── config.py
│   └── logging_config.py
├── tests/             # pytest suite
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## 🚀 Quick start (local)

### 1. Install

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env          # Windows: copy .env.example .env
```

Edit `.env` and set `LLM_PROVIDER` plus the matching API key. **No key?** Leave
them blank — the app runs in deterministic *mock mode*.

### 3. (Optional) Ingest the knowledge base

The bundled travel guides under `rag/documents/` are auto-ingested on first run,
but you can (re)ingest explicitly — or point it at your own folder:

```bash
python -m rag.ingest
python -m rag.ingest path/to/your/docs
```

### 4. Run the backend

```bash
uvicorn api.main:app --reload
```

API docs: http://localhost:8000/docs

### 5. Run the frontend

```bash
streamlit run frontend/streamlit_app.py
```

UI: http://localhost:8501

---

## 🐳 Run with Docker

```bash
docker compose up --build
```

- API:      http://localhost:8000
- Frontend: http://localhost:8501

The `data/` folder is mounted as a volume so SQLite and the Chroma store persist
across restarts. Put your keys in `.env` (read by both services).

---

## 🔌 API reference

| Method | Path                | Description                                   |
|--------|---------------------|-----------------------------------------------|
| POST   | `/plan-trip`        | Run the multi-agent workflow, return a plan   |
| GET    | `/trip-history`     | Recent trips (optional `?user_id=`)           |
| GET    | `/user-preferences` | Aggregated learned preferences (`?user_id=`)  |
| GET    | `/trip/{trip_id}`   | Fetch a single saved plan                     |
| GET    | `/health`           | Liveness + LLM/RAG status                     |

### Example request

```bash
curl -X POST http://localhost:8000/plan-trip \
  -H "Content-Type: application/json" \
  -d '{
        "destination": "Goa",
        "days": 4,
        "budget": 25000,
        "travelers": 2,
        "travel_style": "budget",
        "interests": ["beaches", "food", "nightlife"],
        "departure_city": "Delhi",
        "user_id": "demo-user"
      }'
```

---

## 🧠 Memory

Every generated plan is stored in SQLite. After each trip the user's aggregate
profile is recomputed (favourite destinations, preferred styles, common
interests, average budget) and injected back into agent prompts so subsequent
plans become progressively more personalised.

## 📚 RAG

`rag/knowledge_base.py` chunks documents (sentence-aware, overlapping), embeds
them and stores them in a persistent ChromaDB collection. Agents retrieve
relevant context per request to ground their output. Embeddings use OpenAI when
a key is present and fall back to a deterministic local embedding otherwise.

---

## ⚙️ Configuration

All configuration is via environment variables (see `.env.example`). Key ones:

| Variable               | Default                        | Purpose                          |
|------------------------|--------------------------------|----------------------------------|
| `LLM_PROVIDER`         | `openai`                       | `openai`, `anthropic` or `google`|
| `OPENAI_MODEL`         | `gpt-4o`                       | OpenAI chat model                |
| `ANTHROPIC_MODEL`      | `claude-3-5-sonnet-20240620`   | Anthropic chat model             |
| `GOOGLE_MODEL`         | `gemini-1.5-flash`             | Gemini chat model                |
| `OPENAI_API_KEY`       | _(empty)_                      | OpenAI key                       |
| `ANTHROPIC_API_KEY`    | _(empty)_                      | Anthropic key                    |
| `GOOGLE_API_KEY`       | _(empty)_                      | Gemini (Google AI Studio) key    |
| `ALLOW_MOCK_FALLBACK`  | `true`                         | Use mocks when LLM unavailable   |
| `SQLITE_DB_PATH`       | `./data/travel_planner.db`     | SQLite location                  |
| `CHROMA_PERSIST_DIR`   | `./data/chroma`                | Vector store location            |
| `LOG_LEVEL`            | `INFO`                         | Logging verbosity                |

---

## ✅ Testing

```bash
pytest
```

The suite forces offline mock mode with isolated temporary storage, so it runs
without any API keys or network access. It covers schema validation, RAG
chunking/embeddings, each agent, memory aggregation, the full LangGraph
workflow, and the FastAPI endpoints.

---

## 🛠️ Tech stack

Python · FastAPI · LangGraph · LangChain · OpenAI / Anthropic · ChromaDB ·
SQLite · Pydantic · Streamlit · Docker.

---

## 📝 Notes & extensibility

- **Swap the LLM** by changing `LLM_PROVIDER` and the relevant key — no code
  changes needed.
- **Add knowledge** by dropping `.md`/`.txt` files into `rag/documents/` and
  re-running `python -m rag.ingest`.
- **Add an agent** by subclassing `agents.base.BaseAgent` and registering it as
  a node in `graph/workflow.py`.
```
