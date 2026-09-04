# ============================================================
#  Agentic AI Travel Planner - container image
#  Runs the FastAPI backend by default. The same image can run
#  the Streamlit frontend by overriding the command (see compose).
# ============================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps occasionally needed by chromadb / native wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the application.
COPY . .

# Persisted data (SQLite + Chroma) lives here; mount a volume in production.
RUN mkdir -p /app/data

EXPOSE 8000 8501

# Default: API server. Override `command` to launch Streamlit.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
