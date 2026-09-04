"""
Base class for all specialist agents.

Each agent is a LangGraph node. The base class implements the common lifecycle:

    1. Optionally retrieve RAG grounding context for the request.
    2. Build a request-specific human prompt.
    3. Call the LLM for a validated structured output.
    4. On unavailability or error, fall back to a deterministic mock so the
       whole workflow keeps producing a usable plan (controlled by
       ``ALLOW_MOCK_FALLBACK``).

Subclasses implement: ``output_model``, ``system_prompt``, ``build_prompt`` and
``mock_output``; and optionally override ``retrieval_query`` and ``state_key``.
"""

from __future__ import annotations

import abc
from typing import Optional, Type

from pydantic import BaseModel

from core.config import settings
from core.logging_config import get_logger
from graph.state import PlannerState
from rag.knowledge_base import get_knowledge_base
from services.llm import LLMService, get_llm


class BaseAgent(abc.ABC):
    """Abstract specialist agent / LangGraph node."""

    #: Human-friendly agent name (used in logs and warnings).
    name: str = "agent"
    #: Key under which this agent writes its output into the state.
    state_key: str = "output"

    def __init__(self, llm: Optional[LLMService] = None) -> None:
        self.llm = llm or get_llm()
        self.logger = get_logger(f"agent.{self.name}")

    # ---- Required overrides -------------------------------------------------
    @property
    @abc.abstractmethod
    def output_model(self) -> Type[BaseModel]:
        """The Pydantic model this agent produces."""

    @property
    @abc.abstractmethod
    def system_prompt(self) -> str:
        """The agent's system prompt / persona."""

    @abc.abstractmethod
    def build_prompt(self, state: PlannerState, rag_context: str) -> str:
        """Build the request-specific human prompt."""

    @abc.abstractmethod
    def mock_output(self, state: PlannerState) -> BaseModel:
        """Deterministic fallback output when the LLM is unavailable."""

    # ---- Optional overrides -------------------------------------------------
    def retrieval_query(self, state: PlannerState) -> Optional[str]:
        """Return a RAG query string, or ``None`` to skip retrieval."""
        return None

    # ---- Lifecycle ----------------------------------------------------------
    def _retrieve(self, state: PlannerState) -> str:
        query = self.retrieval_query(state)
        if not query:
            return ""
        try:
            context = get_knowledge_base().retrieve_context(query, k=4)
            if context:
                self.logger.debug("Retrieved RAG context for query: %s", query)
            return context
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("RAG retrieval failed: %s", exc)
            return ""

    def run(self, state: PlannerState) -> dict:
        """LangGraph node entry point. Returns a partial state update."""
        self.logger.info("Running %s agent", self.name)
        warnings = list(state.get("warnings", []))
        rag_context = self._retrieve(state)

        try:
            if self.llm.is_available():
                human_prompt = self.build_prompt(state, rag_context)
                output = self.llm.structured(
                    self.output_model, self.system_prompt, human_prompt
                )
            else:
                self.logger.info("LLM unavailable; using mock output for %s", self.name)
                output = self.mock_output(state)
                warnings.append(f"{self.name}: generated with offline mock data (no LLM key).")
        except Exception as exc:  # noqa: BLE001
            self.logger.exception("%s agent failed", self.name)
            if not settings.allow_mock_fallback:
                raise
            output = self.mock_output(state)
            warnings.append(f"{self.name}: LLM error, used fallback data ({exc.__class__.__name__}).")

        return {self.state_key: output, "warnings": warnings}
