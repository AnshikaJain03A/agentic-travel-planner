"""
LLM provider abstraction.

Wraps LangChain chat models so the rest of the codebase is agnostic to whether
we're talking to OpenAI (GPT-4o) or Anthropic (Claude Sonnet). The provider is
chosen via configuration (``LLM_PROVIDER``).

Key features:
* Lazy, fault-tolerant model construction.
* ``is_available`` so agents can gracefully fall back to deterministic mocks
  when no API key is configured (great for offline dev and CI).
* ``structured`` helper that returns a validated Pydantic object using the
  provider's native structured-output / tool-calling support, with retries.
"""

from __future__ import annotations

from typing import Optional, Type, TypeVar

from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core.config import settings
from core.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMUnavailableError(RuntimeError):
    """Raised when no usable LLM backend is configured."""


class LLMService:
    """Provider-agnostic wrapper around a LangChain chat model."""

    def __init__(self) -> None:
        self.provider = settings.llm_provider.lower()
        self.model_name = settings.active_model
        self._model = self._build_model()

    # ------------------------------------------------------------------
    def _build_model(self):
        """Instantiate the configured chat model, or ``None`` if unavailable."""
        try:
            if self.provider == "anthropic":
                if not settings.has_anthropic_key:
                    logger.warning("Anthropic selected but ANTHROPIC_API_KEY missing")
                    return None
                from langchain_anthropic import ChatAnthropic

                return ChatAnthropic(
                    model=settings.anthropic_model,
                    temperature=settings.llm_temperature,
                    api_key=settings.anthropic_api_key,
                    max_retries=0,  # we handle retries ourselves
                )

            if self.provider in ("google", "gemini"):
                if not settings.has_google_key:
                    logger.warning("Google/Gemini selected but GOOGLE_API_KEY missing")
                    return None
                from langchain_google_genai import ChatGoogleGenerativeAI

                return ChatGoogleGenerativeAI(
                    model=settings.google_model,
                    temperature=settings.llm_temperature,
                    google_api_key=settings.google_api_key,
                    max_retries=0,  # we handle retries ourselves
                )

            # Default: OpenAI
            if not settings.has_openai_key:
                logger.warning("OpenAI selected but OPENAI_API_KEY missing")
                return None
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=settings.openai_model,
                temperature=settings.llm_temperature,
                api_key=settings.openai_api_key,
                max_retries=0,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to construct LLM (%s): %s", self.provider, exc)
            return None

    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
    )
    def _invoke_structured(self, output_model: Type[T], messages) -> T:
        structured = self._model.with_structured_output(output_model)
        return structured.invoke(messages)

    # ------------------------------------------------------------------
    def structured(
        self,
        output_model: Type[T],
        system_prompt: str,
        human_prompt: str,
    ) -> T:
        """Return a validated ``output_model`` instance from the LLM.

        Raises :class:`LLMUnavailableError` if no backend is configured.
        """
        if not self.is_available():
            raise LLMUnavailableError(
                f"No LLM backend available for provider '{self.provider}'."
            )

        messages = [
            ("system", system_prompt),
            ("human", human_prompt),
        ]
        logger.debug("LLM structured call -> %s", output_model.__name__)
        return self._invoke_structured(output_model, messages)


_llm_singleton: Optional[LLMService] = None


def get_llm() -> LLMService:
    """Return the shared :class:`LLMService` singleton."""
    global _llm_singleton
    if _llm_singleton is None:
        _llm_singleton = LLMService()
    return _llm_singleton
