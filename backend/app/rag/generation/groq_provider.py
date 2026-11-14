import time
from typing import Any, Dict, List, Optional
import httpx
from app.core.config import settings
from app.core.errors import (
    LLMProviderError,
    LLMProviderNotConfiguredError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from app.core.logging import logger
from app.rag.generation.base import BaseLLMProvider, LLMGenerationResult


class GroqProvider(BaseLLMProvider):
    """
    Production LLM provider connecting to Groq's high-speed inference API.
    Uses configurable environment keys and parameters without hardcoded credentials.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        api_url: Optional[str] = None,
        timeout_seconds: float = 30.0,
    ):
        self._api_key = api_key or settings.GROQ_API_KEY
        self._model = model or settings.GROQ_MODEL
        self._api_url = api_url or settings.GROQ_API_URL
        self._timeout = timeout_seconds

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(
        self,
        prompt: str,
        system_instruction: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> LLMGenerationResult:
        if not self._api_key or not self._api_key.strip():
            raise LLMProviderNotConfiguredError()

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_instruction}
        ]

        if conversation_history:
            for turn in conversation_history:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self._api_key.strip()}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": settings.LLM_TEMPERATURE,
            "max_tokens": settings.LLM_MAX_TOKENS,
        }

        start_time = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(self._api_url, json=payload, headers=headers)

            latency_ms = (time.perf_counter() - start_time) * 1000

            if response.status_code in (401, 403):
                logger.error("Groq API authentication rejected invalid API key.")
                raise LLMProviderError(
                    "Groq API authentication failed. Verify GROQ_API_KEY in your environment.",
                    status_code=401,
                )
            elif response.status_code == 429:
                logger.warning("Groq API rate limit reached.")
                raise LLMRateLimitError()
            elif response.status_code >= 400:
                logger.error(f"Groq API error ({response.status_code}): {response.text}")
                raise LLMProviderError(
                    f"Groq provider returned error status {response.status_code}.",
                    status_code=502,
                )

            data = response.json()
            choice = data["choices"][0]
            answer_text = choice["message"]["content"]
            finish_reason = choice.get("finish_reason", "stop")
            tokens_used = data.get("usage", {}).get("total_tokens")

            logger.info(
                f"Groq generation successful ({self._model}, {latency_ms:.1f}ms, tokens={tokens_used})"
            )

            return LLMGenerationResult(
                text=answer_text,
                provider=self.provider_name,
                model=self._model,
                tokens_used=tokens_used,
                finish_reason=finish_reason,
                latency_ms=round(latency_ms, 2),
            )

        except httpx.TimeoutException:
            logger.error(f"Groq request timed out after {self._timeout}s")
            raise LLMTimeoutError(timeout_seconds=self._timeout)
        except httpx.RequestError as exc:
            logger.error(f"Groq network connection error: {exc}")
            raise LLMProviderError(
                f"Could not connect to Groq API: {str(exc)}",
                status_code=502,
            )
