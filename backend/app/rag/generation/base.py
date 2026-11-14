from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class LLMGenerationResult(BaseModel):
    """Result payload returned by an LLM provider."""
    text: str = Field(..., description="Generated text response")
    provider: str = Field(..., description="Provider name (e.g. 'groq', 'mock')")
    model: str = Field(..., description="Model identifier")
    tokens_used: Optional[int] = Field(default=None, description="Actual token count consumed")
    finish_reason: Optional[str] = Field(default="stop", description="Generation completion reason")
    latency_ms: Optional[float] = Field(default=None, description="Generation call latency in milliseconds")


class BaseLLMProvider(ABC):
    """Abstract interface for LLM answer generation providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider service."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the active generation model."""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_instruction: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> LLMGenerationResult:
        """
        Executes grounded text generation given a prompt, strict system instruction,
        and optional previous conversation history.
        """
        pass
