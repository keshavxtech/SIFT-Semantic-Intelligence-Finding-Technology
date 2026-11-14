import time
from typing import Dict, List, Optional
from app.core.errors import LLMProviderError, LLMTimeoutError
from app.rag.generation.base import BaseLLMProvider, LLMGenerationResult


class MockLLMProvider(BaseLLMProvider):
    """
    Mock LLM provider for reproducible, zero-cost, offline automated testing.
    Generates deterministic grounded answers without hitting any external API.
    """

    def __init__(
        self,
        model_name: str = "mock-rag-llama-3",
        simulated_answer: Optional[str] = None,
        simulated_timeout: bool = False,
        simulated_error: Optional[str] = None,
        simulated_tokens: int = 120,
    ):
        self._model = model_name
        self.simulated_answer = simulated_answer
        self.simulated_timeout = simulated_timeout
        self.simulated_error = simulated_error
        self.simulated_tokens = simulated_tokens

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return self._model

    async def generate(
        self,
        prompt: str,
        system_instruction: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> LLMGenerationResult:
        if self.simulated_timeout:
            raise LLMTimeoutError(timeout_seconds=5.0)

        if self.simulated_error:
            raise LLMProviderError(message=self.simulated_error, status_code=502)

        start_time = time.perf_counter()

        if self.simulated_answer:
            answer = self.simulated_answer
        else:
            # Deterministically check if passages were included in the prompt
            if "NO RELEVANT DOCUMENT CONTEXT AVAILABLE" in prompt or "No relevant passages found" in prompt:
                answer = "Based on the provided documents, there is insufficient evidence to answer your question."
            elif "Passage [1]" in prompt:
                # Extract first excerpt for grounded response
                lines = [line.strip() for line in prompt.split("\n") if line.strip().startswith("Content:")]
                excerpt = lines[0].replace("Content:", "").strip() if lines else "the documented operational parameters"
                answer = (
                    f"Based on the provided documentation, {excerpt[:180]} [1]. "
                    "These verified parameters govern system operations in full accordance with the cited specifications."
                )
            else:
                answer = "Based on the provided documents, there is insufficient evidence to answer your question."

        latency_ms = (time.perf_counter() - start_time) * 1000

        return LLMGenerationResult(
            text=answer,
            provider=self.provider_name,
            model=self._model,
            tokens_used=self.simulated_tokens,
            finish_reason="stop",
            latency_ms=round(latency_ms, 2),
        )
