"""
LLM Client for Benchmark Infrastructure.

Provides unified interface for LLM API calls across different providers.
"""

import logging
from typing import Any, Optional, Tuple

from openai import BadRequestError, OpenAI

from src.config import (
    ModelConfig,
    AVAILABLE_MODELS,
    get_api_key,
    get_model_config,
    list_models as _list_models,
)

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified LLM client for benchmark infrastructure."""

    # Re-export for backward compatibility
    AVAILABLE_MODELS = AVAILABLE_MODELS

    def __init__(
        self,
        model_key: str = "deepseek-v3.2",
        config: Optional[ModelConfig] = None,
    ):
        """Initialize LLM client.

        Args:
            model_key: Key from AVAILABLE_MODELS
            config: Optional custom ModelConfig (overrides model_key)
        """
        if config:
            self.config = config
        else:
            self.config = get_model_config(model_key)

        api_key = get_api_key(self.config.api_key_env)

        self.client = OpenAI(
            api_key=api_key,
            base_url=self.config.base_url,
        )
        self.model_key = model_key
        logger.info(f"Initialized LLM client: {self.config.model_id}")

    @property
    def is_openrouter(self) -> bool:
        """Whether this client should use OpenRouter-specific request extensions."""
        base_url = (self.config.base_url or "").lower()
        return (
            self.model_key.startswith("openrouter-")
            or self.model_key.startswith("openai-compatible-")
            or "openrouter.ai" in base_url
        )

    def _extract_text(self, value: Any) -> str:
        """Best-effort text extraction from OpenAI/OpenRouter response fields."""
        if value is None:
            return ""

        if isinstance(value, str):
            return value.strip()

        if isinstance(value, list):
            parts = [self._extract_text(item) for item in value]
            return "\n".join(part for part in parts if part).strip()

        if isinstance(value, dict):
            if isinstance(value.get("text"), str):
                return value["text"].strip()

            if isinstance(value.get("content"), str):
                return value["content"].strip()

            if isinstance(value.get("summary"), str):
                return value["summary"].strip()

            if isinstance(value.get("summary"), list):
                return self._extract_text(value["summary"])

            if isinstance(value.get("reasoning"), str):
                return value["reasoning"].strip()

            return ""

        text = getattr(value, "text", None)
        if isinstance(text, str):
            return text.strip()

        summary = getattr(value, "summary", None)
        if summary is not None:
            return self._extract_text(summary)

        content = getattr(value, "content", None)
        if content is not None:
            return self._extract_text(content)

        return ""

    def _looks_like_code(self, text: str) -> bool:
        """Heuristic guard before treating non-content fields as executable code."""
        code_markers = (
            "```python",
            "```",
            "from pymatgen",
            "import pymatgen",
            "from pymatgen.core",
            "import numpy",
            "structure =",
            "slab =",
        )
        stripped = text.lstrip()
        return any(marker in stripped for marker in code_markers)

    def _extract_message_content(self, response: Any) -> str:
        """Normalize content across OpenRouter/OpenAI response variants."""
        choice = response.choices[0]
        message = choice.message

        content = self._extract_text(getattr(message, "content", None))
        if content:
            return content

        reasoning = self._extract_text(getattr(message, "reasoning", None))
        if reasoning and self._looks_like_code(reasoning):
            logger.warning(
                "Model %s returned code in reasoning field; using it as fallback.",
                self.config.model_id,
            )
            return reasoning

        reasoning_details = self._extract_text(getattr(message, "reasoning_details", None))
        if reasoning_details and self._looks_like_code(reasoning_details):
            logger.warning(
                "Model %s returned code in reasoning_details; using it as fallback.",
                self.config.model_id,
            )
            return reasoning_details

        finish_reason = getattr(choice, "finish_reason", None)
        raise ValueError(
            "Model returned empty message content "
            f"(model={self.config.model_id}, finish_reason={finish_reason}, "
            f"reasoning_present={bool(reasoning or reasoning_details)})"
        )

    def _openrouter_reasoning_payload(self, mandatory: bool = False) -> dict[str, Any]:
        """Choose a reasoning policy that keeps OpenRouter code generation stable."""
        if mandatory or self.config.model_id.startswith("minimax/"):
            return {
                "reasoning": {
                    "effort": "low",
                    "exclude": True,
                }
            }

        return {
            "reasoning": {
                "effort": "none",
                "exclude": True,
            }
        }

    def _create_completion(self, request_kwargs: dict[str, Any]) -> Any:
        """Create a chat completion with OpenRouter-specific retries."""
        try:
            return self.client.chat.completions.create(**request_kwargs)
        except BadRequestError as e:
            if self.is_openrouter and "Reasoning is mandatory" in str(e):
                logger.warning(
                    "Model %s requires reasoning; retrying with low reasoning effort.",
                    self.config.model_id,
                )
                request_kwargs["extra_body"] = self._openrouter_reasoning_payload(
                    mandatory=True
                )
                return self.client.chat.completions.create(**request_kwargs)
            raise

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, int]:
        """Generate response from LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max_tokens

        Returns:
            Tuple of (response_content, total_tokens)
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            request_kwargs = {
                "model": self.config.model_id,
                "messages": messages,
                "temperature": temperature or self.config.temperature,
                "max_tokens": max_tokens or self.config.max_tokens,
            }

            # OpenRouter reasoning models can return content=None when they spend
            # the budget on thinking tokens. For code-generation benchmarks we
            # want direct code output, so disable reasoning by default.
            if self.is_openrouter:
                request_kwargs["extra_body"] = self._openrouter_reasoning_payload()

            response = self._create_completion(request_kwargs)

            try:
                content = self._extract_message_content(response)
            except ValueError as e:
                is_minimax_length_issue = (
                    self.config.model_id.startswith("minimax/")
                    and "finish_reason=length" in str(e)
                    and request_kwargs["max_tokens"] < 12000
                )
                if not is_minimax_length_issue:
                    raise

                logger.warning(
                    "Model %s exhausted token budget without content; retrying with higher max_tokens.",
                    self.config.model_id,
                )
                request_kwargs["max_tokens"] = 12000
                request_kwargs["extra_body"] = self._openrouter_reasoning_payload(
                    mandatory=True
                )
                response = self._create_completion(request_kwargs)
                content = self._extract_message_content(response)

            tokens = response.usage.total_tokens if response.usage else 0

            return content, tokens

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

    @classmethod
    def list_models(cls) -> list:
        """List available model keys."""
        return _list_models()
