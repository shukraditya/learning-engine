"""LLM provider abstraction for ATT.

Supports OpenAI, Anthropic, and Ollama via unified interface.
Follows BYOK (Bring Your Own Key) principle - no keys stored server-side.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Any

from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, field_validator
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class LLMConfig(BaseModel):
    """Configuration for LLM provider.

    Supports env var fallbacks for BYOK workflow.
    """

    provider: LLMProvider = Field(
        default=LLMProvider.OPENAI,
        description="LLM provider to use",
    )
    model: str = Field(
        ...,
        description="Model name (e.g., 'gpt-4o', 'claude-3-5-sonnet-20241022')",
    )
    temperature: float = Field(
        default=0.1,
        description="Sampling temperature",
        ge=0.0,
        le=2.0,
    )
    max_tokens: int | None = Field(
        default=None,
        description="Max tokens in response",
    )
    api_key: str | None = Field(
        default=None,
        description="API key (falls back to env var)",
    )
    base_url: str | None = Field(
        default=None,
        description="Base URL for API (e.g., Ollama endpoint)",
    )

    @field_validator("api_key", mode="before")
    @classmethod
    def load_api_key_from_env(cls, v: str | None, info) -> str | None:
        """Load API key from environment if not provided."""
        if v is not None:
            return v

        provider = info.data.get("provider", LLMProvider.OPENAI)

        env_vars = {
            LLMProvider.OPENAI: "OPENAI_API_KEY",
            LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
            LLMProvider.OLLAMA: None,  # Ollama doesn't require API key
        }

        env_var = env_vars.get(provider)
        if env_var:
            return os.environ.get(env_var)
        return None

    @field_validator("base_url", mode="before")
    @classmethod
    def load_base_url_from_env(cls, v: str | None, info) -> str | None:
        """Load base URL from environment for Ollama compatibility."""
        if v is not None:
            return v

        # Support OPENAI_API_BASE for Ollama OpenAI-compatible endpoints
        return os.environ.get("OPENAI_API_BASE")


class LLMResponse(BaseModel):
    """Standardized LLM response."""

    content: str = Field(..., description="Response text content")
    model: str = Field(..., description="Model used")
    usage: dict[str, int] | None = Field(
        default=None,
        description="Token usage (prompt, completion, total)",
    )
    finish_reason: str | None = Field(default=None)


class LLMClient:
    """Unified LLM client with retry logic.

    Wraps provider-specific clients (OpenAI, Anthropic)
    with common retry and error handling.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client: AsyncOpenAI | Any | None = None

        logger.debug(f"Initialized LLMClient: {config.provider.value}/{config.model}")

    async def _get_client(self) -> AsyncOpenAI | Any:
        """Lazy-load the appropriate client."""
        if self._client is not None:
            return self._client

        if self.config.provider == LLMProvider.OPENAI:
            if not self.config.api_key:
                raise ValueError(
                    "OpenAI API key required. Set OPENAI_API_KEY env var "
                    "or pass api_key to config."
                )
            self._client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )

        elif self.config.provider == LLMProvider.ANTHROPIC:
            try:
                from anthropic import AsyncAnthropic
            except ImportError:
                raise ImportError(
                    "anthropic package required. Install with: uv add anthropic"
                )
            if not self.config.api_key:
                raise ValueError(
                    "Anthropic API key required. Set ANTHROPIC_API_KEY env var "
                    "or pass api_key to config."
                )
            self._client = AsyncAnthropic(api_key=self.config.api_key)

        elif self.config.provider == LLMProvider.OLLAMA:
            # Ollama uses OpenAI-compatible endpoint
            base_url = self.config.base_url or "http://localhost:11434/v1"
            self._client = AsyncOpenAI(
                api_key="ollama",  # Ollama doesn't check this
                base_url=base_url,
            )
            logger.debug(f"Using Ollama endpoint: {base_url}")

        return self._client

    @retry(
        retry=retry_if_exception_type((TimeoutError, ConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def complete(
        self,
        system_prompt: str | None,
        user_prompt: str,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Generate completion with retry logic.

        Args:
            system_prompt: System instructions (optional)
            user_prompt: User message content
            response_format: JSON schema for structured output (OpenAI only)

        Returns:
            Standardized LLMResponse
        """
        client = await self._get_client()

        logger.debug(
            f"LLM request: {self.config.provider.value}/{self.config.model}, "
            f"prompt_len={len(user_prompt)}"
        )

        if self.config.provider == LLMProvider.ANTHROPIC:
            return await self._call_anthropic(client, system_prompt, user_prompt)
        else:
            # OpenAI and Ollama use same interface
            return await self._call_openai(
                client, system_prompt, user_prompt, response_format
            )

    async def _call_openai(
        self,
        client: AsyncOpenAI,
        system_prompt: str | None,
        user_prompt: str,
        response_format: dict[str, Any] | None,
    ) -> LLMResponse:
        """Call OpenAI-compatible API."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        if self.config.max_tokens:
            kwargs["max_tokens"] = self.config.max_tokens
        if response_format:
            kwargs["response_format"] = response_format

        try:
            response = await client.chat.completions.create(**kwargs)

            return LLMResponse(
                content=response.choices[0].message.content or "",
                model=response.model,
                usage={
                    "prompt": response.usage.prompt_tokens if response.usage else 0,
                    "completion": response.usage.completion_tokens
                    if response.usage
                    else 0,
                    "total": response.usage.total_tokens if response.usage else 0,
                }
                if response.usage
                else None,
                finish_reason=response.choices[0].finish_reason,
            )
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    async def _call_anthropic(
        self,
        client: Any,
        system_prompt: str | None,
        user_prompt: str,
    ) -> LLMResponse:
        """Call Anthropic API."""
        try:
            response = await client.messages.create(
                model=self.config.model,
                system=system_prompt or "",
                messages=[{"role": "user", "content": user_prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens or 4096,
            )

            return LLMResponse(
                content=response.content[0].text if response.content else "",
                model=response.model,
                usage={
                    "prompt": response.usage.input_tokens,
                    "completion": response.usage.output_tokens,
                    "total": response.usage.input_tokens + response.usage.output_tokens,
                }
                if response.usage
                else None,
                finish_reason=response.stop_reason,
            )
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise


class LLMFactory:
    """Factory for creating LLM clients."""

    @staticmethod
    def create(config: LLMConfig | None = None, **kwargs) -> LLMClient:
        """Create LLM client from config or kwargs.

        Args:
            config: LLMConfig instance, or None to create from kwargs
            **kwargs: Fields for LLMConfig if config not provided

        Returns:
            Configured LLMClient
        """
        if config is None:
            config = LLMConfig(**kwargs)
        return LLMClient(config)

    @staticmethod
    def from_env() -> LLMClient:
        """Create LLM client from environment variables.

        Detects provider from available env vars:
        - ANTHROPIC_API_KEY -> Anthropic
        - OPENAI_API_KEY -> OpenAI
        - OPENAI_API_BASE (with no key) -> Ollama
        """
        if os.environ.get("ANTHROPIC_API_KEY"):
            return LLMFactory.create(
                provider=LLMProvider.ANTHROPIC,
                model=os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
            )
        elif os.environ.get("OPENAI_API_KEY"):
            return LLMFactory.create(
                provider=LLMProvider.OPENAI,
                model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
            )
        else:
            # Default to Ollama
            return LLMFactory.create(
                provider=LLMProvider.OLLAMA,
                model=os.environ.get("OLLAMA_MODEL", "llama3.2"),
            )
