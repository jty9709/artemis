# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single OpenAI-compatible model router for all ARTEMIS execution paths."""

from enum import StrEnum
import hashlib
import os
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from artemis.config.settings import settings



class ModelProvider(StrEnum):
    GOOGLE = "google"
    GEMINI = "google"
    VERTEX_AI = "vertexai"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    XAI = "xai"
    OLLAMA = "ollama"
    VLLM = "vllm"
    CUSTOM = "custom"

    @classmethod
    def from_string(cls, val: Any) -> "ModelProvider":
        """Converts a string or enum into a canonical ModelProvider.

        ``None``/empty means "use the default" (OpenAI); an unrecognized
        non-empty value raises instead of silently routing to Gemini with
        the wrong credentials.
        """
        if isinstance(val, cls):
            return val
        if val is None:
            return cls.OPENAI
        s = str(val).lower().replace("_", "").replace("-", "").strip()
        if not s:
            return cls.OPENAI
        mapping = {
            "google": cls.GOOGLE,
            "gemini": cls.GOOGLE,
            "vertexai": cls.VERTEX_AI,
            "vertex": cls.VERTEX_AI,
            "openai": cls.OPENAI,
            "anthropic": cls.ANTHROPIC,
            "claude": cls.ANTHROPIC,
            "openrouter": cls.OPENROUTER,
            "xai": cls.XAI,
            "grok": cls.XAI,
            "ollama": cls.OLLAMA,
            "vllm": cls.VLLM,
            "custom": cls.CUSTOM,
        }
        provider = mapping.get(s)
        if provider is None:
            raise ValueError(
                f"Unknown LLM provider {val!r}. Valid providers: {sorted(set(mapping))}"
            )
        return provider


class ModelEndpoint(BaseModel):
    """Configuration definition for an LLM/VLM model endpoint."""

    provider: ModelProvider = Field(default=ModelProvider.OPENAI, description="Model provider")
    model_name: str = Field(default="", description="Model name identifier")
    api_key: str | None = Field(default=None, repr=False, description="API Key or secret")
    api_base: str | None = Field(default=None, description="Custom API endpoint base URL")
    temperature: float = Field(default=0.0, description="Sampling temperature")
    max_tokens: int | None = Field(default=None, description="Maximum completion tokens")
    timeout_seconds: float = Field(default=60.0, description="Request timeout in seconds")
    is_multimodal: bool = Field(
        default=True, description="Whether endpoint accepts image/video inputs"
    )
    reasoning_effort: str | None = Field(
        default=None, description="Reasoning effort: 'low', 'medium', 'high'"
    )
    thinking_budget: int | None = Field(default=None, description="Thinking budget token count")
    thinking_level: str | None = Field(
        default=None, description="Thinking level: 'minimal', 'low', 'medium', 'high'"
    )
    include_thoughts: bool | None = Field(
        default=None, description="Whether to include thought traces in response"
    )
    enable_grounding: bool = Field(
        default=False,
        description="Whether to enable Google Search grounding for Gemini endpoints",
    )

    def cache_key(self) -> tuple:
        """Key the client cache by endpoint settings, hashing the API key."""
        api_key_digest = (
            hashlib.sha256(self.api_key.encode("utf-8")).hexdigest()[:16] if self.api_key else None
        )
        return (
            self.provider.value,
            self.model_name,
            self.temperature,
            self.max_tokens,
            self.timeout_seconds,
            self.thinking_budget,
            self.thinking_level,
            self.include_thoughts,
            self.reasoning_effort,
            self.enable_grounding,
            self.api_base,
            api_key_digest,
        )


class ModelFactory:
    """Every model request uses the single OpenAI-compatible endpoint."""

    _cache: dict[tuple, BaseChatModel] = {}

    @staticmethod
    def resolve_endpoint(endpoint: ModelEndpoint) -> ModelEndpoint:
        """Override legacy provider/model choices, including auxiliary fallbacks."""
        if not settings.OPENAI_BASE_URL or not settings.OPENAI_BASE_URL.strip():
            raise ValueError("Configure OPENAI_BASE_URL in .env")
        if not settings.OPENAI_MODEL.strip():
            raise ValueError("Configure OPENAI_MODEL in .env")
        if not settings.OPENAI_API_KEY:
            raise ValueError("Configure OPENAI_API_KEY in .env")
        return endpoint.model_copy(
            update={
                "provider": ModelProvider.OPENAI,
                "model_name": settings.OPENAI_MODEL.strip(),
                "api_base": settings.OPENAI_BASE_URL.strip().rstrip("/"),
                "api_key": settings.OPENAI_API_KEY.get_secret_value(),
                "thinking_budget": None,
                "thinking_level": None,
                "include_thoughts": None,
                "enable_grounding": False,
            }
        )

    @classmethod
    def get_model(cls, endpoint: ModelEndpoint) -> BaseChatModel:
        endpoint = cls.resolve_endpoint(endpoint)
        key = endpoint.cache_key()
        if key not in cls._cache:
            cls._cache[key] = cls.create_model(endpoint)
        return cls._cache[key]

    @classmethod
    def create_model(cls, endpoint: ModelEndpoint) -> BaseChatModel:
        if os.environ.get("ARTEMIS_FAKE_LLM") == "1":
            from artemis.llm.fake_model import FakeChatModel

            delay = float(os.environ.get("ARTEMIS_FAKE_LLM_DELAY_S", "0") or 0)
            return FakeChatModel(delay_s=delay)

        from langchain_openai import ChatOpenAI

        endpoint = cls.resolve_endpoint(endpoint)
        kwargs = {
            "model": endpoint.model_name,
            "temperature": endpoint.temperature,
            "max_tokens": endpoint.max_tokens,
            "api_key": endpoint.api_key,
            "base_url": endpoint.api_base,
            "timeout": endpoint.timeout_seconds,
            # OneAPI exposes Chat Completions, regardless of the model ID.
            "use_responses_api": False,
        }
        if endpoint.reasoning_effort:
            kwargs["reasoning_effort"] = endpoint.reasoning_effort
        return ChatOpenAI(**{k: v for k, v in kwargs.items() if v is not None})
