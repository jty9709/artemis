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

"""Unit tests for ModelFactory instantiation and grounding tool wiring."""

from artemis.llm.router import ModelEndpoint, ModelFactory, ModelProvider
from artemis.config import settings


def test_model_factory_anthropic_instantiation():
    """Verify Anthropic Claude model instantiation with thinking budget."""
    anth_ep = ModelEndpoint(
        provider=ModelProvider.ANTHROPIC,
        model_name="claude-3-7-sonnet-20250219",
        api_key="sk-ant-test-key",
        reasoning_effort="high",
    )
    model = ModelFactory.create_model(anth_ep)
    assert model.model_name == settings.OPENAI_MODEL
    assert model.openai_api_base == settings.OPENAI_BASE_URL


def test_model_factory_openai_instantiation():
    """Verify OpenAI model instantiation with reasoning effort."""
    oai_ep = ModelEndpoint(
        provider=ModelProvider.OPENAI,
        model_name="o3-mini",
        api_key="sk-test-key",
        reasoning_effort="medium",
    )
    model = ModelFactory.create_model(oai_ep)
    assert model.model_name == settings.OPENAI_MODEL
    assert getattr(model, "reasoning_effort", None) == "medium"


def test_robust_chat_model_wrapper_grounding_google():
    """Verify RobustChatModelWrapper auto-injects google_search and server-side tool config for Gemini."""
    from artemis.services.llm import RobustChatModelWrapper
    from langchain_core.tools import tool

    @tool
    def dummy_tool(query: str) -> str:
        """A test tool."""
        return query

    class MockGeminiModel:
        def __init__(self):
            self.bound_tools = []
            self.bound_kwargs = {}

        def bind_tools(self, tools, *args, **kwargs):
            self.bound_tools = tools
            self.bound_kwargs = kwargs
            return self

    # Case 1: enable_grounding=True automatically attaches google_search and tool_config
    ep_google = ModelEndpoint(
        provider=ModelProvider.GOOGLE,
        model_name="gemini-2.5-flash",
        enable_grounding=True,
    )
    wrapper = RobustChatModelWrapper(MockGeminiModel(), endpoint=ep_google)
    bound = wrapper.bind_tools([dummy_tool])

    assert any(isinstance(t, dict) and "google_search" in t for t in bound.base_model.bound_tools)
    assert bound.base_model.bound_kwargs.get("tool_config") == {
        "include_server_side_tool_invocations": True
    }

    # Case 2: Explicit google_search in tools automatically ensures tool_config
    ep_google_no_ground = ModelEndpoint(
        provider=ModelProvider.GOOGLE,
        model_name="gemini-2.5-flash",
        enable_grounding=False,
    )
    wrapper_explicit = RobustChatModelWrapper(MockGeminiModel(), endpoint=ep_google_no_ground)
    bound_explicit = wrapper_explicit.bind_tools([dummy_tool, {"google_search": {}}])

    assert any(
        isinstance(t, dict) and "google_search" in t for t in bound_explicit.base_model.bound_tools
    )
    assert bound_explicit.base_model.bound_kwargs.get("tool_config") == {
        "include_server_side_tool_invocations": True
    }


def test_robust_chat_model_wrapper_grounding_non_google():
    """Verify RobustChatModelWrapper ignores grounding and strips google_search for non-Google providers."""
    from artemis.services.llm import RobustChatModelWrapper
    from langchain_core.tools import tool

    @tool
    def dummy_tool(query: str) -> str:
        """A test tool."""
        return query

    class MockOpenAIModel:
        def __init__(self):
            self.bound_tools = []
            self.bound_kwargs = {}

        def bind_tools(self, tools, *args, **kwargs):
            self.bound_tools = tools
            self.bound_kwargs = kwargs
            return self

    ep_openai = ModelEndpoint(
        provider=ModelProvider.OPENAI,
        model_name="gpt-4o",
        enable_grounding=True,
    )
    wrapper = RobustChatModelWrapper(MockOpenAIModel(), endpoint=ep_openai)
    bound = wrapper.bind_tools([dummy_tool, {"google_search": {}}])

    # Should ignore grounding and strip {"google_search": {}}
    assert len(bound.base_model.bound_tools) == 1
    assert bound.base_model.bound_tools[0].name == "dummy_tool"
    assert "tool_config" not in bound.base_model.bound_kwargs


def test_google_endpoint_is_redirected_to_openai():
    """Legacy Google choices cannot construct a native Google client."""
    from langchain_openai import ChatOpenAI
    from unittest.mock import patch

    ep = ModelEndpoint(provider=ModelProvider.GOOGLE, model_name="gemini-legacy", api_key="fake")
    with patch("google.genai.Client", side_effect=AssertionError("Google client is forbidden")):
        model = ModelFactory.create_model(ep)
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == settings.OPENAI_MODEL
    assert model.openai_api_base == settings.OPENAI_BASE_URL
