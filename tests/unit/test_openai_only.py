"""Ensure every execution path stays on the configured OpenAI-compatible service."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
import pytest

from artemis.config import get_default_llm_config, settings
from artemis.config.llm import deep_merge_llm_config
from artemis.llm.router import ModelEndpoint, ModelFactory, ModelProvider
from artemis.services.llm import _resolve_endpoint, get_configured_llm


@pytest.fixture
def requests(monkeypatch):
    monkeypatch.setattr(settings, "OPENAI_BASE_URL", "https://oneapi.example/v1")
    monkeypatch.setattr(settings, "OPENAI_API_KEY", SecretStr("test-oneapi-key"))
    monkeypatch.setattr(settings, "OPENAI_MODEL", "test-vision-model")
    # Stale Google credentials must never activate another provider.
    monkeypatch.setattr(settings, "GOOGLE_API_KEY", SecretStr("stale-google-key"))
    monkeypatch.delenv("ARTEMIS_FAKE_LLM", raising=False)
    monkeypatch.setattr(
        "google.genai.Client", Mock(side_effect=AssertionError("Google SDK called"))
    )
    captured = []

    def respond(request):
        captured.append(
            (str(request.url), request.headers["authorization"], json.loads(request.content))
        )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": settings.OPENAI_MODEL,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "test reply"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    transport = httpx.MockTransport(respond)
    client = httpx.Client(transport=transport)
    async_client = httpx.AsyncClient(transport=transport)

    def create_model(**kwargs):
        return ChatOpenAI(**kwargs, http_client=client, http_async_client=async_client)

    monkeypatch.setattr("langchain_openai.ChatOpenAI", create_model)
    ModelFactory._cache.clear()
    yield captured
    ModelFactory._cache.clear()
    client.close()


def assert_one_endpoint(captured):
    assert captured
    for url, authorization, payload in captured:
        assert url == "https://oneapi.example/v1/chat/completions"
        assert authorization == "Bearer test-oneapi-key"
        assert payload["model"] == "test-vision-model"


@pytest.mark.asyncio
async def test_all_roles_and_fallbacks_share_one_service(requests):
    cfg = get_default_llm_config()
    ctx = SimpleNamespace(llm_config=cfg)
    for name in type(cfg).model_fields:
        if name == "utils":
            continue
        for fallback in (False, True):
            endpoint = _resolve_endpoint(ctx, name, use_fallback=fallback)
            await ModelFactory.get_model(endpoint).ainvoke("test")
    for name in type(cfg.utils).model_fields:
        for fallback in (False, True):
            endpoint = _resolve_endpoint(ctx, name, is_utils=True, use_fallback=fallback)
            await ModelFactory.get_model(endpoint).ainvoke("test")
    assert_one_endpoint(requests)


def test_legacy_profile_cannot_restore_gemini(requests):
    cfg = deep_merge_llm_config(
        get_default_llm_config(),
        {
            "operator": {
                "provider": "google",
                "model": "legacy-gemini",
                "fallback": {"provider": "google", "model": "legacy-fallback"},
            }
        },
    )
    assert cfg.operator.provider == "openai"
    assert cfg.operator.model == "test-vision-model"
    assert cfg.operator.fallback.provider == "openai"
    endpoint = ModelEndpoint(
        provider=ModelProvider.GOOGLE,
        model_name="legacy-gemini",
        api_key="legacy-key",
        api_base="https://wrong.example",
    )
    ModelFactory.create_model(endpoint).invoke("test")
    assert_one_endpoint(requests)


@pytest.mark.asyncio
async def test_memory_lenses_and_flash_fallback_use_one_service(requests, monkeypatch):
    from artemis.agents.flash.runner import FlashRunner
    from artemis.agents.flash.summarizer import VisualStepSummarizer
    from artemis.memory.chunking import StepCapsuleLens

    ctx = SimpleNamespace(data_engine=None)
    summary = VisualStepSummarizer(ctx, model_name="legacy-gemini")
    await summary._llm.ainvoke("summarize")
    capsule = StepCapsuleLens(model_name="legacy-gemini", fallback_model_name="legacy-fallback")
    await capsule._get_llm().ainvoke("compress")
    await capsule._get_fallback_llm().ainvoke("compress fallback")
    await get_configured_llm().ainvoke("legacy helper")

    def fail(*args, **kwargs):
        raise ValueError("operator configuration failed")

    monkeypatch.setattr("artemis.agents.flash.runner.get_llm", fail)
    runner = FlashRunner.__new__(FlashRunner)
    runner.ctx = ctx
    await runner._init_llm().base_model.ainvoke("flash fallback")
    assert_one_endpoint(requests)


def test_native_clients_never_activate(requests):
    from artemis.agents.explorer.explorer import Explorer
    from artemis.agents.video_analyzer.video_analyzer import VideoAnalyzer

    explorer = Explorer.__new__(Explorer)
    explorer.ctx = SimpleNamespace(_genai_client=Mock())
    explorer.model_name = "gemini-legacy"
    assert explorer._detect_native_engine() is False
    with pytest.raises(RuntimeError, match="disabled"):
        explorer._ensure_genai_client()
    video = VideoAnalyzer.__new__(VideoAnalyzer)
    video.ctx = explorer.ctx
    video._resolve_video_processing = Mock()
    video._init_engine()
    assert video.use_native_gemini is False
    assert video.client is None
    assert video.model_name == "test-vision-model"


@pytest.mark.asyncio
async def test_prewarm_uses_configured_service(requests, monkeypatch):
    from artemis.sdk.agent import Agent

    monkeypatch.setattr("artemis.sdk.agent.publish_startup_progress", Mock())
    agent = Agent.__new__(Agent)
    agent._session_id = None
    await agent._prewarm_llm_connections(api_key="old-google-key")
    assert_one_endpoint(requests)


def test_missing_configuration_never_falls_back_to_another_service(requests, monkeypatch):
    monkeypatch.setattr(settings, "OPENAI_BASE_URL", None)
    with pytest.raises(ValueError, match="OPENAI_BASE_URL"):
        ModelFactory.create_model(ModelEndpoint(provider=ModelProvider.GOOGLE))
    assert not requests


def test_changing_endpoint_key_and_model_refreshes_cached_client(requests, monkeypatch):
    first = ModelFactory.get_model(ModelEndpoint())
    first.invoke("first")
    monkeypatch.setattr(settings, "OPENAI_BASE_URL", "https://new-oneapi.example/v1")
    monkeypatch.setattr(settings, "OPENAI_API_KEY", SecretStr("new-test-key"))
    monkeypatch.setattr(settings, "OPENAI_MODEL", "new-vision-model")
    second = ModelFactory.get_model(ModelEndpoint())
    second.invoke("second")
    assert first is not second
    url, authorization, payload = requests[-1]
    assert url == "https://new-oneapi.example/v1/chat/completions"
    assert authorization == "Bearer new-test-key"
    assert payload["model"] == "new-vision-model"


@pytest.mark.asyncio
async def test_image_and_function_schema_stay_on_chat_completions(requests):
    model = ModelFactory.create_model(ModelEndpoint())
    tool = {
        "type": "function",
        "function": {
            "name": "report",
            "description": "Report a result",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Inspect this image"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,example"}},
        ]
    )
    await model.bind_tools([tool]).ainvoke([message])
    assert_one_endpoint(requests)
    payload = requests[-1][2]
    assert payload["tools"][0]["function"]["name"] == "report"
    assert payload["messages"][0]["content"][1]["type"] == "image_url"


@pytest.mark.asyncio
async def test_ocr_uses_the_same_endpoint_and_key(requests, monkeypatch):
    import base64
    import io
    from PIL import Image
    from artemis.utils.ocr_api import perform_ocr

    monkeypatch.setattr(settings, "ARTEMIS_OCR_ENABLED", True)
    image = Image.new("RGB", (100, 40), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    screenshot = base64.b64encode(buffer.getvalue()).decode()

    labels = [
        {
            "text": "Hello",
            "position": [
                {"x": 0, "y": 0},
                {"x": 40, "y": 0},
                {"x": 40, "y": 20},
                {"x": 0, "y": 20},
            ],
        }
    ]

    def respond(request):
        requests.append(
            (str(request.url), request.headers["authorization"], json.loads(request.content))
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(labels)}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        result = await perform_ocr(screenshot, client=client)
    assert result == labels
    assert_one_endpoint(requests)


@pytest.mark.asyncio
async def test_diagnostics_verify_only_the_configured_service(requests, monkeypatch):
    from artemis.core.diagnostics.probes.credentials_probe import LLMCredentialsProbe
    from artemis.utils.credentials_validator import validate_api_key

    probe = await LLMCredentialsProbe().probe()
    assert [p["provider"] for p in probe.metadata["providers"]] == ["openai"]
    captured = []

    def respond(request):
        captured.append(request)
        return httpx.Response(200, json={"data": [{"id": "test-vision-model"}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(
        "artemis.utils.credentials_validator.httpx.AsyncClient", lambda **kw: client
    )
    valid, _ = await validate_api_key("openai", "test-oneapi-key")
    assert valid
    assert str(captured[0].url) == "https://oneapi.example/v1/models"
    assert captured[0].headers["authorization"] == "Bearer test-oneapi-key"
    valid, _ = await validate_api_key("google", "stale-google-key")
    assert not valid
    assert len(captured) == 1
