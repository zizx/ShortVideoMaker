"""Fluxion AI integration regression: real SDK encode/decode, while the HTTP layer
never reaches the network or consumes quota."""

import json
from unittest.mock import patch

import httpx
import pytest
from openai import OpenAI

from app.config import config
from app.models.llm_provider import get_llm_provider
from app.services import llm


@pytest.mark.parametrize(
    "base_url,model,expected_url,expected_model",
    [
        ("", "", "https://fluxionai.space/v1/chat/completions", "gpt-5.5"),
        ("  ", "  ", "https://fluxionai.space/v1/chat/completions", "gpt-5.5"),
        (
            "https://gateway.example.com/custom/v1/",
            "custom-group-model",
            "https://gateway.example.com/custom/v1/chat/completions",
            "custom-group-model",
        ),
    ],
)
def test_fluxionai_defaults_and_group_overrides(
    base_url, model, expected_url, expected_model
):
    """Empty config uses defaults and custom group values are not overridden;
    other providers' configuration is left untouched."""
    snapshot = dict(config.app)
    requests = []

    def respond(request):
        requests.append(request)
        assert str(request.url) == expected_url
        assert request.headers["Authorization"] == "Bearer test-fluxion-key"
        assert json.loads(request.content) == {
            "model": expected_model,
            "messages": [{"role": "user", "content": "生成中文文案"}],
        }
        return httpx.Response(
            200,
            json={
                "id": "test-completion",
                "object": "chat.completion",
                "created": 0,
                "model": expected_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "你好，世界！"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    # Keep the real SDK's URL joining and response parsing and replace only the
    # HTTP transport, so a mock cannot hide protocol problems.
    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        with OpenAI(
            api_key="test-fluxion-key",
            base_url=base_url.strip() or "https://fluxionai.space/v1",
            http_client=http_client,
            max_retries=0,
        ) as client:
            with patch.object(llm, "OpenAI", return_value=client) as factory:
                result = llm._generate_response(
                    "生成中文文案",
                    app_config={
                        "llm_provider": "fluxionai",
                        "fluxionai_api_key": "test-fluxion-key",
                        "fluxionai_base_url": base_url,
                        "fluxionai_model_name": model,
                    },
                )
                factory.assert_called_once_with(
                    api_key="test-fluxion-key",
                    base_url=base_url.strip() or "https://fluxionai.space/v1",
                )
    assert result == "你好，世界！"
    assert len(requests) == 1
    assert config.app == snapshot


def test_fluxionai_missing_key_fails_before_network():
    """A missing key keeps the unified error message and creates no client or
    billable request."""
    with patch.object(llm, "OpenAI") as factory:
        result = llm._generate_response(
            "test", app_config={"llm_provider": "fluxionai"}
        )
    assert result.startswith("Error:")
    assert "fluxionai: api_key is not set" in result
    factory.assert_not_called()


@pytest.mark.parametrize("status", [401, 403, 429, 500])
def test_fluxionai_http_errors_are_not_successful_text(status):
    """Authentication, group-permission, rate-limit and server errors must take
    the error path and never be treated as normal copy."""

    def respond(request):
        return httpx.Response(
            status,
            json={"error": {"message": "test upstream failure", "type": "api_error"}},
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        with OpenAI(
            api_key="test-key",
            base_url="https://fluxionai.space/v1",
            http_client=http_client,
            max_retries=0,
        ) as client:
            with patch.object(llm, "OpenAI", return_value=client):
                result = llm._generate_response(
                    "test",
                    app_config={
                        "llm_provider": "fluxionai",
                        "fluxionai_api_key": "test-key",
                    },
                )
    assert result.startswith("Error:")
    assert str(status) in result


def test_fluxionai_registry_metadata():
    """Lock the default model, protocol, promotion parameters and model link so
    different entry points cannot disagree."""
    provider = get_llm_provider("fluxionai")
    assert provider.default_model == "gpt-5.5"
    assert provider.default_base_url == "https://fluxionai.space/v1"
    assert provider.adapter == "openai_compatible"
    assert provider.requires_api_key
    assert (
        provider.api_key_url
        == "https://fluxionai.space/register?source=github&campaign=moneyprinterturbo&promo=MONEYPRINTERTURBO"
    )
    assert provider.model_docs_url == "https://fluxionai.space/model-plaza"
