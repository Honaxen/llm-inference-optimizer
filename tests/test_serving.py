"""
Tests for the serving layer (serving/vllm_server.py).

These tests mock the vLLM engine instead of loading a real model — the
point is to verify the FastAPI layer (request validation, response shape,
error handling, metrics) works correctly, not to re-test vLLM itself.
Running against a real model belongs in a separate manual/integration
check, not in the unit test suite.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent / "serving"))

import vllm_server  # noqa: E402


class FakeCompletionOutput:
    def __init__(self, text, token_ids):
        self.text = text
        self.token_ids = token_ids


class FakeRequestOutput:
    def __init__(self, prompt_token_ids, completion_text, completion_token_ids):
        self.prompt_token_ids = prompt_token_ids
        self.outputs = [FakeCompletionOutput(completion_text, completion_token_ids)]


@pytest.fixture
def client(monkeypatch):
    """
    Set up a TestClient with a fake engine and config, bypassing the real
    lifespan startup (which would try to load an actual model onto a GPU).
    """
    vllm_server.server_config = {"model": "fake-model-for-tests"}

    fake_engine = MagicMock()

    async def fake_generate(prompt, sampling_params, request_id):
        # vLLM's real generate() is an async generator; mimic that shape,
        # yielding one final output like a completed (non-streaming) request.
        yield FakeRequestOutput(
            prompt_token_ids=[1, 2, 3],
            completion_text="This is a fake response.",
            completion_token_ids=[4, 5, 6, 7],
        )

    fake_engine.generate = fake_generate
    vllm_server.engine = fake_engine

    with TestClient(vllm_server.app) as test_client:
        yield test_client

    vllm_server.engine = None
    vllm_server.server_config = {}


def test_health_when_engine_ready(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"] == "fake-model-for-tests"


def test_health_when_engine_not_ready(client):
    vllm_server.engine = None
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "starting"


def test_generate_returns_expected_shape(client):
    response = client.post("/generate", json={"prompt": "Hello, world"})
    assert response.status_code == 200

    body = response.json()
    assert body["response"] == "This is a fake response."
    assert body["model"] == "fake-model-for-tests"
    assert body["prompt_tokens"] == 3
    assert body["completion_tokens"] == 4
    assert body["latency_ms"] >= 0


def test_generate_rejects_missing_prompt(client):
    response = client.post("/generate", json={})
    assert response.status_code == 422  # FastAPI/Pydantic validation error


def test_generate_returns_503_when_engine_not_ready(client):
    vllm_server.engine = None
    response = client.post("/generate", json={"prompt": "Hello"})
    assert response.status_code == 503


def test_generate_returns_500_on_engine_error(client):
    async def failing_generate(prompt, sampling_params, request_id):
        raise RuntimeError("engine exploded")
        yield  # pragma: no cover — unreachable, makes this an async generator

    vllm_server.engine.generate = failing_generate
    response = client.post("/generate", json={"prompt": "Hello"})
    assert response.status_code == 500


def test_metrics_endpoint_exposes_prometheus_format(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "inference_requests_total" in response.text
