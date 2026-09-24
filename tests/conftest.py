from __future__ import annotations

import pytest

from model_backend import cloud_runner
from tests.fakes import FakeAnthropic, FakeOllama


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """No test touches the real data dir, a real API key, or a real Ollama."""
    monkeypatch.setenv("COUNCIL_DATA_DIR", str(tmp_path / "data"))
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
                "COUNCIL_BACKEND", "COUNCIL_SETTINGS", "OLLAMA_HOST"):
        monkeypatch.delenv(var, raising=False)
    cloud_runner.client.cache_clear()
    cloud_runner._probe.cache_clear()
    yield
    cloud_runner.client.cache_clear()
    cloud_runner._probe.cache_clear()


@pytest.fixture
def fake_ollama(monkeypatch):
    server = FakeOllama().start()
    monkeypatch.setenv("OLLAMA_HOST", server.url)
    yield server
    server.shutdown()


@pytest.fixture
def fake_anthropic(monkeypatch):
    server = FakeAnthropic().start()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", server.url)
    yield server
    server.shutdown()
