"""Tests for the LLM provider adapters (books.rag.providers).

No real network or model loads: each adapter is driven with an injected fake
client, and the local-import failure path is checked by hiding the module.
"""

import builtins

import pytest

from books.rag import providers


# ---------------------------------------------------------------------------
# Fake SDK clients
# ---------------------------------------------------------------------------


class _FakeAnthropicStream:
    def __init__(self, deltas):
        self.text_stream = iter(deltas)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeAnthropic:
    def __init__(self, deltas):
        self._deltas = deltas
        self.messages = self

    def stream(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeAnthropicStream(self._deltas)


class _Delta:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.delta = _Delta(content)


class _OpenAIChunk:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _FakeOpenAI:
    def __init__(self, deltas):
        self._deltas = deltas
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return iter(_OpenAIChunk(d) for d in self._deltas)


class _FakeLlama:
    def __init__(self, deltas):
        self._deltas = deltas

    def create_chat_completion(self, **kwargs):
        self.last_kwargs = kwargs
        return iter({"choices": [{"delta": {"content": d}}]} for d in self._deltas)


# ---------------------------------------------------------------------------
# Streaming behaviour
# ---------------------------------------------------------------------------


def test_anthropic_stream_yields_deltas():
    client = providers.AnthropicClient(
        "claude-x", api_key_env="X", base_url="", max_tokens=100,
        client=_FakeAnthropic(["Hello", " world"]),
    )
    out = list(client.stream(system="s", user="u", temperature=0.0))
    assert out == ["Hello", " world"]


def test_openai_stream_yields_deltas_and_skips_none():
    client = providers.OpenAIClient(
        "gpt-x", api_key_env="X", base_url="", max_tokens=100,
        client=_FakeOpenAI(["foo", None, "bar"]),
    )
    out = list(client.stream(system="s", user="u", temperature=0.0))
    assert out == ["foo", "bar"]


def test_local_stream_yields_deltas():
    client = providers.LocalClient(
        model_path="ignored", n_ctx=2048, max_tokens=100,
        client=_FakeLlama(["a", "b", "c"]),
    )
    out = list(client.stream(system="s", user="u", temperature=0.0))
    assert out == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Configuration / dispatch
# ---------------------------------------------------------------------------


def test_make_client_dispatches_on_provider(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(providers.config, "rag_model", lambda: "claude-x")
    monkeypatch.setattr(providers.config, "rag_api_key_env", lambda: "ANTHROPIC_API_KEY")
    monkeypatch.setattr(providers.config, "rag_base_url", lambda: "")
    monkeypatch.setattr(providers.config, "rag_reserve_output", lambda: 256)

    captured = {}

    class _Stub:
        def __init__(self, *a, **k):
            captured["kind"] = "anthropic"

    monkeypatch.setattr(providers, "AnthropicClient", _Stub)
    providers.make_client("anthropic")
    assert captured["kind"] == "anthropic"


def test_make_client_unknown_provider_raises():
    with pytest.raises(providers.LLMError):
        providers.make_client("nope")


def test_anthropic_missing_key_raises(monkeypatch):
    monkeypatch.delenv("MISSING_KEY", raising=False)
    with pytest.raises(providers.LLMError, match="MISSING_KEY"):
        providers.AnthropicClient("m", api_key_env="MISSING_KEY", base_url="", max_tokens=10)


def test_local_missing_llama_cpp_raises_install_hint(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "llama_cpp":
            raise ImportError("no module")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(providers.LLMError, match="uv sync --extra local"):
        providers.LocalClient(model_path="x", n_ctx=2048, max_tokens=10)


def test_local_missing_model_file_raises(monkeypatch):
    # llama_cpp import succeeds (stubbed) but the path is invalid.
    import sys
    import types

    fake_mod = types.ModuleType("llama_cpp")
    fake_mod.Llama = lambda **kw: None
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_mod)
    with pytest.raises(providers.LLMError, match="model_path"):
        providers.LocalClient(model_path="/nonexistent.gguf", n_ctx=2048, max_tokens=10)
