"""Pluggable LLM backends for ``book ask``.

Three adapters behind a common :class:`LLMClient` protocol:

* :class:`AnthropicClient` — the Anthropic API (Claude).
* :class:`OpenAIClient`    — any OpenAI-compatible endpoint.  Its ``base_url``
  knob makes it cover hosted GPT *and* local servers (Ollama, LM Studio, vLLM,
  llama.cpp's server).
* :class:`LocalClient`     — an in-process GGUF model via ``llama-cpp-python``
  (optional ``[local]`` extra), for fully offline use with no server.

Each SDK is imported lazily inside its constructor so non-RAG commands — and
RAG runs that use a different provider — never pay for unused imports.  All
constructors accept an injected ``client`` for testing.
"""

import os
from collections.abc import Iterator
from typing import Any, Protocol

from books import config


class LLMError(Exception):
    """Raised for provider configuration or call failures surfaced to the user."""


class LLMClient(Protocol):
    """A streaming chat completion backend."""

    def stream(self, *, system: str, user: str, temperature: float) -> Iterator[str]:
        """Yield answer text deltas for the given system + user messages."""
        ...


class AnthropicClient:
    """Anthropic Messages API backend."""

    def __init__(
        self,
        model: str,
        *,
        api_key_env: str,
        base_url: str,
        max_tokens: int,
        client: Any | None = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        if client is not None:
            self._client = client
            return
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - anthropic is a core dep
            raise LLMError("the `anthropic` package is not installed") from exc
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise LLMError(f"no API key found in ${api_key_env}")
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = anthropic.Anthropic(**kwargs)

    def stream(self, *, system: str, user: str, temperature: float) -> Iterator[str]:
        with self._client.messages.stream(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        ) as stream:
            yield from stream.text_stream


class OpenAIClient:
    """OpenAI-compatible Chat Completions backend (hosted or local server)."""

    def __init__(
        self,
        model: str,
        *,
        api_key_env: str,
        base_url: str,
        max_tokens: int,
        client: Any | None = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        if client is not None:
            self._client = client
            return
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - openai is a core dep
            raise LLMError("the `openai` package is not installed") from exc
        # Local OpenAI-compatible servers usually need no real key; supply a
        # placeholder when a custom base_url is set and no key is present.
        api_key = os.environ.get(api_key_env) or ("not-needed" if base_url else None)
        if not api_key:
            raise LLMError(f"no API key found in ${api_key_env}")
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**kwargs)

    def stream(self, *, system: str, user: str, temperature: float) -> Iterator[str]:
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=temperature,
            max_tokens=self._max_tokens,
            stream=True,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        for chunk in resp:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class LocalClient:
    """In-process GGUF model via ``llama-cpp-python``."""

    def __init__(
        self,
        *,
        model_path: str,
        n_ctx: int,
        max_tokens: int,
        client: Any | None = None,
    ) -> None:
        self._max_tokens = max_tokens
        if client is not None:
            self._client = client
            return
        try:
            import llama_cpp  # type: ignore[import-not-found]
        except ImportError as exc:
            raise LLMError(
                "the local backend needs llama-cpp-python — run "
                "`uv sync --extra local`"
            ) from exc
        if not model_path or not os.path.exists(model_path):
            raise LLMError(
                f"rag.model_path does not point to a model file: {model_path!r}"
            )
        self._client = llama_cpp.Llama(
            model_path=model_path, n_ctx=n_ctx, verbose=False
        )

    def stream(self, *, system: str, user: str, temperature: float) -> Iterator[str]:
        resp = self._client.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=self._max_tokens,
            stream=True,
        )
        for chunk in resp:
            delta = chunk["choices"][0]["delta"].get("content")
            if delta:
                yield delta


def make_client(provider: str | None = None) -> LLMClient:
    """Build the configured (or explicitly named) LLM client.

    *provider* overrides ``rag.provider`` for a one-off run; everything else is
    read from config.
    """
    provider = provider or config.rag_provider()
    max_tokens = config.rag_reserve_output()
    if provider == "anthropic":
        return AnthropicClient(
            config.rag_model(),
            api_key_env=config.rag_api_key_env(),
            base_url=config.rag_base_url(),
            max_tokens=max_tokens,
        )
    if provider == "openai":
        return OpenAIClient(
            config.rag_model(),
            api_key_env=config.rag_api_key_env(),
            base_url=config.rag_base_url(),
            max_tokens=max_tokens,
        )
    if provider == "local":
        return LocalClient(
            model_path=config.rag_model_path(),
            n_ctx=config.rag_n_ctx(),
            max_tokens=max_tokens,
        )
    raise LLMError(f"unknown rag.provider: {provider!r}")
