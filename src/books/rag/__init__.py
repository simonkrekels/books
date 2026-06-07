"""Retrieval-augmented generation for ``book ask``.

Three small, independently testable pieces:

* :mod:`books.rag.budget`    — pack ranked chunks into a token budget.
* :mod:`books.rag.prompt`    — build the numbered context + citation references.
* :mod:`books.rag.providers` — pluggable LLM backends (Anthropic / OpenAI / local).
"""
