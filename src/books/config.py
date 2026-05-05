"""Configuration access — thin wrappers around a confuse_ ``Configuration``.

The package ships ``config_default.yaml`` next to this module; user overrides
are read from confuse's standard application directory (on macOS that's
``~/Library/Application Support/book/config.yaml``; can be overridden by the
``BOOKDIR`` environment variable). Every helper here returns the resolved,
expanded value — call sites never touch the raw confuse view.

.. _confuse: https://confuse.readthedocs.io/
"""

from pathlib import Path

import confuse

# Singleton configuration. confuse merges the shipped default YAML with any
# user override file at the OS-appropriate location.
config = confuse.Configuration("book", "books")


def _expand(s: str) -> Path:
    """Expand ``~`` and resolve to an absolute path. Does not require existence."""
    return Path(s).expanduser().resolve()


def library_dir() -> Path:
    """Where imported PDF files live (templated subdirs underneath)."""
    return _expand(config["library_dir"].as_str())


def db_path() -> Path:
    """Path to the SQLite library file."""
    return _expand(config["db_path"].as_str())


def chroma_dir() -> Path:
    """Directory holding the Chroma vector index (created on first use)."""
    return _expand(config["chroma_dir"].as_str())


def import_mode() -> str:
    """How to place PDFs during import: ``copy``, ``move``, or ``symlink``."""
    mode = config["import"]["mode"].as_str()
    if mode not in ("copy", "move", "symlink"):
        raise ValueError(f"import.mode must be copy/move/symlink, got: {mode!r}")
    return mode


def path_template() -> str:
    """Format-string template used to build the on-disk PDF path."""
    return config["import"]["path_template"].as_str()


def crossref_user_agent() -> str:
    """User-Agent string sent to Crossref. Including a mailto qualifies for the
    polite-pool (better rate limits)."""
    return config["metadata"]["user_agent"].as_str()


def metadata_sources() -> list[str]:
    """List of metadata sources to consult during import (e.g. ``[crossref, arxiv]``)."""
    return [s for s in config["metadata"]["sources"].get(list)]


def embedder_model() -> str:
    """HuggingFace model name for the local sentence-transformers embedder."""
    return config["index"]["model"].as_str()


def chunk_tokens() -> int:
    """Target chunk length in tokens (converted to chars in the chunker)."""
    return int(config["index"]["chunk_tokens"].as_number())


def chunk_overlap() -> int:
    """Token overlap between consecutive chunks (preserves context across boundaries)."""
    return int(config["index"]["chunk_overlap"].as_number())


def embedder_device() -> str:
    """Torch device string for the embedder: ``cpu``, ``mps``, or ``cuda``."""
    return config["index"]["device"].as_str()


def embedder_offline() -> bool:
    """If true, never contact HuggingFace Hub — use only cached weights."""
    return bool(config["index"]["offline"].get(bool))
