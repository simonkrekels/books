# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

```bash
uv sync                        # Install all dependencies
uv run pytest -q               # Run full test suite (~2 seconds)
uv run pytest tests/test_db.py # Run a single test file
uv run ruff check .            # Lint
uv run mypy src/books          # Type check
uv run book [cmd]              # Run a subcommand during development
```

## Architecture

`book` is a beets-style CLI bibliography manager: it imports PDFs, fetches metadata from Crossref/arXiv/Open Library, organizes files via configurable path templates, and provides full-text semantic search via a local Chroma vector index.

### Data flow for `book import`

```
PDF → pdf_meta.py (sniff DOI/arXiv/ISBN, weighted scoring)
    → crossref.py / arxiv.py / openlibrary.py (REST lookups → PaperMatch)
    → interactive.py (Rich prompt: [A]pply / [S]kip / [M]anual / [Q]uit)
    → db.py (SQLite: papers, authors, paper_authors, tags)
    → paths.py (render template → copy/move/symlink PDF)
    → indexer.py → extract.py → chunker.py → embedder.py → chroma.py
```

**SQLite is the source of truth.** The Chroma vector index is derived; rebuild with `book reindex --all`. Indexing failures set `needs_reindex=1` rather than rolling back the import.

### Key modules

| Module | Responsibility |
|---|---|
| `cli.py` | Typer app; subcommand registration |
| `config.py` + `config_default.yaml` | confuse-based typed config; defaults shipped with package |
| `db.py` | SQLite schema (v2), migrations, `connect()` context manager |
| `query.py` | Parameterized SQL builder from CLI filter flags |
| `importer.py` | Orchestrate full import pipeline; deduplication via SHA-256 |
| `paths.py` | `slugify()`, `render_template()`, `place_pdf()` |
| `interactive.py` | Rich prompts for import confirmation |
| `metadata/pdf_meta.py` | DOI/arXiv/ISBN extraction with weighted page scoring |
| `index/indexer.py` | Orchestrate extract→chunk→embed→upsert; singleton embedder |
| `index/chroma.py` | ChromaIndex wrapper around PersistentClient |

### Configuration

User config lives at `~/Library/Application Support/book/config.yaml` (macOS). Key defaults from `config_default.yaml`:

- `library_dir: ~/Documents/papers`
- `db_path: ~/.local/share/book/library.db`
- `import.mode: move` (copy | move | symlink)
- `import.path_template: "{author_last}/{year}/{title_slug}.pdf"`
- `index.model: BAAI/bge-small-en-v1.5`
- `index.offline: false` — set `true` to skip HuggingFace Hub checks

### Database schema (v2)

Tables: `papers`, `authors`, `paper_authors` (preserves order), `tags`, `schema_version`. `init_db()` runs on every `connect()` and is idempotent — migrations are additive only.

### Testing

Tests use `tmp_db` fixture for isolated SQLite databases and `pytest-httpx` for mocking HTTP clients. No real network calls or filesystem side effects in tests.
