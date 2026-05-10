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

`book` is a beets-style CLI bibliography manager: it imports PDFs, fetches metadata from Crossref/arXiv/Open Library, organizes files via configurable path templates, and provides hybrid full-text search (BM25 via SQLite FTS5 + cosine via Chroma, fused with Reciprocal Rank Fusion).

### Data flow for `book import`

```
PDF → pdf_meta.py (sniff DOI/arXiv/ISBN, weighted scoring)
    → crossref.py / arxiv.py / openlibrary.py (REST lookups → PaperMatch)
    → interactive.py (Rich prompt: [A]pply / [S]kip / [M]anual / [U]se PDF / [R]etry / [Q]uit)
    → db.py (SQLite: papers, authors, paper_authors, tags)
    → paths.py (render template → copy/move/symlink PDF)
    → indexer.py → extract.py → chunker.py → embedder.py → chroma.py
```

**SQLite is the source of truth.** The `chunks` table stores all chunk text and powers BM25 search. The Chroma vector index is derived; rebuild with `book reindex --all`. Indexing failures set `needs_reindex=1` rather than rolling back the import.

### Key modules

| Module | Responsibility |
|---|---|
| `cli.py` | Typer app; subcommand registration |
| `config.py` + `config_default.yaml` | confuse-based typed config; defaults shipped with package |
| `db.py` | SQLite schema (v2), migrations, `connect()` context manager |
| `query.py` | Parameterized SQL builder from CLI filter flags |
| `importer.py` | Orchestrate full import pipeline; SHA-256 dedup + early DOI/arXiv duplicate check |
| `paths.py` | `slugify()`, `render_template()`, `place_pdf()` |
| `interactive.py` | Rich prompts for import; `manual_entry_form()`, `build_match_from_pdf_meta()` |
| `metadata/pdf_meta.py` | DOI/arXiv/ISBN extraction with weighted page scoring |
| `index/indexer.py` | Orchestrate extract→chunk→embed→upsert; singleton embedder |
| `index/fts.py` | SQLite FTS5 BM25 helpers (`upsert_paper`, `search`, `rrf_fuse`) |
| `index/chroma.py` | ChromaIndex wrapper around PersistentClient |
| `commands/tag_cmd.py` | `book tag add/rm/ls` — manage paper tags |
| `commands/bibtex_cmd.py` | `book bibtex` — export BibTeX entries by id or tag |

### Configuration

User config lives at `~/Library/Application Support/book/config.yaml` (macOS). Key defaults from `config_default.yaml`:

- `library_dir: ~/Documents/papers`
- `db_path: ~/.local/share/book/library.db`
- `import.mode: move` (copy | move | symlink)
- `import.path_template: "{author_last}/{year}/{title_slug}.pdf"`
- `index.model: BAAI/bge-small-en-v1.5`
- `index.hybrid: true` — fuse BM25 (SQLite FTS5) + cosine via RRF; falls back to cosine-only if chunks table is empty
- `index.query_prompt` — instruction prefix prepended to queries for BGE asymmetric retrieval (not applied to indexed chunks)
- `index.offline: true` — skips HuggingFace Hub network checks (set in user config)

Switching embedding models requires `book reindex --all` — Chroma stores vectors at a fixed dimension and silently returns wrong results if the query model mismatches the indexed model.

### Database schema (v3)

Tables: `papers`, `authors`, `paper_authors` (preserves order), `tags`, `schema_version`, `chunks`, `chunks_fts` (FTS5 virtual table, rowids mirror `chunks.id`). `init_db()` runs on every `connect()` and is idempotent — migrations are additive only. New columns are added via `ALTER TABLE` in `_migrate()`, never by modifying `SCHEMA` directly.

`chunks` is the canonical text store for all indexed content; `chunks_fts` is kept in sync manually by `index/fts.py` — every write to `chunks` must pair with a corresponding FTS write.

### Testing

Tests use `tmp_db` fixture for isolated SQLite databases and `pytest-httpx` for mocking HTTP clients. No real network calls or filesystem side effects in tests.
