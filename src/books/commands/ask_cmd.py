"""``book ask`` — answer a question from the library with a cited LLM response.

Retrieves the most relevant chunks (reusing :mod:`books.retrieval`), packs them
into a model-sized token budget, sends them to the configured LLM, and streams
back an answer with inline ``[n]`` citations followed by a references list.
"""

import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from books import config, db

console = Console()


def run(
    question: str = typer.Argument(..., help="Question to ask about the library."),
    k: int | None = typer.Option(
        None, "-k", "--max-chunks", help="Override rag.max_chunks for this run."
    ),
    author: str | None = typer.Option(None, "--author", help="Filter by author family name (substring)."),
    tag: str | None = typer.Option(None, "--tag", help="Filter by tag."),
    year_min: int | None = typer.Option(None, "--year-min", help="Minimum publication year (inclusive)."),
    year_max: int | None = typer.Option(None, "--year-max", help="Maximum publication year (inclusive)."),
    provider: str | None = typer.Option(None, "--provider", help="Override rag.provider (anthropic|openai|local)."),
    model: str | None = typer.Option(None, "--model", help="Override rag.model for this run."),
    show_context: bool = typer.Option(
        False, "--show-context", help="Print the packed chunks before answering."
    ),
) -> None:
    """Retrieve relevant chunks and synthesize a cited answer with an LLM.

    The number of chunks is bounded by ``rag.context_budget`` (token budget,
    sized to your model's context window) and ``rag.max_chunks``.  Configure the
    backend under the ``rag:`` config section; the local provider additionally
    needs ``uv sync --extra local``.  Use ``--author``/``--tag``/``--year-*`` to
    restrict the search to a subset of the library first.
    """
    from books.query import resolve_paper_ids
    from books.rag import budget, prompt, providers
    from books.retrieval import RetrievalUnavailable, retrieve

    max_chunks = k if k is not None else config.rag_max_chunks()

    with db.connect() as conn:
        allowed_ids = resolve_paper_ids(
            conn, author=author, tag=tag, year_min=year_min, year_max=year_max
        )
        if allowed_ids is not None and not allowed_ids:
            console.print("[yellow]no papers match the given filters[/yellow]")
            raise typer.Exit(code=1)

        try:
            results = retrieve(conn, question, max_chunks, allowed_ids=allowed_ids)
        except RetrievalUnavailable:
            console.print("[yellow]no chunks indexed yet[/yellow] — try `book reindex --all`")
            raise typer.Exit(code=1)

        if not results:
            console.print("[dim]no matches[/dim]")
            raise typer.Exit(code=1)

        packed = budget.pack(
            results,
            budget_tokens=config.rag_context_budget(),
            max_chunks=max_chunks,
        )
        context, refs = prompt.build_context(conn, packed)

    if show_context:
        for ref in refs:
            console.print(f"[cyan][{ref.n}][/cyan] [dim]{ref.label()}[/dim]")
        console.print()

    system = config.rag_system_prompt() or prompt.DEFAULT_SYSTEM_PROMPT
    user = prompt.build_user_message(context, question)

    if model is not None:
        config.config["rag"]["model"].set(model)

    try:
        client = providers.make_client(provider)
    except providers.LLMError as exc:
        console.print(f"[red]LLM error:[/red] {exc}")
        raise typer.Exit(code=1)

    answer = _stream_answer(client, system, user, config.rag_temperature())

    if not answer.strip():
        console.print("[dim]the model returned no answer[/dim]")
        raise typer.Exit(code=1)

    _render_references(refs)


def _stream_answer(client, system: str, user: str, temperature: float) -> str:
    """Stream the model's answer into a live panel, return the full text."""
    from books.rag.providers import LLMError

    chunks: list[str] = []
    try:
        with Live(console=console, refresh_per_second=12) as live:
            for delta in client.stream(system=system, user=user, temperature=temperature):
                chunks.append(delta)
                live.update(Panel(Markdown("".join(chunks)), title="Answer", border_style="green"))
    except LLMError as exc:
        console.print(f"[red]LLM error:[/red] {exc}")
        raise typer.Exit(code=1)
    except Exception as exc:  # surface SDK/runtime errors without a traceback
        console.print(f"[red]LLM error:[/red] {exc}")
        raise typer.Exit(code=1)
    return "".join(chunks)


def _render_references(refs) -> None:
    """Print the numbered references list under the answer."""
    if not refs:
        return
    body = Text()
    for ref in refs:
        body.append(f"[{ref.n}] ", style="cyan")
        body.append(ref.label())
        if ref.title:
            body.append(f"\n    {ref.title}", style="italic dim")
        body.append("\n")
    console.print(Panel(body, title="References", border_style="blue"))
