"""Top-level Typer application — wires each subcommand module's ``run`` function
into the ``book`` CLI. Subcommand bodies live in :mod:`books.commands.*`."""

import typer

from books.commands import (
    import_cmd,
    ls_cmd,
    reindex_cmd,
    rm_cmd,
    search_cmd,
    show_cmd,
)

app = typer.Typer(
    name="book",
    help="CLI bibliography manager — import PDFs, search them, organize a library.",
    no_args_is_help=True,
    add_completion=False,
)

# Each command module exposes a `run` callable; Typer turns it into a subcommand.
app.command("import")(import_cmd.run)
app.command("ls")(ls_cmd.run)
app.command("show")(show_cmd.run)
app.command("rm")(rm_cmd.run)
app.command("search")(search_cmd.run)
app.command("reindex")(reindex_cmd.run)
