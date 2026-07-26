"""Colorful terminal UI for the OpenDynamicGGUF CLI."""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from typing import Any, Iterable, Sequence

try:
    from rich import box
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.table import Table
    from rich.text import Text
    from rich.theme import Theme

    _HAS_RICH = True
except ImportError:  # pragma: no cover
    _HAS_RICH = False


_THEME = (
    Theme(
        {
            "odg.brand": "bold cyan",
            "odg.step": "bold bright_white",
            "odg.ok": "bold green",
            "odg.warn": "bold yellow",
            "odg.err": "bold red",
            "odg.muted": "dim",
            "odg.key": "cyan",
            "odg.val": "white",
            "odg.next": "bold magenta",
            "odg.done": "green",
            "odg.pending": "dim",
            "odg.running": "yellow",
            "odg.failed": "red",
        }
    )
    if _HAS_RICH
    else None
)

console = Console(theme=_THEME, highlight=False) if _HAS_RICH else None


def enabled(explain: bool = True) -> bool:
    return bool(explain and _HAS_RICH and console is not None)


def plain_print(*args: Any, **kwargs: Any) -> None:
    print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Banners / headers
# ---------------------------------------------------------------------------


def step_banner(
    step_no: int,
    title: str,
    *,
    model: str,
    run_id: str,
    root: str,
    goal: str,
    bullets: Sequence[str] | None = None,
    explain: bool = True,
) -> None:
    if not enabled(explain):
        if explain:
            print(f"\nOpenDynamicGGUF — Step {step_no:02d}: {title}")
            print(f"  model  : {model}")
            print(f"  run_id : {run_id}")
            print(f"  root   : {root}")
            print(f"  goal   : {goal}")
        return

    assert console is not None
    header = Text()
    header.append("OpenDynamicGGUF", style="odg.brand")
    header.append("  ·  ", style="odg.muted")
    header.append(f"Step {step_no:02d}", style="odg.step")
    header.append(f"  {title}", style="bold")

    body = Table.grid(padding=(0, 2))
    body.add_column(style="odg.key", no_wrap=True)
    body.add_column(style="odg.val")
    body.add_row("model", model)
    body.add_row("run_id", run_id)
    body.add_row("root", root)
    body.add_row("goal", goal)
    if bullets:
        body.add_row("", "")
        for b in bullets:
            body.add_row("•", b)

    console.print()
    console.print(
        Panel(
            body,
            title=header,
            title_align="left",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )


def already_done(
    step_no: int,
    step_id: str,
    *,
    run_id: str,
    path: Any,
    output: dict | None,
    summary: str | None = None,
    explain: bool = True,
) -> None:
    if not enabled(explain):
        if explain:
            print(f"Step {step_no:02d} already checkpointed as done — loading from store.")
            print(f"  run   : {run_id}")
            print(f"  path  : {path}")
            print("  (pass --force to re-run)")
            if output is not None:
                print("\n=== stored output ===")
                print(json.dumps(output, indent=2))
            if summary:
                print("\n" + summary)
        elif output is not None:
            print(json.dumps(output, indent=2))
        return

    assert console is not None
    console.print()
    console.print(
        Panel(
            Text.from_markup(
                f"[odg.ok]✓ Step {step_no:02d} already done[/]\n"
                f"[odg.muted]Loading checkpoint — pass [bold]--force[/] to re-run.[/]\n\n"
                f"[odg.key]run[/]   {run_id}\n"
                f"[odg.key]path[/]  {path}"
            ),
            title="[odg.ok]checkpoint hit[/]",
            border_style="green",
            box=box.ROUNDED,
        )
    )
    if output is not None:
        json_panel(output, title="stored output")
    if summary:
        run_summary(summary)


def checkpoint_saved(
    *,
    run_id: str,
    step_dir: Any,
    files: Sequence[str] | None = None,
    explain: bool = True,
) -> None:
    files = files or ("input.json", "output.json", "status.json", "log.txt")
    if not enabled(explain):
        if explain:
            print("\n=== checkpoint saved ===")
            print(f"  run_id   : {run_id}")
            print(f"  step dir : {step_dir}")
            print(f"  files    : {', '.join(files)}")
        return

    assert console is not None
    console.print()
    console.print(
        Panel(
            Text.from_markup(
                f"[odg.ok]✓ Durable checkpoint written[/]\n\n"
                f"[odg.key]run_id[/]    {run_id}\n"
                f"[odg.key]step dir[/]  {step_dir}\n"
                f"[odg.key]files[/]     {', '.join(files)}"
            ),
            title="[odg.ok]saved[/]",
            border_style="green",
            box=box.ROUNDED,
        )
    )


def error(step_no: int, step_id: str, exc: BaseException, path: Any) -> None:
    if not _HAS_RICH or console is None:
        print(f"\nERROR in Step {step_no:02d} {step_id}: {exc}", file=sys.stderr)
        print(f"Checkpointed failure → {path}")
        return
    console.print()
    console.print(
        Panel(
            Text.from_markup(
                f"[odg.err]✗ Step {step_no:02d} {step_id} failed[/]\n\n"
                f"[odg.err]{exc}[/]\n\n"
                f"[odg.muted]Checkpointed failure →[/] {path}"
            ),
            title="[odg.err]error[/]",
            border_style="red",
            box=box.ROUNDED,
        )
    )


# ---------------------------------------------------------------------------
# Live processing
# ---------------------------------------------------------------------------


@contextmanager
def working(message: str, *, explain: bool = True):
    """Spinner while a step computes."""
    import time

    if not enabled(explain):
        if explain:
            print(f"\n… {message}")
        t0 = time.perf_counter()
        yield
        if explain:
            print(f"✓ {message} ({time.perf_counter() - t0:.2f}s)")
        return
    assert console is not None
    t0 = time.perf_counter()
    with console.status(
        Text.from_markup(f"[odg.running]⚙ {message}[/]"),
        spinner="dots12",
        spinner_style="cyan",
    ):
        yield
    dt = time.perf_counter() - t0
    console.print(
        Text.from_markup(f"[odg.ok]✓[/] [odg.muted]{message}[/] [odg.key]({dt:.2f}s)[/]")
    )


@contextmanager
def progress_bar(total: int, description: str = "Working", *, explain: bool = True):
    """Determinate progress bar for multi-item work."""
    if not enabled(explain) or total <= 0:
        yield None
        return
    assert console is not None
    with Progress(
        SpinnerColumn("dots12"),
        TextColumn("[odg.key]{task.description}"),
        BarColumn(bar_width=28, complete_style="cyan", finished_style="green"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task_id = progress.add_task(description, total=total)
        yield lambda n=1, **kw: progress.update(task_id, advance=n, **kw)


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------


def section(title: str, *, explain: bool = True) -> None:
    if not enabled(explain):
        if explain:
            print(f"\n=== {title} ===")
        return
    assert console is not None
    console.print()
    console.print(Text(f"▸ {title}", style="bold cyan"))


def bullets(lines: Iterable[str], *, explain: bool = True) -> None:
    if not enabled(explain):
        if explain:
            for line in lines:
                print(f"  • {line}")
        return
    assert console is not None
    for line in lines:
        console.print(Text.from_markup(f"  [cyan]•[/] {line}"))


def kv(rows: Sequence[tuple[str, Any]], *, explain: bool = True, check: bool = True) -> None:
    if not enabled(explain):
        if explain:
            for k, v in rows:
                mark = "✓ " if check else "  "
                print(f"  {mark}{k:<18}: {v}")
        return
    assert console is not None
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="odg.ok" if check else "odg.muted", width=2)
    table.add_column(style="odg.key", no_wrap=True)
    table.add_column(style="odg.val")
    for k, v in rows:
        table.add_row("✓" if check else "·", str(k), str(v))
    console.print(table)


def notes(items: Sequence[str], *, explain: bool = True) -> None:
    if not items:
        return
    section("notes", explain=explain)
    bullets(items, explain=explain)


def next_step(text: str, *, explain: bool = True) -> None:
    if not enabled(explain):
        if explain:
            print("\n=== next ===")
            print(f"  {text}")
        return
    assert console is not None
    console.print()
    console.print(
        Panel(
            Text(text, style="odg.next"),
            title="[odg.next]next[/]",
            border_style="magenta",
            box=box.ROUNDED,
            padding=(0, 2),
        )
    )


def json_panel(data: Any, *, title: str = "JSON", explain: bool = True) -> None:
    text = json.dumps(data, indent=2) if not isinstance(data, str) else data
    if not enabled(explain):
        if explain:
            print(f"\n=== {title} ===")
            print(text)
        else:
            print(text)
        return
    assert console is not None
    console.print()
    console.print(
        Panel(
            text,
            title=f"[odg.muted]{title}[/]",
            border_style="dim",
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )


def warn(msg: str, *, explain: bool = True) -> None:
    if not enabled(explain):
        if explain:
            print(f"  Warning: {msg}")
        return
    assert console is not None
    console.print(Text.from_markup(f"[odg.warn]⚠ {msg}[/]"))


def info(msg: str, *, explain: bool = True) -> None:
    if not enabled(explain):
        if explain:
            print(msg)
        return
    assert console is not None
    console.print(Text.from_markup(f"[odg.muted]{msg}[/]"))


def verdict_badge(verdict: str, *, explain: bool = True) -> None:
    if not enabled(explain):
        if explain:
            print(f"  Verdict: {verdict}")
        return
    assert console is not None
    style = {
        "RELEASE": "bold white on green",
        "PROVISIONAL": "bold black on yellow",
        "FAIL": "bold white on red",
    }.get(str(verdict).upper(), "bold white on blue")
    console.print(Text(f"  {verdict}  ", style=style))


def run_summary(summary_text: str, *, explain: bool = True) -> None:
    """Render store.summary() text as a colored step table when possible."""
    if not enabled(explain):
        if explain:
            print("\n" + summary_text)
        return
    assert console is not None
    lines = summary_text.strip().splitlines()
    meta_rows: list[tuple[str, str]] = []
    step_rows: list[tuple[str, str, str]] = []
    in_steps = False
    for line in lines:
        if line.startswith("STEP") or line.startswith("-"):
            in_steps = True
            continue
        if not in_steps and ":" in line:
            k, _, v = line.partition(":")
            meta_rows.append((k.strip(), v.strip()))
        elif in_steps and line.strip():
            parts = line.split()
            if len(parts) >= 2:
                step = parts[0]
                status = parts[1]
                finished = " ".join(parts[2:]) if len(parts) > 2 else "-"
                step_rows.append((step, status, finished))

    console.print()
    meta = Table.grid(padding=(0, 2))
    meta.add_column(style="odg.key", no_wrap=True)
    meta.add_column(style="odg.val")
    for k, v in meta_rows:
        meta.add_row(k, v)

    steps = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold cyan")
    steps.add_column("Step", style="white")
    steps.add_column("Status")
    steps.add_column("Finished", style="odg.muted")
    style_map = {
        "done": "odg.done",
        "failed": "odg.failed",
        "running": "odg.running",
        "pending": "odg.pending",
    }
    for step, status, finished in step_rows:
        steps.add_row(step, Text(status, style=style_map.get(status, "")), finished)

    console.print(
        Panel(
            Group(meta, Text(""), steps),
            title="[odg.brand]run status[/]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )


def runs_table(runs: Sequence[Any]) -> None:
    if not _HAS_RICH or console is None:
        print(f"{'RUN_ID':<42} {'STATUS':<10} MODEL")
        print("-" * 80)
        for m in runs:
            print(f"{m.run_id:<42} {m.status:<10} {m.model_ref}")
        return
    table = Table(
        title="OpenDynamicGGUF runs",
        box=box.ROUNDED,
        header_style="bold cyan",
        border_style="cyan",
    )
    table.add_column("Run ID", style="white")
    table.add_column("Status")
    table.add_column("Model", style="odg.val")
    style_map = {
        "done": "odg.done",
        "failed": "odg.failed",
        "running": "odg.running",
        "pending": "odg.pending",
    }
    for m in runs:
        table.add_row(
            m.run_id,
            Text(m.status, style=style_map.get(m.status, "")),
            m.model_ref,
        )
    console.print(table)

