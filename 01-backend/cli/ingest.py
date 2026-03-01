"""CLI entry point for tree ingestion.

Usage:
    python -m cli.ingest \
        --source "03-data/vault/books/raw/Database Internals.pdf" \
        --output-dir "03-data/vault/books/processed" \
        --provider openai \
        --model gpt-4o \
        --verbose
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.status import Status
from rich.style import Style
from rich.table import Table
from rich.text import Text
from typing_extensions import Annotated

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.tree_engine import TreeEngine
from services.llm_factory import LLMConfig, LLMProvider

app = typer.Typer(
    name="att-ingest",
    help="Ingest CS textbooks into hierarchical Knowledge Trees",
    rich_markup_mode="rich",
)
console = Console()


def setup_logging(verbose: bool) -> None:
    """Configure Loguru logging."""
    logger.remove()  # Remove default handler

    level = "DEBUG" if verbose else "INFO"
    format_str = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    logger.add(sys.stderr, level=level, format=format_str, colorize=True)


class ProgressManager:
    """Manages rich progress display for ingestion pipeline."""

    def __init__(self):
        self.progress = Progress(
            SpinnerColumn(spinner_name="dots", style="cyan"),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=40, complete_style="green", finished_style="green"),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        )
        self.tasks = {}

    def __enter__(self):
        self.progress.start()
        return self

    def __exit__(self, *args):
        self.progress.stop()

    def add_task(self, name: str, total: int) -> int:
        """Add a new progress task."""
        task_id = self.progress.add_task(name, total=total)
        self.tasks[name] = task_id
        return task_id

    def update(self, name: str, advance: int = 1, description: str | None = None):
        """Update task progress."""
        if name in self.tasks:
            kwargs = {"advance": advance}
            if description:
                kwargs["description"] = description
            self.progress.update(self.tasks[name], **kwargs)

    def complete(self, name: str):
        """Mark task as complete."""
        if name in self.tasks:
            self.progress.update(self.tasks[name], completed=True)


@app.command()
def ingest(
    source: Annotated[
        Path,
        typer.Option(
            "--source",
            "-s",
            help="Path to PDF file to ingest",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Output directory for processed books",
        ),
    ] = Path("03-data/vault/books/processed"),
    provider: Annotated[
        LLMProvider,
        typer.Option(
            "--provider",
            "-p",
            help="LLM provider to use (defaults to Ollama for local-first)",
            case_sensitive=False,
        ),
    ] = LLMProvider.OLLAMA,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="Model name (e.g., 'gpt-4o', 'llama3:8b')",
        ),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option(
            "--api-key",
            "-k",
            help="API key (or use env var: OPENAI_API_KEY, ANTHROPIC_API_KEY)",
        ),
    ] = None,
    api_base: Annotated[
        str | None,
        typer.Option(
            "--api-base",
            "-b",
            help="Base URL for API (for Ollama: http://localhost:11434/v1)",
        ),
    ] = None,
    max_chapters: Annotated[
        int | None,
        typer.Option(
            "--max-chapters",
            "-n",
            help="Limit to first N chapters (for testing)",
            min=1,
        ),
    ] = None,
    skip_exercises: Annotated[
        bool,
        typer.Option(
            "--skip-exercises",
            help="Skip exercise generation (Pass 3)",
        ),
    ] = False,
    temperature: Annotated[
        float,
        typer.Option(
            "--temperature",
            "-t",
            help="LLM temperature (0-2)",
            min=0.0,
            max=2.0,
        ),
    ] = 0.1,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose debug output",
        ),
    ] = False,
) -> None:
    """Ingest a textbook PDF into hierarchical Knowledge Trees.

    Examples:
        # Default: Use Ollama (llama3:8b) - recommended for local processing
        python -m cli.ingest --source book.pdf

        # Use specific Ollama model
        python -m cli.ingest --source book.pdf --model llama3.1:8b

        # Use OpenAI
        python -m cli.ingest --source book.pdf --provider openai

        # Process only first 3 chapters
        python -m cli.ingest --source book.pdf --max-chapters 3

        # With verbose debug output
        python -m cli.ingest --source book.pdf --verbose
    """
    setup_logging(verbose)

    # Resolve paths relative to workspace root
    workspace_root = Path(__file__).parent.parent.parent

    if not source.is_absolute():
        source = workspace_root / source

    if not output_dir.is_absolute():
        output_dir = workspace_root / output_dir

    # Set default models based on provider
    if model is None:
        model_defaults = {
            LLMProvider.OPENAI: "gpt-4o",
            LLMProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
            LLMProvider.OLLAMA: "llama3:8b",
        }
        model = model_defaults.get(provider, "llama3:8b")

    # Check API key for non-Ollama providers
    if provider != LLMProvider.OLLAMA and api_key is None:
        env_var = f"{provider.value.upper()}_API_KEY"
        if not os.environ.get(env_var):
            console.print(
                Panel(
                    f"[bold red]API Key Required[/bold red]\n\n"
                    f"Set [cyan]{env_var}[/cyan] environment variable\n"
                    f"or use [cyan]--api-key[/cyan] flag",
                    border_style="red",
                )
            )
            raise typer.Exit(1)

    # Check Ollama availability
    if provider == LLMProvider.OLLAMA:
        import urllib.request

        try:
            req = urllib.request.Request("http://localhost:11434/api/tags")
            with urllib.request.urlopen(req, timeout=2) as resp:
                pass  # Ollama is running
        except Exception:
            console.print(
                Panel(
                    "[bold red]Ollama Not Available[/bold red]\n\n"
                    "Make sure Ollama is running:\n"
                    "  [cyan]ollama serve[/cyan]\n\n"
                    "And the model is pulled:\n"
                    f"  [cyan]ollama pull {model}[/cyan]",
                    border_style="red",
                )
            )
            raise typer.Exit(1)

    # Print header
    console.print()
    console.print(
        Panel(
            f"[bold cyan]ATT Tree Decomposer[/bold cyan]\n"
            f"Converting textbooks into interactive knowledge graphs",
            border_style="cyan",
        )
    )
    console.print()

    # Configuration table
    config_table = Table(show_header=False, box=None, padding=(0, 2))
    config_table.add_row("[dim]Source:[/dim]", f"[white]{source.name}[/white]")
    config_table.add_row("[dim]Output:[/dim]", f"[white]{output_dir}[/white]")
    config_table.add_row(
        "[dim]Provider:[/dim]",
        f"[green]{provider.value}[/green]" if provider == LLMProvider.OLLAMA else f"[blue]{provider.value}[/blue]",
    )
    config_table.add_row("[dim]Model:[/dim]", f"[white]{model}[/white]")
    config_table.add_row("[dim]Max Chapters:[/dim]", f"[white]{max_chapters or 'all'}[/white]")
    config_table.add_row("[dim]Skip Exercises:[/dim]", f"[white]{'yes' if skip_exercises else 'no'}[/white]")
    console.print(config_table)
    console.print()

    # Create LLM config
    llm_config = LLMConfig(
        provider=provider,
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=api_base,
    )

    # Initialize engine
    engine = TreeEngine(
        llm_config=llm_config,
        output_dir=output_dir,
        verbose=verbose,
    )

    # Run pipeline with progress display
    try:
        output_files = asyncio.run(
            _run_with_progress(
                engine=engine,
                source=source,
                max_chapters=max_chapters,
                skip_exercises=skip_exercises,
            )
        )

        # Success output
        console.print()
        console.print(
            Panel(
                f"[bold green]✓ Processing Complete[/bold green]\n\n"
                f"[white]{len(output_files)} chapters written[/white]",
                border_style="green",
            )
        )

        for f in output_files:
            console.print(f"  [dim]→[/dim] [cyan]{f}[/cyan]")

        # Next steps hint
        console.print()
        console.print(
            "[dim]Next steps:[/dim]\n"
            "  [cyan]1.[/cyan] Validate: [white]python -m cli.ingest validate[/white]\n"
            "  [cyan]2.[/cyan] Visualize: [white]python -m cli.ingest visualize[/white]\n"
            "  [cyan]3.[/cyan] List chapters: [white]python -m cli.ingest list-chapters[/white]"
        )
        console.print()

    except Exception as e:
        console.print()
        console.print(
            Panel(
                f"[bold red]✗ Processing Failed[/bold red]\n\n[red]{e}[/red]",
                border_style="red",
            )
        )
        if verbose:
            import traceback
            console.print("\n[dim]Traceback:[/dim]")
            console.print(traceback.format_exc())
        raise typer.Exit(1)


async def _run_with_progress(
    engine: TreeEngine,
    source: Path,
    max_chapters: int | None,
    skip_exercises: bool,
) -> list[Path]:
    """Run the engine with rich progress display."""

    with ProgressManager() as pm:
        # Phase 0: PDF -> Markdown
        doc_task = pm.add_task("[yellow]Phase 0: PDF → Markdown[/yellow]", total=1)

        with Status("[cyan]Converting PDF to Markdown with Marker...[/cyan]", console=console):
            pages_meta, full_md = engine.doc_processor.pdf_to_markdown(source)
            total_pages = pages_meta.get("page_count", 0)
            book_slug = engine._slugify(source.stem)
            md_path = engine.output_dir / book_slug / f"{book_slug}.md"

        pm.update("[yellow]Phase 0: PDF → Markdown[/yellow]", advance=1)
        pm.complete("[yellow]Phase 0: PDF → Markdown[/yellow]")
        console.print(f"  [dim]→ {total_pages} pages → {len(full_md)} chars Markdown[/dim]")

        # Phase 1: Book Structure
        struct_task = pm.add_task("[yellow]Phase 1: Extracting book structure[/yellow]", total=1)

        with Status("[cyan]Analyzing table of contents...[/cyan]", console=console):
            book_structure = await engine.structure_agent.extract(document)

        pm.update("[yellow]Phase 1: Extracting book structure[/yellow]", advance=1)
        pm.complete("[yellow]Phase 1: Extracting book structure[/yellow]")
        console.print(
            f"  [dim]→ '{book_structure.title}' by {book_structure.author}, "
            f"{len(book_structure.chapters)} chapters[/dim]"
        )

        # Limit chapters if specified
        chapters = book_structure.chapters
        if max_chapters:
            chapters = chapters[:max_chapters]

        # Phase 1.5: Split Markdown by chapters
        split_task = pm.add_task("[yellow]Phase 1.5: Splitting Markdown[/yellow]", total=1)
        with Status("[cyan]Splitting Markdown by chapters...[/cyan]", console=console):
            chapter_mds = engine.split_markdown_by_chapters(full_md, book_structure, pages_meta)
        pm.update("[yellow]Phase 1.5: Splitting Markdown[/yellow]", advance=1)
        pm.complete("[yellow]Phase 1.5: Splitting Markdown[/yellow]")

        # Phase 2: Chapter Decomposition
        chap_task = pm.add_task(
            "[yellow]Phase 2: Decomposing chapters[/yellow]",
            total=len(chapters),
        )

        output_files: list[Path] = []

        for idx, chapter_info in enumerate(chapters):
            chapter_num = idx + 1
            pm.update(
                "[yellow]Phase 2: Decomposing chapters[/yellow]",
                description=f"[yellow]Phase 2: Decomposing chapters[/yellow] [dim]({chapter_info.id})[/dim]",
            )

            try:
                chapter_md = chapter_mds.get(chapter_info.id, "")
                if not chapter_md:
                    console.print(f"  [yellow]⚠ No Markdown for {chapter_info.id}, skipping[/yellow]")
                    pm.update("[yellow]Phase 2: Decomposing chapters[/yellow]", advance=1)
                    continue

                # Save chapter Markdown
                chapter_md_path = (
                    engine.output_dir / book_slug / "chapters" / f"{chapter_info.id}.md"
                )
                chapter_md_path.parent.mkdir(parents=True, exist_ok=True)
                with open(chapter_md_path, "w", encoding="utf-8") as f:
                    f.write(chapter_md)

                # Decompose chapter
                chapter_structure = await engine.decomposition_agent.decompose(
                    chapter_md=chapter_md,
                    chapter_info=chapter_info,
                    book_context={
                        "title": book_structure.title,
                        "author": book_structure.author,
                    },
                )

                # Build tree
                tree_builder = engine.tree_builder_class(book_slug, verbose=engine.verbose)
                nodes = tree_builder.build_chapter_tree(chapter_structure, chapter_info)

                # Track nodes for prerequisite linking
                engine.all_nodes.extend(nodes)
                nodes = tree_builder.link_prerequisites(engine.all_nodes, chapter_num)

                # Phase 3: Exercise Generation (if enabled)
                if not skip_exercises:
                    concept_count = sum(1 for n in nodes if n.level == 3)  # CONCEPT level
                    ex_task_name = f"[yellow]Phase 3: Chapter {chapter_num} exercises[/yellow]"
                    ex_task = pm.add_task(ex_task_name, total=concept_count)

                    nodes = await _generate_exercises_with_progress(
                        engine, nodes, chapter_md, pm, ex_task_name
                    )

                    pm.complete(ex_task_name)

                # Write chapter JSON
                writer = engine.writer_class(engine.output_dir, verbose=engine.verbose)
                file_path = writer.write_chapter(nodes, chapter_info, book_structure)
                output_files.append(file_path)

                pm.update("[yellow]Phase 2: Decomposing chapters[/yellow]", advance=1)

            except Exception as e:
                console.print(f"  [red]✗ Chapter {chapter_info.id} failed: {e}[/red]")
                continue

        pm.complete("[yellow]Phase 2: Decomposing chapters[/yellow]")
        return output_files


async def _generate_exercises_with_progress(
    engine: TreeEngine,
    nodes: list,
    chapter_context: str,
    pm: ProgressManager,
    task_name: str,
) -> list:
    """Generate exercises with progress tracking."""
    from core.node_schema import ConceptNode, NodeLevel

    concept_nodes = [n for n in nodes if n.level == NodeLevel.CONCEPT]
    prereq_map = {n.id: n for n in engine.all_nodes}

    semaphore = asyncio.Semaphore(3)  # Max 3 concurrent

    async def gen_with_progress(node):
        async with semaphore:
            concept = ConceptNode(
                id=node.id.split("-")[-1],
                title=node.breadcrumb.split(">")[-1].strip(),
                content=node.content,
                key_terms=node.metadata.key_terms,
                complexity=node.metadata.complexity,
                prerequisites=node.metadata.prerequisites,
                source_page_start=node.source_page_start,
                source_page_end=node.source_page_end,
            )

            prereq_nodes = [
                prereq_map[p] for p in node.metadata.prerequisites if p in prereq_map
            ]

            try:
                exercise = await engine.exercise_agent.generate(
                    concept, chapter_context, prereq_nodes
                )
                node.exercise_config = exercise
            except Exception:
                pass  # Fallback already in agent

            pm.update(task_name, advance=1)
            return node

    await asyncio.gather(*[gen_with_progress(n) for n in concept_nodes])
    return nodes


@app.command()
def validate(
    chapter_file: Annotated[
        Path,
        typer.Argument(
            help="Path to chapter JSON file to validate",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = False,
) -> None:
    """Validate a processed chapter file against the schema."""
    import json

    from core.node_schema import ChapterManifest, KnowledgeNode

    setup_logging(verbose)

    console.print(f"\n[dim]Validating:[/dim] [cyan]{chapter_file}[/cyan]\n")

    try:
        with open(chapter_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        with Status("[cyan]Validating manifest...[/cyan]", console=console):
            manifest = ChapterManifest.model_validate(data)

        console.print(f"  [green]✓[/green] Manifest valid: [white]{manifest.chapter_title}[/white]")
        console.print(f"  [dim]  Total nodes: {manifest.total_nodes}[/dim]")
        console.print(f"  [dim]  Leaf nodes: {manifest.leaf_nodes}[/dim]")

        with Status("[cyan]Validating nodes...[/cyan]", console=console):
            for node in manifest.nodes:
                KnowledgeNode.model_validate(node.model_dump())

        console.print(f"  [green]✓[/green] All {len(manifest.nodes)} nodes valid\n")

    except Exception as e:
        console.print(f"  [red]✗ Validation failed: {e}[/red]\n")
        raise typer.Exit(1)


@app.command()
def list_chapters(
    book_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory containing processed chapter files",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
    ],
) -> None:
    """List all chapters in a processed book."""
    import json

    setup_logging(False)

    chapter_files = sorted(book_dir.glob("chapter-*.json"))

    if not chapter_files:
        # Try subdirectories
        chapter_files = sorted(book_dir.rglob("chapter-*.json"))

    if not chapter_files:
        console.print(f"\n[yellow]No chapter files found in {book_dir}[/yellow]\n")
        return

    console.print(f"\n[bold cyan]Chapters in {book_dir.name}:[/bold cyan]\n")

    table = Table(
        "#",
        "Title",
        "Nodes",
        "Exercises",
        box=None,
        padding=(0, 2),
    )
    table.columns[0].style = "dim"
    table.columns[2].justify = "right"
    table.columns[3].justify = "right"

    for cf in chapter_files:
        with open(cf, "r", encoding="utf-8") as f:
            data = json.load(f)

        chapter_num = data.get("chapter_number", "?")
        title = data.get("chapter_title", "Unknown")
        nodes = data.get("total_nodes", 0)
        exercises = data.get("leaf_nodes", 0)

        table.add_row(
            str(chapter_num),
            f"[white]{title}[/white]",
            f"[dim]{nodes}[/dim]",
            f"[green]{exercises}[/green]" if exercises > 0 else "[dim]0[/dim]",
        )

    console.print(table)
    console.print()


@app.command()
def visualize(
    book_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory containing processed chapter files",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
    ] = Path("03-data/vault/books/processed"),
    no_open: Annotated[
        bool,
        typer.Option(
            "--no-open",
            help="Don't open browser automatically",
        ),
    ] = False,
) -> None:
    """Launch interactive knowledge tree visualizer in browser.

    Opens the D3.js visualization with all chapters loaded.
    Click nodes to view details, exercises, and navigate prerequisites.
    """
    import webbrowser

    setup_logging(False)

    # Find visualizer HTML
    workspace_root = Path(__file__).parent.parent.parent
    viz_path = workspace_root / "02-frontend/public/tree-visualizer.html"

    if not viz_path.exists():
        console.print(f"\n[red]Visualizer not found: {viz_path}[/red]\n")
        raise typer.Exit(1)

    # Find chapter files
    chapter_files = sorted(book_dir.glob("chapter-*.json"))
    if not chapter_files:
        chapter_files = sorted(book_dir.rglob("chapter-*.json"))

    console.print()
    if chapter_files:
        console.print(f"[green]✓[/green] Found [white]{len(chapter_files)}[/white] chapter files")
        for cf in chapter_files[:5]:
            console.print(f"  [dim]→ {cf.name}[/dim]")
        if len(chapter_files) > 5:
            console.print(f"  [dim]... and {len(chapter_files) - 5} more[/dim]")
    else:
        console.print(f"[yellow]![/yellow] No chapter files found - visualizer will use demo data")
        console.print(f"  [dim]Load your own JSON files via the UI[/dim]")

    # Open in browser
    viz_url = f"file://{viz_path.absolute()}"

    if not no_open:
        console.print(f"\n[cyan]→ Opening visualizer...[/cyan]\n")
        webbrowser.open(viz_url)
    else:
        console.print(f"\n[dim]Visualizer URL:[/dim] {viz_url}\n")


@app.command()
def convert(
    source: Annotated[
        Path,
        typer.Argument(
            help="Path to PDF file to convert",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output Markdown file path (default: {book-slug}.md)",
        ),
    ] = None,
    split_chapters: Annotated[
        bool,
        typer.Option(
            "--split-chapters",
            "-s",
            help="Also split Markdown into chapter files",
        ),
    ] = False,
) -> None:
    """Convert PDF to Markdown only (no LLM processing).

    Useful for reviewing the Markdown before running the full tree pipeline.

    Examples:
        # Convert to single Markdown file
        python -m cli.ingest convert book.pdf

        # Convert and split by chapters
        python -m cli.ingest convert book.pdf --split-chapters

        # Custom output path
        python -m cli.ingest convert book.pdf -o output/my-book.md
    """
    setup_logging(False)

    workspace_root = Path(__file__).parent.parent.parent

    console.print()
    console.print(
        Panel(
            f"[bold cyan]PDF → Markdown Conversion[/bold cyan]",
            border_style="cyan",
        )
    )
    console.print()

    with Status("[cyan]Converting PDF with Marker...[/cyan]", console=console):
        from core.tree_engine import DocumentProcessor

        processor = DocumentProcessor(verbose=False)

        if output:
            output_path = output
        else:
            book_slug = source.stem.lower().replace(" ", "-")
            output_path = Path(f"{book_slug}.md")

        pages_meta, md_content = processor.pdf_to_markdown(source, output_path)

    console.print(f"[green]✓[/green] Converted [white]{len(document.pages)}[/white] pages")
    console.print(f"[green]✓[/green] Markdown: [cyan]{output_path}[/cyan] ([white]{len(md_content)}[/white] chars)")

    if split_chapters:
        console.print()
        console.print("[cyan]Splitting by chapters...[/cyan]")

        # Need to get book structure first
        from services.llm_factory import LLMFactory
        from core.tree_engine import BookStructureExtractionAgent

        llm_client = LLMFactory.from_env()
        structure_agent = BookStructureExtractionAgent(llm_client)

        import asyncio

        pages_meta = {"page_count": len(md_content) // 2000}  # Rough estimate
        book_structure = asyncio.run(
            structure_agent.extract_from_markdown(pages_meta, md_content)
        )

        # Split markdown (using TreeEngine's split method)
        from core.tree_engine import TreeEngine
        engine_for_split = TreeEngine()
        chapter_mds = engine_for_split.split_markdown_by_chapters(
            md_content, book_structure, pages_meta
        )

        chapters_dir = output_path.parent / f"{output_path.stem}-chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)

        for ch_id, ch_md in chapter_mds.items():
            ch_path = chapters_dir / f"{ch_id}.md"
            with open(ch_path, "w", encoding="utf-8") as f:
                f.write(ch_md)

        console.print(f"[green]✓[/green] Split into [white]{len(chapter_mds)}[/white] chapter files")
        console.print(f"  [dim]→ {chapters_dir}[/dim]")

    console.print()
    console.print("Next steps:")
    console.print("  [cyan]1.[/cyan] Review/edit the Markdown file")
    console.print("  [cyan]2.[/cyan] Run full pipeline: [white]python -m cli.ingest --source book.pdf[/white]")
    console.print()


if __name__ == "__main__":
    app()
