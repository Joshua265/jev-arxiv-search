from __future__ import annotations

import asyncio
import os
from collections import Counter

import typer

from arxiv_walker.config import load_settings
from arxiv_walker.decision.base import DecisionError
from arxiv_walker.decision.factory import make_decision_engine
from arxiv_walker.index.metadata import MetadataStore
from arxiv_walker.index.taxonomy import build_tree
from arxiv_walker.ingest.kaggle_snapshot import ingest_snapshot
from arxiv_walker.search.rerank import rerank_papers
from arxiv_walker.search.walker import Walker

app = typer.Typer(no_args_is_help=True)


@app.command()
def ingest(
    jsonl: str = typer.Option(..., "--jsonl", help="Path to arxiv-metadata-oai-snapshot.json"),
    data_dir: str | None = typer.Option(None, "--data-dir", help="Data directory"),
    batch_size: int | None = typer.Option(None, "--batch-size", help="Batch size for ingestion"),
    no_progress: bool = typer.Option(False, "--no-progress", help="Disable progress bar"),
) -> None:
    settings = load_settings()
    data_dir_val = data_dir or settings.data_dir
    batch_size_val = batch_size or settings.ingest.batch_size

    report = ingest_snapshot(
        jsonl_path=jsonl,
        data_dir=data_dir_val,
        batch_size=batch_size_val,
        progress=not no_progress,
    )

    typer.echo(f"Papers written: {report.papers_written}")
    typer.echo(f"Categories written: {report.categories_written}")
    typer.echo(f"Skipped: {report.skipped}")
    typer.echo(f"Parquet path: {report.parquet_path}")
    typer.echo(f"DuckDB path: {report.db_path}")


@app.command(name="build-index")
def build_index(
    data_dir: str | None = typer.Option(None, "--data-dir", help="Data directory"),
) -> None:
    settings = load_settings()
    data_dir_val = data_dir or settings.data_dir
    db_path = f"{data_dir_val}/arxiv.duckdb"

    nodes = build_tree(db_path, settings)

    typer.echo(f"Total nodes: {len(nodes)}")

    depth_counts: Counter[int] = Counter()
    for node in nodes:
        depth_counts[node.depth] += 1

    typer.echo("\nBreakdown by depth:")
    for depth in sorted(depth_counts):
        typer.echo(f"  depth {depth}: {depth_counts[depth]} nodes")

    sorted_nodes = sorted(nodes, key=lambda n: n.count, reverse=True)
    typer.echo("\nTop 10 largest nodes by paper count:")
    for node in sorted_nodes[:10]:
        typer.echo(f"  {node.id}: {node.count} papers")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    data_dir: str | None = typer.Option(None, "--data-dir", help="Data directory"),
    backend: str | None = typer.Option(None, "--backend", help="Decision backend (jev or mock)"),
    top_k: int | None = typer.Option(None, "--top-k", help="Number of final results"),
    verbose: bool = typer.Option(False, "--verbose", help="Show traversal stats"),
) -> None:
    settings = load_settings()
    data_dir_val = data_dir or settings.data_dir
    db_path = f"{data_dir_val}/arxiv.duckdb"

    if backend is not None:
        settings.decision.backend = backend

    if top_k is not None:
        settings.search.final_top_k = top_k

    if settings.decision.backend == "jev" and not os.environ.get("OPENROUTER_API_KEY"):
        typer.echo(
            "Error: OPENROUTER_API_KEY is not set. "
            "Copy .env.example to .env and add your key, or use --backend mock.",
            err=True,
        )
        raise typer.Exit(1)

    try:
        engine = make_decision_engine(settings.decision)
    except DecisionError as e:
        typer.echo(f"Error creating decision engine: {e}", err=True)
        raise typer.Exit(1)

    walker = Walker(db_path, settings, engine)
    result = asyncio.run(walker.search(query))

    typer.echo(f"\nResults ({len(result.final_papers)} papers):")
    for i, paper in enumerate(result.final_papers, 1):
        score_str = f" (score: {paper.get('score', 'N/A'):.4f})" if "score" in paper else ""
        typer.echo(
            f"  {i}. [{paper['arxiv_id']}] {paper['title']}\n"
            f"     Category: {paper['primary_category']}  "
            f"Submitted: {paper['submitted_at']}{score_str}"
        )

    if verbose:
        typer.echo("\nTraversal stats:")
        typer.echo(f"  Nodes visited: {result.nodes_visited}")
        typer.echo(f"  Decision calls: {result.decision_calls}")
        typer.echo(f"  Branches scored: {result.branches_scored}")
        typer.echo(f"  Total candidates: {len(result.candidates)}")


@app.command()
def rerank(
    query: str = typer.Argument(..., help="Search query for reranking context"),
    ids: list[str] = typer.Option(..., "--ids", help="ArXiv IDs to rerank"),
    data_dir: str | None = typer.Option(None, "--data-dir", help="Data directory"),
    backend: str | None = typer.Option(None, "--backend", help="Decision backend (jev or mock)"),
) -> None:
    settings = load_settings()
    data_dir_val = data_dir or settings.data_dir
    db_path = f"{data_dir_val}/arxiv.duckdb"

    if backend is not None:
        settings.decision.backend = backend

    if settings.decision.backend == "jev" and not os.environ.get("OPENROUTER_API_KEY"):
        typer.echo(
            "Error: OPENROUTER_API_KEY is not set. "
            "Copy .env.example to .env and add your key, or use --backend mock.",
            err=True,
        )
        raise typer.Exit(1)

    try:
        engine = make_decision_engine(settings.decision)
    except DecisionError as e:
        typer.echo(f"Error creating decision engine: {e}", err=True)
        raise typer.Exit(1)

    meta = MetadataStore(db_path)
    papers = meta.get_papers(ids)

    if not papers:
        typer.echo("No papers found for the given IDs.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Reranking {len(papers)} papers...")
    reranked = asyncio.run(rerank_papers(engine, query, papers))

    typer.echo("\nReranked results:")
    for paper, score in reranked:
        typer.echo(
            f"  [{paper['arxiv_id']}] {paper['title']}  "
            f"(score: {score:.4f})"
        )


if __name__ == "__main__":
    app()
