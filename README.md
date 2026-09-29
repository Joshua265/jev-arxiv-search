# arxiv-walker

Jev-guided best-first search over a local arXiv metadata mirror. arXiv is an ingestion source, not the runtime search engine: the Kaggle metadata snapshot is converted to Parquet + DuckDB, a taxonomy + year-bucket tree is built locally, and a best-first walker uses Jev (via OpenRouter's Decisions API) to decide which branches to explore. BM25 supplies cheap recall within leaf nodes; Jev reranks the finalists.

## Architecture

Kaggle snapshot → Parquet + DuckDB → taxonomy tree (group/category/year-bucket) → best-first walker → Jev branch scoring (noul) → BM25 shortlist → Jev rerank (noul + score) → top 25.

## Prerequisites

- Python 3.11+
- `uv` (recommended) or pip
- An OpenRouter API key (for the `jev` backend) — get one at https://openrouter.ai/keys
- ~6 GB free disk for the Parquet + DuckDB mirror
- (Optional) Nix with flakes for the dev shell

## Installation

### Option A: uv (recommended)

```bash
git clone <repo> && cd arxiv-walker
uv sync
```

### Option B: Nix dev shell

```bash
git clone <repo> && cd arxiv-walker
nix develop        # provides python313, uv, and libstdc++ for duckdb
uv sync
```

The `flake.nix` sets `LD_LIBRARY_PATH` so the DuckDB and PyArrow wheels find `libstdc++.so.6` on NixOS.

## Setup

1. Obtain the arXiv metadata snapshot (`arxiv-metadata-oai-snapshot.json`) from the Kaggle dataset `Cornell-University/arxiv` and place it in the project root (or any path).

2. Configure your API key:
```bash
cp .env.example .env
# edit .env and set OPENROUTER_API_KEY=sk-or-...
```
Or export it inline: `export OPENROUTER_API_KEY=sk-or-...`

3. Ingest the snapshot (streaming, ~3.18M lines, takes a few minutes):
```bash
uv run arxiv-walker ingest --jsonl arxiv-metadata-oai-snapshot.json
```
This writes `data/parquet/papers.parquet`, `data/parquet/paper_categories.parquet`, and `data/arxiv.duckdb`.

4. (Optional) Verify the tree:
```bash
uv run arxiv-walker build-index
```

## Usage

### Search

```bash
# with Jev (requires OPENROUTER_API_KEY)
uv run arxiv-walker search "test-time methods for detecting hallucinations without logits"

# offline / testing (no API calls)
uv run arxiv-walker search "graph neural networks for molecular property prediction" --backend mock

# verbose traversal stats
uv run arxiv-walker search "..." --verbose
```

Output: a numbered list of the top 25 papers with arxiv id, title, category, and submission date.

### Rerank a known set of IDs

```bash
uv run arxiv-walker rerank "scaling laws for language models" --ids 2001.08361 2203.15556 2308.10766
```

## Configuration

Defaults live in `config/default.yaml`. Override via env vars prefixed `ARXIV_WALKER_` or by editing the file. Key settings:

| Key | Default | Description |
|---|---|---|
| `decision.backend` | `jev` | `jev` (OpenRouter) or `mock` (offline) |
| `decision.model` | `typesafe/jev-1.13` | Pinned Jev model |
| `search.beam_width` | `3` | Branches expanded per level |
| `search.max_depth` | `5` | Max tree depth |
| `search.branch_threshold` | `0.35` | Min noul score to expand a branch |
| `search.candidate_cutoff` | `200` | Node is a leaf if paper count <= this |
| `search.bm25_top_k` | `100` | BM25 shortlist per leaf |
| `search.rerank_top_k` | `50` | Papers Jev evaluates in rerank |
| `search.final_top_k` | `25` | Final returned papers |
| `search.exploration_branches` | `3` | Top branches always expanded |

## Development

Lint and test from within `nix develop`:

```bash
nix develop
ruff check src tests          # nix-provided ruff (the uv wheel does not run on NixOS)
uv run pytest -q
```

## Project layout

```
src/arxiv_walker/
  ingest/      JSONL -> Parquet + DuckDB
  index/       taxonomy tree, BM25, metadata store
  decision/    DecisionEngine protocol, Jev/OpenRouter, mock
  search/      best-first walker, branch scoring, rerank
  cli.py       Typer CLI
config/default.yaml
data/          generated (gitignored)
```

## Notes

- arXiv metadata is CC0; link users back to arXiv for full text.
- The walker never calls the arXiv API at runtime; all search is local.
- Jev is accessed via OpenRouter's Decisions API (`POST /api/alpha/decisions`); it returns typed probabilities, not generated text.
