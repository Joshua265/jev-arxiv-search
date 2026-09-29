from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

from arxiv_walker.ingest.normalize import normalize_record


@dataclass
class IngestReport:
    papers_written: int = 0
    categories_written: int = 0
    skipped: int = 0
    parquet_path: str = ""
    db_path: str = ""


PAPERS_SCHEMA = pa.schema([
    pa.field("arxiv_id", pa.string()),
    pa.field("title", pa.string()),
    pa.field("abstract", pa.string()),
    pa.field("primary_category", pa.string()),
    pa.field("submitted_at", pa.string()),
    pa.field("updated_at", pa.string()),
    pa.field("doi", pa.string()),
    pa.field("journal_ref", pa.string()),
    pa.field("comments", pa.string()),
    pa.field("license", pa.string()),
    pa.field("authors_json", pa.string()),
    pa.field("versions_json", pa.string()),
])

PAPER_CATEGORIES_SCHEMA = pa.schema([
    pa.field("arxiv_id", pa.string()),
    pa.field("category", pa.string()),
])


def count_lines(path: str) -> int:
    result = subprocess.run(
        ["wc", "-l", path],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(result.stdout.strip().split()[0])


def _write_batch(
    papers_writer: pq.ParquetWriter,
    categories_writer: pq.ParquetWriter,
    papers_batch: list[dict[str, Any]],
    categories_batch: list[dict[str, str]],
) -> None:
    papers_table = pa.Table.from_pylist(papers_batch, schema=PAPERS_SCHEMA)
    papers_writer.write_table(papers_table)
    if categories_batch:
        cat_table = pa.Table.from_pylist(categories_batch, schema=PAPER_CATEGORIES_SCHEMA)
        categories_writer.write_table(cat_table)


def ingest_snapshot(
    jsonl_path: str,
    data_dir: str,
    batch_size: int = 50000,
    progress: bool = True,
) -> IngestReport:
    data_root = Path(data_dir)
    parquet_dir = data_root / "parquet"
    index_dir = data_root / "indexes"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    papers_path = str(parquet_dir / "papers.parquet")
    categories_path = str(parquet_dir / "paper_categories.parquet")
    db_path = str(data_root / "arxiv.duckdb")

    total = count_lines(jsonl_path) if progress else 0

    report = IngestReport(parquet_path=papers_path, db_path=db_path)

    papers_batch: list[dict[str, Any]] = []
    categories_batch: list[dict[str, str]] = []

    papers_writer = pq.ParquetWriter(papers_path, PAPERS_SCHEMA)
    categories_writer = pq.ParquetWriter(categories_path, PAPER_CATEGORIES_SCHEMA)

    try:
        with open(jsonl_path) as f:
            iterator = f
            if progress:
                iterator = tqdm.tqdm(f, total=total, unit="lines", desc="Ingesting")

            for line in iterator:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    report.skipped += 1
                    continue

                record = normalize_record(raw)
                if record is None:
                    report.skipped += 1
                    continue

                papers_batch.append(record)
                for cat in record["categories"]:
                    categories_batch.append({"arxiv_id": record["arxiv_id"], "category": cat})

                if len(papers_batch) >= batch_size:
                    _write_batch(papers_writer, categories_writer, papers_batch, categories_batch)
                    report.papers_written += len(papers_batch)
                    report.categories_written += len(categories_batch)
                    papers_batch.clear()
                    categories_batch.clear()

        if papers_batch:
            _write_batch(papers_writer, categories_writer, papers_batch, categories_batch)
            report.papers_written += len(papers_batch)
            report.categories_written += len(categories_batch)
    finally:
        papers_writer.close()
        categories_writer.close()

    _build_duckdb(db_path, parquet_dir)
    return report


def _build_duckdb(db_path: str, parquet_dir: Path) -> None:
    import duckdb

    papers_parquet = str(parquet_dir / "papers.parquet")
    categories_parquet = str(parquet_dir / "paper_categories.parquet")

    con = duckdb.connect(db_path)
    try:
        con.execute(f"""
            CREATE OR REPLACE TABLE papers AS
            SELECT * FROM read_parquet('{papers_parquet}')
        """)
        con.execute(f"""
            CREATE OR REPLACE TABLE paper_categories AS
            SELECT * FROM read_parquet('{categories_parquet}')
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_papers_id ON papers(arxiv_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_pc_id ON paper_categories(arxiv_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_pc_cat ON paper_categories(category)")

        con.execute("""
            CREATE OR REPLACE TABLE category_stats AS
            SELECT
                pc.category,
                COUNT(*) AS paper_count,
                MIN(p.submitted_at) AS min_date,
                MAX(p.submitted_at) AS max_date,
                '[]' AS top_terms_json
            FROM paper_categories pc
            JOIN papers p ON p.arxiv_id = pc.arxiv_id
            GROUP BY pc.category
        """)
    finally:
        con.close()
