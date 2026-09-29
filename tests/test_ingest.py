from __future__ import annotations

import json

import duckdb
import pyarrow.parquet as pq
import pytest

from arxiv_walker.ingest.kaggle_snapshot import (
    IngestReport,
    ingest_snapshot,
)
from arxiv_walker.ingest.normalize import (
    collapse_ws,
    normalize_record,
    parse_created,
    split_categories,
)

FAKE_RECORDS = [
    {
        "id": "0704.0001",
        "title": "  Calculation of prompt diphoton\n  cross sections  ",
        "abstract": "  A fully differential calculation\nin perturbative QCD is presented.\n",
        "categories": "hep-ph nucl-th",
        "versions": [
            {"version": "v1", "created": "Mon, 2 Apr 2007 19:18:42 GMT"},
            {"version": "v2", "created": "Tue, 24 Jul 2007 20:10:27 GMT"},
        ],
        "update_date": "2008-11-26",
        "doi": "10.1103/PhysRevD.76.013009",
        "journal-ref": "Phys.Rev.D76:013009,2007",
        "comments": "37 pages, 15 figures",
        "license": None,
        "authors": "C. Balázs, E. L. Berger",
        "authors_parsed": [
            ["Balázs", "C.", ""],
            ["Berger", "E. L.", ""],
        ],
    },
    {
        "id": "0704.0002",
        "title": "  Second paper title  ",
        "abstract": "  Second paper abstract.\n",
        "categories": "astro-ph",
        "versions": [
            {"version": "v1", "created": "Tue, 3 Apr 2007 10:00:00 GMT"},
        ],
        "update_date": "2007-04-03",
        "doi": "",
        "journal-ref": "",
        "comments": "",
        "license": "",
        "authors": "Jane Doe",
        "authors_parsed": None,
    },
    {
        "id": "0704.0003",
        "title": "  Third paper  ",
        "abstract": "  Third abstract.\n",
        "categories": "math.GT math.QA",
        "versions": [],
        "update_date": "2007-04-04",
        "doi": "",
        "journal-ref": "",
        "comments": "",
        "license": "",
        "authors": "John Smith",
        "authors_parsed": None,
    },
]


class TestNormalize:
    def test_collapse_ws(self) -> None:
        assert collapse_ws("  hello\n  world  ") == "hello world"
        assert collapse_ws(None) == ""
        assert collapse_ws("  ") == ""

    def test_split_categories(self) -> None:
        assert split_categories("hep-ph nucl-th") == ["hep-ph", "nucl-th"]
        assert split_categories(None) == []
        assert split_categories("") == []

    def test_parse_created(self) -> None:
        result = parse_created("Mon, 2 Apr 2007 19:18:42 GMT")
        assert "2007-04-02" in result
        assert "19:18:42" in result
        assert parse_created("invalid") == ""

    def test_normalize_record_valid(self) -> None:
        record = normalize_record(FAKE_RECORDS[0])
        assert record is not None
        assert record["arxiv_id"] == "0704.0001"
        assert record["title"] == "Calculation of prompt diphoton cross sections"
        assert record["primary_category"] == "hep-ph"
        assert "2007-04-02" in record["submitted_at"]
        assert record["updated_at"] == "2008-11-26"
        assert record["doi"] == "10.1103/PhysRevD.76.013009"
        assert record["journal_ref"] == "Phys.Rev.D76:013009,2007"
        authors = json.loads(record["authors_json"])
        assert authors[0][0] == "Balázs"

    def test_normalize_record_no_versions(self) -> None:
        record = normalize_record(FAKE_RECORDS[2])
        assert record is not None
        assert record["submitted_at"] == ""
        assert record["primary_category"] == "math.GT"

    def test_normalize_record_missing_id(self) -> None:
        assert normalize_record({"title": "x", "abstract": "y"}) is None

    def test_normalize_record_missing_title(self) -> None:
        assert normalize_record({"id": "x", "abstract": "y"}) is None

    def test_normalize_record_missing_abstract(self) -> None:
        assert normalize_record({"id": "x", "title": "y"}) is None


class TestIngestSnapshot:
    def test_ingest_snapshot(self, tmp_path: pytest.TempPathFactory) -> None:
        jsonl_path = tmp_path / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for rec in FAKE_RECORDS:
                f.write(json.dumps(rec) + "\n")

        data_dir = str(tmp_path / "data")
        report = ingest_snapshot(
            jsonl_path=str(jsonl_path),
            data_dir=data_dir,
            batch_size=2,
            progress=False,
        )

        assert isinstance(report, IngestReport)
        assert report.papers_written == 3
        assert report.categories_written == 5  # hep-ph, nucl-th, astro-ph, math.GT, math.QA
        assert report.skipped == 0

        parquet_path = tmp_path / "data" / "parquet" / "papers.parquet"
        assert parquet_path.exists()

        table = pq.read_table(str(parquet_path))
        assert table.num_rows == 3
        assert table.column("arxiv_id").to_pylist() == ["0704.0001", "0704.0002", "0704.0003"]

        cat_path = tmp_path / "data" / "parquet" / "paper_categories.parquet"
        cat_table = pq.read_table(str(cat_path))
        assert cat_table.num_rows == 5

        db_path = tmp_path / "data" / "arxiv.duckdb"
        assert db_path.exists()

        con = duckdb.connect(str(db_path))
        try:
            row = con.execute("SELECT COUNT(*) FROM papers").fetchone()
            assert row[0] == 3

            row = con.execute("SELECT COUNT(*) FROM paper_categories").fetchone()
            assert row[0] == 5

            stats = con.execute(
                "SELECT category, paper_count FROM category_stats ORDER BY category"
            ).fetchall()
            categories_found = {r[0]: r[1] for r in stats}
            assert categories_found["hep-ph"] == 1
            assert categories_found["nucl-th"] == 1
            assert categories_found["astro-ph"] == 1
            assert categories_found["math.GT"] == 1
            assert categories_found["math.QA"] == 1
        finally:
            con.close()
