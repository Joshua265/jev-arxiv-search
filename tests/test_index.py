from __future__ import annotations

import duckdb
import pytest

from arxiv_walker.index.lexical import BM25Index
from arxiv_walker.index.metadata import MetadataStore
from arxiv_walker.index.taxonomy import (
    SearchNode,
    build_tree,
    expand,
    group_of,
    node_filters_to_sql,
)


@pytest.fixture
def db_path(tmp_path: pytest.TempPathFactory) -> str:
    path = str(tmp_path / "test.duckdb")
    con = duckdb.connect(path)
    try:
        con.execute("""
            CREATE TABLE papers (
                arxiv_id VARCHAR,
                title VARCHAR,
                abstract VARCHAR,
                primary_category VARCHAR,
                submitted_at VARCHAR,
                updated_at VARCHAR,
                doi VARCHAR,
                journal_ref VARCHAR,
                comments VARCHAR,
                license VARCHAR,
                authors_json VARCHAR,
                versions_json VARCHAR
            )
        """)
        con.execute("""
            CREATE TABLE paper_categories (
                arxiv_id VARCHAR,
                category VARCHAR
            )
        """)
        papers = [
            ("2401.0001", "Deep Learning for NLP",
             "We apply transformers to text classification.",
             "cs.CL", "2024-01-15T00:00:00"),
            ("2401.0002", "Reinforcement Learning in Robotics",
             "A survey of RL methods for robotic control.",
             "cs.RO", "2024-02-20T00:00:00"),
            ("2401.0003", "Graph Neural Networks",
             "GNNs for molecular property prediction.",
             "cs.LG", "2023-11-10T00:00:00"),
            ("2401.0004", "Algebraic Topology",
             "An introduction to homotopy theory.",
             "math.AT", "2024-03-05T00:00:00"),
            ("2401.0005", "Combinatorial Designs",
             "New results in combinatorial design theory.",
             "math.CO", "2022-06-01T00:00:00"),
            ("2401.0006", "Quantum Error Correction",
             "Surface codes and fault tolerance.",
             "quant-ph", "2024-04-10T00:00:00"),
        ]
        for p in papers:
            con.execute(
                "INSERT INTO papers VALUES (?, ?, ?, ?, ?, '', '', '', '', '', '[]', '[]')",
                p,
            )
        cat_assignments = [
            ("2401.0001", "cs.CL"),
            ("2401.0001", "cs.AI"),
            ("2401.0002", "cs.RO"),
            ("2401.0002", "cs.AI"),
            ("2401.0003", "cs.LG"),
            ("2401.0003", "cs.AI"),
            ("2401.0004", "math.AT"),
            ("2401.0005", "math.CO"),
            ("2401.0006", "quant-ph"),
        ]
        for aid, cat in cat_assignments:
            con.execute("INSERT INTO paper_categories VALUES (?, ?)", [aid, cat])
    finally:
        con.close()
    return path


class TestGroupOf:
    def test_cs(self) -> None:
        assert group_of("cs.AI") == "cs"

    def test_math(self) -> None:
        assert group_of("math.CO") == "math"

    def test_hep_ph(self) -> None:
        assert group_of("hep-ph") == "physics"

    def test_quant_ph(self) -> None:
        assert group_of("quant-ph") == "physics"

    def test_unknown_prefix(self) -> None:
        assert group_of("bio.GN") == "bio"


class TestBuildTree:
    def test_tree_structure(self, db_path: str) -> None:
        class FakeSettings:
            recent_days = 365
            medium_days = 730

        nodes = build_tree(db_path, FakeSettings())
        node_map = {n.id: n for n in nodes}

        assert "root" in node_map
        assert node_map["root"].depth == 0
        assert node_map["root"].count == 6

        assert "group:cs" in node_map
        assert node_map["group:cs"].depth == 1
        assert node_map["group:cs"].count == 3

        assert "group:math" in node_map
        assert node_map["group:math"].depth == 1
        assert node_map["group:math"].count == 2

        assert "group:physics" in node_map
        assert node_map["group:physics"].depth == 1

        assert "cat:cs.AI" in node_map
        assert node_map["cat:cs.AI"].depth == 2
        assert node_map["cat:cs.AI"].count == 3

        assert "cat:cs.LG" in node_map
        assert node_map["cat:cs.LG"].depth == 2
        assert node_map["cat:cs.LG"].count == 1

        assert "cat:math.CO" in node_map
        assert node_map["cat:math.CO"].depth == 2
        assert node_map["cat:math.CO"].count == 1

        for cat_key in ("cs.AI", "cs.LG", "cs.RO", "math.AT", "math.CO", "quant-ph"):
            for bucket in ("recent", "medium", "older"):
                bid = f"cat:{cat_key}:{bucket}"
                assert bid in node_map, f"Missing node {bid}"
                assert node_map[bid].depth == 3

    def test_top_terms_and_titles(self, db_path: str) -> None:
        class FakeSettings:
            recent_days = 365
            medium_days = 730

        nodes = build_tree(db_path, FakeSettings())
        node_map = {n.id: n for n in nodes}
        root = node_map["root"]
        assert len(root.top_terms) > 0
        assert len(root.representative_titles) > 0


class TestExpand:
    def test_expand_root(self, db_path: str) -> None:
        class FakeSettings:
            recent_days = 365
            medium_days = 730

        nodes = build_tree(db_path, FakeSettings())
        root = [n for n in nodes if n.id == "root"][0]
        children = expand(root, nodes)
        assert all(n.parent_id == "root" for n in children)
        assert any(n.id == "group:cs" for n in children)
        assert any(n.id == "group:math" for n in children)

    def test_expand_category(self, db_path: str) -> None:
        class FakeSettings:
            recent_days = 365
            medium_days = 730

        nodes = build_tree(db_path, FakeSettings())
        cat_node = [n for n in nodes if n.id == "cat:cs.AI"][0]
        children = expand(cat_node, nodes)
        assert all(n.parent_id == "cat:cs.AI" for n in children)
        assert len(children) == 3
        assert any(n.id == "cat:cs.AI:recent" for n in children)
        assert any(n.id == "cat:cs.AI:medium" for n in children)
        assert any(n.id == "cat:cs.AI:older" for n in children)


class TestNodeFiltersToSql:
    def test_category(self) -> None:
        sql = node_filters_to_sql({"category": "cs.AI"})
        assert "pc.category = 'cs.AI'" in sql

    def test_group(self) -> None:
        sql = node_filters_to_sql({"group": "cs"})
        assert "pc.category = 'cs'" in sql
        assert "LIKE 'cs.%'" in sql

    def test_empty(self) -> None:
        assert node_filters_to_sql({}) == ""

    def test_year_bucket_recent(self) -> None:
        sql = node_filters_to_sql({"category": "cs.AI", "year_bucket": "recent"})
        assert "pc.category = 'cs.AI'" in sql
        assert "CAST(p.submitted_at AS TIMESTAMP) >=" in sql

    def test_year_bucket_older(self) -> None:
        sql = node_filters_to_sql({"category": "cs.AI", "year_bucket": "older"})
        assert "CAST(p.submitted_at AS TIMESTAMP) <" in sql


class TestBM25Index:
    def test_search_returns_hits(self, db_path: str) -> None:
        bm25 = BM25Index(db_path)
        node = SearchNode(id="cat:cs.AI", parent_id="group:cs", filters={"category": "cs.AI"})
        hits = bm25.search(node, "deep learning transformer", top_k=10)
        assert len(hits) > 0
        assert any("2401.0001" in h.arxiv_id for h in hits)

    def test_search_empty_node(self, db_path: str) -> None:
        bm25 = BM25Index(db_path)
        node = SearchNode(id="cat:math.AT", parent_id="group:math", filters={"category": "math.AT"})
        hits = bm25.search(node, "homotopy", top_k=10)
        assert len(hits) > 0

    def test_search_no_match(self, db_path: str) -> None:
        bm25 = BM25Index(db_path)
        node = SearchNode(id="cat:cs.AI", parent_id="group:cs", filters={"category": "cs.AI"})
        hits = bm25.search(node, "zzzzzzzzzzzzzzzzzzz", top_k=10)
        assert len(hits) == 0


class TestMetadataStore:
    def test_get_paper(self, db_path: str) -> None:
        store = MetadataStore(db_path)
        paper = store.get_paper("2401.0001")
        assert paper is not None
        assert paper["arxiv_id"] == "2401.0001"
        assert "Deep Learning" in paper["title"]

    def test_get_paper_not_found(self, db_path: str) -> None:
        store = MetadataStore(db_path)
        assert store.get_paper("9999.9999") is None

    def test_get_papers(self, db_path: str) -> None:
        store = MetadataStore(db_path)
        papers = store.get_papers(["2401.0001", "2401.0003"])
        assert len(papers) == 2
        assert papers[0]["arxiv_id"] == "2401.0001"
        assert papers[1]["arxiv_id"] == "2401.0003"

    def test_get_papers_preserves_order(self, db_path: str) -> None:
        store = MetadataStore(db_path)
        papers = store.get_papers(["2401.0003", "2401.0001"])
        assert papers[0]["arxiv_id"] == "2401.0003"
        assert papers[1]["arxiv_id"] == "2401.0001"

    def test_count_for_filters(self, db_path: str) -> None:
        store = MetadataStore(db_path)
        assert store.count_for_filters({"category": "cs.AI"}) == 3
        assert store.count_for_filters({"group": "math"}) == 2
        assert store.count_for_filters({}) == 6

    def test_titles_for_filters(self, db_path: str) -> None:
        store = MetadataStore(db_path)
        titles = store.titles_for_filters({"category": "cs.AI"}, limit=2)
        assert len(titles) == 2
        assert all(isinstance(t, str) for t in titles)
