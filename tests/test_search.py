from __future__ import annotations

from datetime import datetime

import duckdb
import pytest

from arxiv_walker.config import Settings
from arxiv_walker.decision.mock import MockDecisionEngine
from arxiv_walker.index.taxonomy import SearchNode
from arxiv_walker.search.expansion import build_branch_questions, build_rerank_questions
from arxiv_walker.search.node import is_leaf, node_summary
from arxiv_walker.search.priority import PriorityQueue, compute_priority
from arxiv_walker.search.rerank import rerank_papers
from arxiv_walker.search.walker import Walker


@pytest.fixture
def db_path(tmp_path: pytest.TempPathFactory) -> str:
    path = str(tmp_path / "test_search.duckdb")
    con = duckdb.connect(path)
    try:
        con.execute("""
            CREATE TABLE papers (
                arxiv_id VARCHAR,
                title VARCHAR,
                abstract VARCHAR,
                primary_category VARCHAR,
                submitted_at TIMESTAMP,
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
            ("2401.0001", "Transformers for Text Classification",
             "We apply transformer architectures to text classification tasks.",
             "cs.AI", datetime(2025, 1, 15)),
            ("2401.0002", "Deep Reinforcement Learning for Robotics",
             "A survey of deep RL methods for robotic control and manipulation.",
             "cs.AI", datetime(2025, 2, 20)),
            ("2401.0003", "Graph Neural Networks for Molecular Prediction",
             "GNNs applied to molecular property prediction and drug discovery.",
             "cs.AI", datetime(2025, 3, 10)),
            ("2401.0004", "Expert Systems in Medical Diagnosis",
             "Rule-based expert systems for medical diagnosis and decision support.",
             "cs.AI", datetime(2010, 1, 1)),
            ("2401.0005", "Genetic Algorithms for Combinatorial Optimization",
             "Evolutionary approaches to hard combinatorial optimization problems.",
             "cs.AI", datetime(2010, 6, 1)),
            ("2401.0006", "Large Language Model Training at Scale",
             "Techniques for training large language models efficiently.",
             "cs.LG", datetime(2025, 4, 10)),
            ("2401.0007", "Attention Mechanisms in Computer Vision",
             "Attention-based architectures for visual recognition tasks.",
             "cs.LG", datetime(2025, 5, 15)),
            ("2401.0008", "Bayesian Deep Learning Methods",
             "Bayesian approaches to uncertainty estimation in deep neural networks.",
             "cs.LG", datetime(2025, 6, 20)),
            ("2401.0009", "Neural Architecture Search with Reinforcement Learning",
             "Using RL to automatically discover optimal neural network architectures.",
             "cs.AI", datetime(2025, 7, 1)),
            ("2401.0010", "Fuzzy Logic Control Systems",
             "Fuzzy logic approaches to industrial control system design.",
             "cs.AI", datetime(2010, 3, 15)),
        ]
        for p in papers:
            con.execute(
                "INSERT INTO papers VALUES (?, ?, ?, ?, ?, '', '', '', '', '', '[]', '[]')",
                p,
            )
        cat_assignments = [
            ("2401.0001", "cs.AI"),
            ("2401.0002", "cs.AI"),
            ("2401.0003", "cs.AI"),
            ("2401.0004", "cs.AI"),
            ("2401.0005", "cs.AI"),
            ("2401.0006", "cs.LG"),
            ("2401.0007", "cs.LG"),
            ("2401.0008", "cs.LG"),
            ("2401.0009", "cs.AI"),
            ("2401.0010", "cs.AI"),
        ]
        for aid, cat in cat_assignments:
            con.execute("INSERT INTO paper_categories VALUES (?, ?)", [aid, cat])
    finally:
        con.close()
    return path


@pytest.fixture
def settings() -> Settings:
    s = Settings()
    s.search.candidate_cutoff = 0
    s.search.max_depth = 10
    s.search.final_top_k = 5
    s.search.rerank_top_k = 10
    s.search.bm25_top_k = 100
    s.search.exploration_branches = 3
    s.search.branch_threshold = 0.35
    s.search.extra_lexical_branch = True
    s.search.beam_width = 3
    return s


@pytest.fixture
def engine() -> MockDecisionEngine:
    return MockDecisionEngine(noul_value=0.9, score_value=2.5)


class TestNode:
    def test_is_leaf_by_count(self) -> None:
        s = Settings()
        s.search.candidate_cutoff = 200
        node = SearchNode(id="test", parent_id=None, count=50, depth=1)
        assert is_leaf(node, s)

    def test_is_leaf_by_depth(self) -> None:
        s = Settings()
        s.search.max_depth = 3
        node = SearchNode(id="test", parent_id=None, count=1000, depth=3)
        assert is_leaf(node, s)

    def test_is_not_leaf(self) -> None:
        s = Settings()
        s.search.candidate_cutoff = 200
        s.search.max_depth = 5
        node = SearchNode(id="test", parent_id=None, count=1000, depth=1)
        assert not is_leaf(node, s)

    def test_node_summary(self) -> None:
        node = SearchNode(
            id="cat:cs.AI",
            parent_id="group:cs",
            filters={"category": "cs.AI"},
            count=5,
            top_terms=["deep", "learning", "transformer"],
            representative_titles=["Paper A", "Paper B"],
            depth=2,
        )
        summary = node_summary(node)
        assert summary["id"] == "cat:cs.AI"
        assert summary["depth"] == 2
        assert summary["count"] == 5
        assert "top_terms" in summary
        assert "representative_titles" in summary
        assert "filters" in summary
        assert "centroid" not in summary


class TestExpansion:
    def test_build_branch_questions(self) -> None:
        parent = SearchNode(id="root", parent_id=None, count=10, depth=0)
        children = [
            SearchNode(id="group:cs", parent_id="root", count=6, depth=1,
                       top_terms=["deep", "learning"], representative_titles=["CS Paper"]),
            SearchNode(id="group:math", parent_id="root", count=2, depth=1,
                       top_terms=["algebra", "topology"], representative_titles=["Math Paper"]),
        ]
        questions = build_branch_questions("machine learning", parent, children)
        assert "group:cs" in questions
        assert "group:math" in questions
        for q in questions.values():
            assert q["type"] == "noul"
            assert "machine learning" in q["instructions"]
            assert "group:" in q["instructions"]

    def test_build_rerank_questions(self) -> None:
        papers = [
            {"arxiv_id": "2401.0001", "title": "Test Title", "abstract": "Test abstract."},
            {"arxiv_id": "2401.0002", "title": "Another Title", "abstract": "Another abstract."},
        ]
        questions = build_rerank_questions("machine learning", papers)
        assert "relevant_2401.0001" in questions
        assert "evidence_2401.0001" in questions
        assert "relevant_2401.0002" in questions
        assert "evidence_2401.0002" in questions
        assert questions["relevant_2401.0001"]["type"] == "noul"
        assert questions["evidence_2401.0001"]["type"] == "score"
        assert "criteria" in questions["evidence_2401.0001"]

    def test_build_rerank_questions_respects_max(self) -> None:
        papers = [
            {"arxiv_id": f"2401.{i:04d}", "title": f"Paper {i}", "abstract": ""}
            for i in range(10)
        ]
        questions = build_rerank_questions("test", papers, max_papers_per_call=3)
        assert len(questions) == 6  # 3 papers * 2 questions each


class TestPriorityQueue:
    def test_push_pop_order(self) -> None:
        q = PriorityQueue()
        n1 = SearchNode(id="a", parent_id=None, count=1, depth=0)
        n2 = SearchNode(id="b", parent_id=None, count=1, depth=0)
        n3 = SearchNode(id="c", parent_id=None, count=1, depth=0)
        q.push(n1, 0.5)
        q.push(n2, 1.0)
        q.push(n3, 0.8)
        node_a, pri_a = q.pop()
        assert node_a.id == "b"
        assert pri_a == 1.0
        node_b, pri_b = q.pop()
        assert node_b.id == "c"
        assert pri_b == 0.8
        node_c, pri_c = q.pop()
        assert node_c.id == "a"
        assert pri_c == 0.5

    def test_pop_empty(self) -> None:
        q = PriorityQueue()
        assert q.pop() is None

    def test_len(self) -> None:
        q = PriorityQueue()
        assert len(q) == 0
        n = SearchNode(id="a", parent_id=None, count=1, depth=0)
        q.push(n, 1.0)
        assert len(q) == 1

    def test_peek_priority(self) -> None:
        q = PriorityQueue()
        assert q.peek_priority() is None
        n = SearchNode(id="a", parent_id=None, count=1, depth=0)
        q.push(n, 0.7)
        assert q.peek_priority() == 0.7


class TestComputePriority:
    def test_multiplication(self) -> None:
        node = SearchNode(id="test", parent_id=None, count=1, depth=0)
        assert compute_priority(node, 0.8, 1.0) == 0.8
        assert compute_priority(node, 0.8, 0.5) == 0.4

    def test_floor(self) -> None:
        node = SearchNode(id="test", parent_id=None, count=1, depth=0)
        assert compute_priority(node, 0.0, 1.0) == 0.01
        assert compute_priority(node, -0.5, 1.0) == 0.01


@pytest.mark.asyncio
async def test_walker_search(db_path: str, settings: Settings, engine: MockDecisionEngine) -> None:
    walker = Walker(db_path, settings, engine)
    result = await walker.search("deep learning transformer")
    assert result.nodes_visited > 0
    assert result.decision_calls > 0
    assert len(result.final_papers) > 0
    for p in result.final_papers:
        assert "arxiv_id" in p
        assert "title" in p


@pytest.mark.asyncio
async def test_walker_search_query_match(
    db_path: str, settings: Settings, engine: MockDecisionEngine
) -> None:
    walker = Walker(db_path, settings, engine)
    result = await walker.search("transformer text classification")
    assert len(result.final_papers) > 0
    ids = [p["arxiv_id"] for p in result.final_papers]
    assert "2401.0001" in ids


@pytest.mark.asyncio
async def test_rerank_papers(
    db_path: str, settings: Settings, engine: MockDecisionEngine
) -> None:
    papers = [
        {"arxiv_id": "2401.0001", "title": "Transformers for Text Classification",
         "abstract": "We apply transformer architectures to text classification tasks."},
        {"arxiv_id": "2401.0006", "title": "Large Language Model Training at Scale",
         "abstract": "Techniques for training large language models efficiently."},
    ]
    result = await rerank_papers(engine, "deep learning", papers, chunk_size=25)
    assert len(result) == 2
    assert all(isinstance(r, tuple) and len(r) == 2 for r in result)
    assert result[0][1] >= result[1][1]  # sorted desc
    # With noul=0.9 and score=2.5, final = 0.9 * (2.5/3.0) = 0.75
    assert abs(result[0][1] - 0.75) < 0.01
