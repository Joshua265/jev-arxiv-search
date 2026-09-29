from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import duckdb

from arxiv_walker.index.stopwords import STOPWORDS

GROUP_MAP: dict[str, str] = {
    "cs": "cs",
    "math": "math",
    "physics": "physics",
    "stat": "stat",
    "eess": "eess",
    "q-bio": "q-bio",
    "q-fin": "q-fin",
    "econ": "econ",
    "astro-ph": "physics",
    "cond-mat": "physics",
    "hep-ph": "physics",
    "hep-th": "physics",
    "hep-ex": "physics",
    "hep-lat": "physics",
    "nucl-ex": "physics",
    "nucl-th": "physics",
    "gr-qc": "physics",
    "quant-ph": "physics",
    "nlin": "physics",
    "cha-dyn": "physics",
    "adap-org": "physics",
    "cmp-lg": "physics",
    "patt-sol": "physics",
    "solv-int": "physics",
    "supr-con": "physics",
    "math-ph": "physics",
}

GROUP_NAMES: dict[str, str] = {
    "cs": "Computer Science",
    "math": "Mathematics",
    "physics": "Physics",
    "stat": "Statistics",
    "eess": "Electrical Engineering and Systems Science",
    "q-bio": "Quantitative Biology",
    "q-fin": "Quantitative Finance",
    "econ": "Economics",
}


@dataclass
class SearchNode:
    id: str
    parent_id: str | None
    filters: dict[str, Any] = field(default_factory=dict)
    count: int = 0
    top_terms: list[str] = field(default_factory=list)
    representative_titles: list[str] = field(default_factory=list)
    centroid: list[float] | None = None
    depth: int = 0


def group_of(category: str) -> str:
    prefix = category.split(".")[0] if "." in category else category
    return GROUP_MAP.get(prefix, prefix)


def group_name(group_key: str) -> str:
    return GROUP_NAMES.get(group_key, group_key.capitalize())


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in re.split(r"\W+", text) if t and t.lower() not in STOPWORDS]


def _top_terms(
    con: duckdb.DuckDBPyConnection, filters: dict[str, Any], limit: int = 10
) -> list[str]:
    where = node_filters_to_sql(filters)
    sql = "SELECT title, abstract FROM papers p"
    if "category" in filters or "group" in filters:
        sql += " JOIN paper_categories pc ON p.arxiv_id = pc.arxiv_id"
    if where:
        sql += f" WHERE {where}"
    sql += " LIMIT 10000"
    try:
        rows = con.execute(sql).fetchall()
    except Exception:
        return []
    counter: Counter[str] = Counter()
    for title, abstract in rows:
        for t in _tokenize(title or "") + _tokenize(abstract or ""):
            counter[t] += 1
    return [w for w, _ in counter.most_common(limit)]


def _representative_titles(
    con: duckdb.DuckDBPyConnection, filters: dict[str, Any], limit: int = 5
) -> list[str]:
    where = node_filters_to_sql(filters)
    sql = "SELECT title FROM papers p"
    if "category" in filters or "group" in filters:
        sql += " JOIN paper_categories pc ON p.arxiv_id = pc.arxiv_id"
    if where:
        sql += f" WHERE {where}"
    sql += " ORDER BY CAST(p.submitted_at AS TIMESTAMP) DESC NULLS LAST LIMIT ?"
    try:
        rows = con.execute(sql, [limit]).fetchall()
    except Exception:
        return []
    return [r[0] for r in rows if r[0]]


def _count_papers(con: duckdb.DuckDBPyConnection, filters: dict[str, Any]) -> int:
    where = node_filters_to_sql(filters)
    sql = "SELECT COUNT(DISTINCT p.arxiv_id) FROM papers p"
    if "category" in filters or "group" in filters:
        sql += " JOIN paper_categories pc ON p.arxiv_id = pc.arxiv_id"
    if where:
        sql += f" WHERE {where}"
    try:
        row = con.execute(sql).fetchone()
        return row[0] if row else 0
    except Exception:
        return 0


def build_tree(db_path: str, settings: Any) -> list[SearchNode]:
    con = duckdb.connect(db_path, read_only=True)
    try:
        nodes: list[SearchNode] = []
        root = SearchNode(id="root", parent_id=None, filters={}, depth=0)
        root.count = _count_papers(con, {})
        root.top_terms = _top_terms(con, {}, 10)
        root.representative_titles = _representative_titles(con, {}, 5)
        nodes.append(root)

        cat_rows = con.execute(
            "SELECT DISTINCT category FROM paper_categories ORDER BY category"
        ).fetchall()
        all_categories = [r[0] for r in cat_rows]

        groups: dict[str, list[str]] = {}
        for cat in all_categories:
            g = group_of(cat)
            groups.setdefault(g, []).append(cat)

        for gkey, cats in sorted(groups.items()):
            gid = f"group:{gkey}"
            gfilters: dict[str, Any] = {"group": gkey}
            gnode = SearchNode(id=gid, parent_id="root", filters=gfilters, depth=1)
            gnode.count = _count_papers(con, gfilters)
            gnode.top_terms = _top_terms(con, gfilters, 10)
            gnode.representative_titles = _representative_titles(con, gfilters, 5)
            nodes.append(gnode)

            for cat in sorted(cats):
                cid = f"cat:{cat}"
                cfilters: dict[str, Any] = {"category": cat}
                cnode = SearchNode(id=cid, parent_id=gid, filters=cfilters, depth=2)
                cnode.count = _count_papers(con, cfilters)
                cnode.top_terms = _top_terms(con, cfilters, 10)
                cnode.representative_titles = _representative_titles(con, cfilters, 5)
                nodes.append(cnode)

                yb = getattr(getattr(settings, "ingest", None), "year_buckets", None)
                recent_days = getattr(yb, "recent_days", 730)
                medium_days = getattr(yb, "medium_days", 1825)

                for bucket_key, _label, _cutoff_days in [
                    ("recent", "recent", recent_days),
                    ("medium", "medium", medium_days),
                    ("older", "older", None),
                ]:
                    bid = f"{cid}:{bucket_key}"
                    bfilters: dict[str, Any] = {
                        "category": cat,
                        "year_bucket": bucket_key,
                        "recent_days": recent_days,
                        "medium_days": medium_days,
                    }
                    bnode = SearchNode(id=bid, parent_id=cid, filters=bfilters, depth=3)
                    bnode.count = _count_papers(con, bfilters)
                    bnode.top_terms = _top_terms(con, bfilters, 10)
                    bnode.representative_titles = _representative_titles(con, bfilters, 5)
                    nodes.append(bnode)

    finally:
        con.close()
    return nodes


def node_filters_to_sql(filters: dict[str, Any]) -> str:
    clauses: list[str] = []
    if "category" in filters:
        clauses.append(f"pc.category = '{filters['category']}'")
    if "group" in filters:
        gkey = filters["group"]
        prefixes = [k for k, v in GROUP_MAP.items() if v == gkey]
        if not prefixes:
            prefixes = [gkey]
        group_clauses: list[str] = []
        for p in prefixes:
            group_clauses.append(
                f"(pc.category = '{p}' OR pc.category LIKE '{p}.%')"
            )
        clauses.append(f"({' OR '.join(group_clauses)})")
    if "year_bucket" in filters:
        bucket = filters["year_bucket"]
        rd = filters.get("recent_days", 730)
        md = filters.get("medium_days", 1825)
        if bucket == "recent":
            clauses.append(
                f"CAST(p.submitted_at AS TIMESTAMP) >= CURRENT_DATE - INTERVAL '{rd}' DAY"
            )
        elif bucket == "medium":
            clauses.append(
                f"CAST(p.submitted_at AS TIMESTAMP) >= CURRENT_DATE - INTERVAL '{md}' DAY "
                f"AND CAST(p.submitted_at AS TIMESTAMP) < CURRENT_DATE - INTERVAL '{rd}' DAY"
            )
        elif bucket == "older":
            clauses.append(
                f"CAST(p.submitted_at AS TIMESTAMP) < CURRENT_DATE - INTERVAL '{md}' DAY"
            )
    if "min_date" in filters:
        clauses.append(f"CAST(p.submitted_at AS TIMESTAMP) >= '{filters['min_date']}'")
    if "max_date" in filters:
        clauses.append(f"CAST(p.submitted_at AS TIMESTAMP) <= '{filters['max_date']}'")
    return " AND ".join(clauses)


def expand(node: SearchNode, all_nodes: list[SearchNode]) -> list[SearchNode]:
    return [n for n in all_nodes if n.parent_id == node.id]
