from __future__ import annotations

from typing import Any

import duckdb

from arxiv_walker.index.taxonomy import node_filters_to_sql


class MetadataStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def get_paper(self, arxiv_id: str) -> dict[str, Any] | None:
        con = duckdb.connect(self.db_path, read_only=True)
        try:
            row = con.execute(
                "SELECT * FROM papers WHERE arxiv_id = ?", [arxiv_id]
            ).fetchone()
            if row is None:
                return None
            desc = [d[0] for d in con.description]
            return dict(zip(desc, row))
        finally:
            con.close()

    def get_papers(self, arxiv_ids: list[str]) -> list[dict[str, Any]]:
        if not arxiv_ids:
            return []
        con = duckdb.connect(self.db_path, read_only=True)
        try:
            placeholders = ", ".join("?" for _ in arxiv_ids)
            rows = con.execute(
                f"SELECT * FROM papers WHERE arxiv_id IN ({placeholders})",
                arxiv_ids,
            ).fetchall()
            desc = [d[0] for d in con.description]
            lookup = {r[0]: dict(zip(desc, r)) for r in rows}
            return [lookup[a] for a in arxiv_ids if a in lookup]
        finally:
            con.close()

    def count_for_filters(self, filters: dict[str, Any]) -> int:
        con = duckdb.connect(self.db_path, read_only=True)
        try:
            where = node_filters_to_sql(filters)
            sql = "SELECT COUNT(DISTINCT p.arxiv_id) FROM papers p"
            if "category" in filters or "group" in filters:
                sql += " JOIN paper_categories pc ON p.arxiv_id = pc.arxiv_id"
            if where:
                sql += f" WHERE {where}"
            row = con.execute(sql).fetchone()
            return row[0] if row else 0
        finally:
            con.close()

    def titles_for_filters(self, filters: dict[str, Any], limit: int = 5) -> list[str]:
        con = duckdb.connect(self.db_path, read_only=True)
        try:
            where = node_filters_to_sql(filters)
            sql = "SELECT p.title FROM papers p"
            if "category" in filters or "group" in filters:
                sql += " JOIN paper_categories pc ON p.arxiv_id = pc.arxiv_id"
            if where:
                sql += f" WHERE {where}"
            sql += " ORDER BY p.submitted_at DESC NULLS LAST LIMIT ?"
            rows = con.execute(sql, [limit]).fetchall()
            return [r[0] for r in rows if r[0]]
        finally:
            con.close()
