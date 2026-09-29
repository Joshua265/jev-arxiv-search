from __future__ import annotations

import re
from dataclasses import dataclass

import duckdb

from arxiv_walker.index.stopwords import STOPWORDS
from arxiv_walker.index.taxonomy import SearchNode, node_filters_to_sql


@dataclass
class PaperHit:
    arxiv_id: str
    title: str
    score: float


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in re.split(r"\W+", text) if t and t.lower() not in STOPWORDS]


class BM25Index:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def search(self, node: SearchNode, query: str, top_k: int = 100) -> list[PaperHit]:
        from rank_bm25 import BM25Okapi

        con = duckdb.connect(self.db_path, read_only=True)
        try:
            where = node_filters_to_sql(node.filters)
            sql = "SELECT p.arxiv_id, p.title, p.abstract FROM papers p"
            if "category" in node.filters or "group" in node.filters:
                sql += " JOIN paper_categories pc ON p.arxiv_id = pc.arxiv_id"
            if where:
                sql += f" WHERE {where}"
            sql += " LIMIT 50000"
            rows = con.execute(sql).fetchall()
        finally:
            con.close()

        if not rows:
            return []

        corpus = [_tokenize(f"{r[1] or ''} {r[2] or ''}") for r in rows]
        bm25 = BM25Okapi(corpus)
        query_tokens = _tokenize(query)
        scores = bm25.get_scores(query_tokens)

        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        hits: list[PaperHit] = []
        for idx, score in indexed[:top_k]:
            if score != 0:
                hits.append(PaperHit(arxiv_id=rows[idx][0], title=rows[idx][1], score=float(score)))
        return hits
