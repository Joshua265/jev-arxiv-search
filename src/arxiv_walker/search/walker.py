from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from arxiv_walker.config import Settings
from arxiv_walker.decision.base import DecisionEngine, DecisionError
from arxiv_walker.index.lexical import BM25Index
from arxiv_walker.index.metadata import MetadataStore
from arxiv_walker.index.taxonomy import SearchNode, build_tree
from arxiv_walker.search.expansion import build_branch_questions
from arxiv_walker.search.node import is_leaf, node_summary
from arxiv_walker.search.priority import PriorityQueue, compute_priority
from arxiv_walker.search.rerank import rerank_papers

logger = logging.getLogger(__name__)


@dataclass
class WalkerResult:
    candidates: list[dict[str, Any]] = field(default_factory=list)
    nodes_visited: int = 0
    decision_calls: int = 0
    branches_scored: int = 0
    final_papers: list[dict[str, Any]] = field(default_factory=list)


class Walker:
    def __init__(self, db_path: str, settings: Settings, engine: DecisionEngine) -> None:
        self.db_path = db_path
        self.settings = settings
        self.engine = engine
        self.tree = build_tree(db_path, settings)
        self.bm25 = BM25Index(db_path)
        self.meta = MetadataStore(db_path)

        self.children_map: dict[str, list[SearchNode]] = {}
        for node in self.tree:
            pid = node.parent_id or "root"
            self.children_map.setdefault(pid, []).append(node)

    async def search(self, query: str) -> WalkerResult:
        root = next((n for n in self.tree if n.id == "root"), None)
        if root is None:
            return WalkerResult()

        queue = PriorityQueue()
        queue.push(root, 1.0)

        candidates: list[dict[str, Any]] = []
        seen_paper_ids: set[str] = set()
        nodes_visited = 0
        decision_calls = 0
        branches_scored = 0

        max_nodes = min(
            self.settings.search.beam_width * self.settings.search.max_depth * 4,
            50,
        )
        max_candidates = 500

        while len(queue) > 0 and nodes_visited < max_nodes and len(candidates) < max_candidates:
            popped = queue.pop()
            if popped is None:
                break
            node, current_priority = popped
            nodes_visited += 1

            children = self.children_map.get(node.id, [])

            if is_leaf(node, self.settings) or not children:
                hits = self.bm25.search(node, query, top_k=self.settings.search.bm25_top_k)
                for hit in hits:
                    if hit.arxiv_id in seen_paper_ids:
                        continue
                    paper = self.meta.get_paper(hit.arxiv_id)
                    if paper is None:
                        continue
                    paper["score"] = hit.score
                    candidates.append(paper)
                    seen_paper_ids.add(hit.arxiv_id)
                continue

            questions = build_branch_questions(query, node, children)
            try:
                answers = await self.engine.decide(
                    state={"query": query, "parent": node_summary(node)},
                    questions=questions,
                )
            except DecisionError:
                logger.warning("Branch scoring decision failed for node %s", node.id)
                continue

            decision_calls += 1
            branches_scored += len(children)

            scored: dict[str, float] = {}
            for child in children:
                ans = answers.get(child.id, {})
                score = ans.get("noul", 0.0) if isinstance(ans, dict) else 0.0
                scored[child.id] = score

            sorted_children = sorted(children, key=lambda c: scored.get(c.id, 0.0), reverse=True)

            selected: list[SearchNode] = []
            for child in sorted_children[: self.settings.search.exploration_branches]:
                selected.append(child)

            for child in sorted_children[self.settings.search.exploration_branches :]:
                if scored.get(child.id, 0.0) >= self.settings.search.branch_threshold:
                    selected.append(child)

            if self.settings.search.extra_lexical_branch:
                best_child: SearchNode | None = None
                best_score = -1.0
                for child in children:
                    child_hits = self.bm25.search(child, query, top_k=1)
                    if child_hits and child_hits[0].score > best_score:
                        best_score = child_hits[0].score
                        best_child = child
                if best_child is not None and best_child.id not in {c.id for c in selected}:
                    selected.append(best_child)

            for child in selected:
                child_score = scored.get(child.id, 0.0)
                priority = compute_priority(child, child_score, current_priority)
                queue.push(child, priority)

        candidates.sort(key=lambda p: p.get("score", 0.0), reverse=True)
        rerank_candidates = candidates[: self.settings.search.rerank_top_k]

        reranked = await rerank_papers(
            self.engine, query, rerank_candidates, chunk_size=25
        )

        final_papers_raw = reranked[: self.settings.search.final_top_k]
        final_ids = [p["arxiv_id"] for p, _score in final_papers_raw]
        final_papers = self.meta.get_papers(final_ids)

        return WalkerResult(
            candidates=candidates,
            nodes_visited=nodes_visited,
            decision_calls=decision_calls,
            branches_scored=branches_scored,
            final_papers=final_papers,
        )
