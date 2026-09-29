from __future__ import annotations

import asyncio
import logging
from typing import Any

from arxiv_walker.decision.base import DecisionEngine, DecisionError
from arxiv_walker.search.expansion import build_rerank_questions

logger = logging.getLogger(__name__)


async def rerank_papers(
    engine: DecisionEngine,
    query: str,
    papers: list[dict[str, Any]],
    chunk_size: int = 25,
) -> list[tuple[dict[str, Any], float]]:
    results: list[tuple[dict[str, Any], float]] = []

    async def process_chunk(chunk: list[dict[str, Any]]) -> list[tuple[dict[str, Any], float]]:
        questions = build_rerank_questions(query, chunk, max_papers_per_call=len(chunk))
        try:
            answers = await engine.decide(state={"query": query}, questions=questions)
        except DecisionError:
            logger.warning("Rerank decision call failed for chunk of %d papers", len(chunk))
            return [(p, 0.0) for p in chunk]

        chunk_results: list[tuple[dict[str, Any], float]] = []
        for paper in chunk:
            aid = paper["arxiv_id"]
            relevant_key = f"relevant_{aid}"
            evidence_key = f"evidence_{aid}"

            relevant_ans = answers.get(relevant_key, {})
            evidence_ans = answers.get(evidence_key, {})

            relevant_score = (
                relevant_ans.get("noul", 0.0) if isinstance(relevant_ans, dict) else 0.0
            )
            evidence_score = (
                evidence_ans.get("score", 0.0) if isinstance(evidence_ans, dict) else 0.0
            )

            final = relevant_score * (evidence_score / 3.0)
            chunk_results.append((paper, final))
        return chunk_results

    chunks = [papers[i : i + chunk_size] for i in range(0, len(papers), chunk_size)]
    chunk_results = await asyncio.gather(*[process_chunk(c) for c in chunks])

    for cr in chunk_results:
        results.extend(cr)

    results.sort(key=lambda x: x[1], reverse=True)
    return results
