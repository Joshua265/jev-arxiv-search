from __future__ import annotations

from typing import Any

from arxiv_walker.index.taxonomy import SearchNode


def build_branch_questions(
    query: str, parent: SearchNode, children: list[SearchNode]
) -> dict[str, dict]:
    questions: dict[str, dict] = {}
    for child in children:
        questions[child.id] = {
            "type": "noul",
            "instructions": (
                f"Would exploring branch {child.id} have a meaningful chance "
                'of containing papers that substantively help answer the '
                f'research query: "{query}"? Semantic relevance matters more '
                "than simple keyword overlap. Branch description: "
                f"top terms={child.top_terms[:8]}, "
                f"representative titles={child.representative_titles[:3]}, "
                f"paper count={child.count}."
            ),
        }
    return questions


def build_rerank_questions(
    query: str, papers: list[dict[str, Any]], max_papers_per_call: int = 25
) -> dict[str, dict]:
    questions: dict[str, dict] = {}
    for paper in papers[:max_papers_per_call]:
        arxiv_id = paper["arxiv_id"]
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        questions[f"relevant_{arxiv_id}"] = {
            "type": "noul",
            "instructions": (
                f'Is this paper substantively relevant to: "{query}"? '
                f"Title: {title}. Abstract: {abstract}"
            ),
        }
        questions[f"evidence_{arxiv_id}"] = {
            "type": "score",
            "instructions": (
                f'How directly does the evidence in this abstract address: "{query}"? '
                f"Title: {title}. Abstract: {abstract}"
            ),
            "criteria": [
                "not relevant",
                "weakly connected",
                "useful",
                "directly relevant",
            ],
        }
    return questions
