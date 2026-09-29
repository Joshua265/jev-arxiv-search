from __future__ import annotations

from typing import Any

from arxiv_walker.config import Settings
from arxiv_walker.index.taxonomy import SearchNode

__all__ = ["SearchNode", "is_leaf", "node_summary"]


def is_leaf(node: SearchNode, settings: Settings) -> bool:
    return (
        node.count <= settings.search.candidate_cutoff
        or node.depth >= settings.search.max_depth
    )


def node_summary(node: SearchNode) -> dict[str, Any]:
    return {
        "id": node.id,
        "depth": node.depth,
        "count": node.count,
        "top_terms": node.top_terms,
        "representative_titles": node.representative_titles,
        "filters": node.filters,
    }
