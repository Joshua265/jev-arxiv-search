from __future__ import annotations

import heapq

from arxiv_walker.index.taxonomy import SearchNode


class PriorityQueue:
    def __init__(self) -> None:
        self._heap: list[tuple[float, int, SearchNode]] = []
        self._counter = 0

    def push(self, node: SearchNode, priority: float) -> None:
        heapq.heappush(self._heap, (-priority, self._counter, node))
        self._counter += 1

    def pop(self) -> tuple[SearchNode, float] | None:
        if not self._heap:
            return None
        neg_priority, _counter, node = heapq.heappop(self._heap)
        return node, -neg_priority

    def __len__(self) -> int:
        return len(self._heap)

    def peek_priority(self) -> float | None:
        if not self._heap:
            return None
        return -self._heap[0][0]


def compute_priority(
    node: SearchNode, branch_score: float, parent_priority: float = 1.0
) -> float:
    return parent_priority * max(branch_score, 0.01)
