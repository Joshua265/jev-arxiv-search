from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


class DecisionError(Exception):
    pass


class DecisionTimeoutError(DecisionError):
    pass


@runtime_checkable
class DecisionEngine(Protocol):
    async def decide(self, state: Any, questions: dict[str, dict]) -> dict[str, Any]:
        ...
