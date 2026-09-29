from __future__ import annotations

from typing import Any


class MockDecisionEngine:
    def __init__(
        self,
        noul_value: float = 0.8,
        score_value: float = 2.0,
        choice_value: str = "default",
    ) -> None:
        self._noul_value = noul_value
        self._score_value = score_value
        self._choice_value = choice_value

    async def decide(self, state: Any, questions: dict[str, dict]) -> dict[str, Any]:
        answers: dict[str, Any] = {}
        for key, spec in questions.items():
            qtype = spec.get("type", "")
            if qtype == "noul":
                answers[key] = {"type": "noul", "noul": self._noul_value}
            elif qtype == "score":
                answers[key] = {
                    "type": "score",
                    "score": self._score_value,
                    "confidence": 1.0,
                }
            elif qtype == "choice":
                answers[key] = {
                    "type": "choice",
                    "choice": self._choice_value,
                    "confidence": 1.0,
                }
            else:
                answers[key] = {"type": qtype}
        return answers
