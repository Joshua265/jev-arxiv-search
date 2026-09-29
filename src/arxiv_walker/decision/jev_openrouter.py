from __future__ import annotations

import asyncio
from typing import Any

import httpx

from arxiv_walker.decision.base import DecisionError, DecisionTimeoutError


class OpenRouterJevEngine:
    def __init__(
        self,
        api_key: str,
        model: str = "typesafe/jev-1.13",
        base_url: str = "https://openrouter.ai/api/alpha",
        timeout_sec: float = 60.0,
        max_retries: int = 3,
        _transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._timeout_sec = timeout_sec
        self._max_retries = max_retries
        self._transport = _transport

    async def decide(self, state: Any, questions: dict[str, dict]) -> dict[str, Any]:
        if not self._api_key:
            raise DecisionError("OPENROUTER_API_KEY is empty")

        url = f"{self._base_url}/decisions"
        body = {"model": self._model, "state": state, "questions": questions}
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(self._max_retries + 1):
            try:
                async with httpx.AsyncClient(
                    transport=self._transport,
                    timeout=httpx.Timeout(self._timeout_sec),
                ) as client:
                    response = await client.post(url, json=body, headers=headers)
            except httpx.TimeoutException as e:
                raise DecisionTimeoutError(str(e)) from e

            if response.status_code == 200:
                data = response.json()
                return data["answers"]

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After", "1")
                try:
                    sleep_sec = min(float(retry_after), 10.0)
                except ValueError:
                    sleep_sec = 1.0
                await asyncio.sleep(sleep_sec)
                continue

            if 400 <= response.status_code < 500:
                raise DecisionError(
                    f"HTTP {response.status_code}: {response.text}"
                )

            if 500 <= response.status_code < 600:
                if attempt < self._max_retries:
                    backoff = 2 ** attempt
                    await asyncio.sleep(backoff)
                    continue
                raise DecisionError(
                    f"HTTP {response.status_code}: {response.text}"
                )

            raise DecisionError(
                f"Unexpected HTTP {response.status_code}: {response.text}"
            )

        raise DecisionError("Max retries exceeded")
