from __future__ import annotations

import httpx
import pytest

from arxiv_walker.decision.base import DecisionEngine, DecisionError, DecisionTimeoutError
from arxiv_walker.decision.factory import make_decision_engine
from arxiv_walker.decision.jev_openrouter import OpenRouterJevEngine
from arxiv_walker.decision.mock import MockDecisionEngine


@pytest.mark.asyncio
async def test_mock_noul() -> None:
    engine: DecisionEngine = MockDecisionEngine(noul_value=0.95)
    answers = await engine.decide("state", {"q1": {"type": "noul"}})
    assert answers["q1"] == {"type": "noul", "noul": 0.95}


@pytest.mark.asyncio
async def test_mock_score() -> None:
    engine: DecisionEngine = MockDecisionEngine(score_value=3.0)
    answers = await engine.decide("state", {"q1": {"type": "score"}})
    assert answers["q1"] == {"type": "score", "score": 3.0, "confidence": 1.0}


@pytest.mark.asyncio
async def test_mock_choice() -> None:
    engine: DecisionEngine = MockDecisionEngine(choice_value="option_a")
    answers = await engine.decide("state", {"q1": {"type": "choice"}})
    assert answers["q1"] == {"type": "choice", "choice": "option_a", "confidence": 1.0}


@pytest.mark.asyncio
async def test_mock_unknown_type() -> None:
    engine: DecisionEngine = MockDecisionEngine()
    answers = await engine.decide("state", {"q1": {"type": "custom"}})
    assert answers["q1"] == {"type": "custom"}


@pytest.mark.asyncio
async def test_mock_multiple_questions() -> None:
    engine: DecisionEngine = MockDecisionEngine()
    questions = {
        "q1": {"type": "noul"},
        "q2": {"type": "score"},
        "q3": {"type": "choice"},
    }
    answers = await engine.decide("state", questions)
    assert answers["q1"] == {"type": "noul", "noul": 0.8}
    assert answers["q2"] == {"type": "score", "score": 2.0, "confidence": 1.0}
    assert answers["q3"] == {"type": "choice", "choice": "default", "confidence": 1.0}


@pytest.mark.asyncio
async def test_openrouter_empty_api_key() -> None:
    engine = OpenRouterJevEngine(api_key="")
    with pytest.raises(DecisionError, match="OPENROUTER_API_KEY is empty"):
        await engine.decide("state", {"q1": {"type": "noul"}})


@pytest.mark.asyncio
async def test_openrouter_success() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"answers": {"q1": {"type": "noul", "noul": 0.9}}},
        )

    transport = httpx.MockTransport(handler)
    engine = OpenRouterJevEngine(api_key="test-key", _transport=transport)
    answers = await engine.decide("state", {"q1": {"type": "noul"}})
    assert answers == {"q1": {"type": "noul", "noul": 0.9}}


@pytest.mark.asyncio
async def test_openrouter_429_retry() -> None:
    call_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(
            200,
            json={"answers": {"q1": {"type": "noul", "noul": 0.9}}},
        )

    transport = httpx.MockTransport(handler)
    engine = OpenRouterJevEngine(api_key="test-key", _transport=transport)
    answers = await engine.decide("state", {"q1": {"type": "noul"}})
    assert answers == {"q1": {"type": "noul", "noul": 0.9}}
    assert call_count == 2


@pytest.mark.asyncio
async def test_openrouter_4xx_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="bad request")

    transport = httpx.MockTransport(handler)
    engine = OpenRouterJevEngine(api_key="test-key", _transport=transport)
    with pytest.raises(DecisionError, match="HTTP 400"):
        await engine.decide("state", {"q1": {"type": "noul"}})


@pytest.mark.asyncio
async def test_openrouter_5xx_then_success() -> None:
    call_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(500, text="server error")
        return httpx.Response(
            200,
            json={"answers": {"q1": {"type": "noul", "noul": 0.9}}},
        )

    transport = httpx.MockTransport(handler)
    engine = OpenRouterJevEngine(api_key="test-key", _transport=transport)
    answers = await engine.decide("state", {"q1": {"type": "noul"}})
    assert answers == {"q1": {"type": "noul", "noul": 0.9}}
    assert call_count == 2


@pytest.mark.asyncio
async def test_openrouter_timeout() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    transport = httpx.MockTransport(handler)
    engine = OpenRouterJevEngine(
        api_key="test-key", _transport=transport, timeout_sec=0.001
    )
    with pytest.raises(DecisionTimeoutError):
        await engine.decide("state", {"q1": {"type": "noul"}})


def test_make_engine_mock() -> None:
    class FakeConfig:
        backend = "mock"

    engine = make_decision_engine(FakeConfig())  # type: ignore[arg-type]
    assert isinstance(engine, MockDecisionEngine)


def test_make_engine_unknown() -> None:
    class FakeConfig:
        backend = "nonexistent"

    with pytest.raises(ValueError, match="Unknown decision backend"):
        make_decision_engine(FakeConfig())  # type: ignore[arg-type]


def test_make_engine_jev_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    class FakeConfig:
        backend = "jev"
        model = "typesafe/jev-1.13"
        base_url = "https://openrouter.ai/api/alpha"
        timeout_sec = 60
        max_retries = 3

    with pytest.raises(DecisionError, match="OPENROUTER_API_KEY"):
        make_decision_engine(FakeConfig())  # type: ignore[arg-type]


def test_decision_engine_protocol() -> None:
    assert isinstance(MockDecisionEngine(), DecisionEngine)
    assert isinstance(OpenRouterJevEngine(api_key="k"), DecisionEngine)
