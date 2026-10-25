"""Test doubles shared by several test modules."""

from datetime import date

from observable_analytics_agent.models.base import ChatModel, ModelReply, ModelRequest
from observable_analytics_agent.observability.pricing import ModelPricing, PriceTier, PricingTable

PRICED_MODEL = "priced-model"
FAKE_INPUT_PRICE = 1.0
FAKE_OUTPUT_PRICE = 2.0
FAKE_CACHE_WRITE_5M_PRICE = 1.25
FAKE_CACHE_WRITE_1H_PRICE = 2.0
FAKE_CACHE_READ_PRICE = 0.1


class FakeClock:
    def __init__(self) -> None:
        self.seconds = 0.0

    def __call__(self) -> float:
        return self.seconds

    def advance(self, seconds: float) -> None:
        self.seconds += seconds


class ClockAdvancingModel:
    """Makes every call to the wrapped model take `seconds` on the fake clock."""

    def __init__(self, inner: ChatModel, clock: FakeClock, seconds: float) -> None:
        self._inner = inner
        self._clock = clock
        self._seconds = seconds

    def complete(self, request: ModelRequest) -> ModelReply:
        self._clock.advance(self._seconds)
        return self._inner.complete(request)


def fake_pricing() -> PricingTable:
    tier = PriceTier(
        max_prompt_tokens=None,
        input=FAKE_INPUT_PRICE,
        output=FAKE_OUTPUT_PRICE,
        cache_write_5m=FAKE_CACHE_WRITE_5M_PRICE,
        cache_write_1h=FAKE_CACHE_WRITE_1H_PRICE,
        cache_read=FAKE_CACHE_READ_PRICE,
    )
    return PricingTable(
        as_of=date(2026, 1, 1),
        source="test fixture",
        currency="USD",
        models={PRICED_MODEL: ModelPricing(tiers=[tier])},
    )
