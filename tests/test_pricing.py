"""Cost comes from the usage the API reports and a versioned price table."""

from datetime import date

import pytest
from pydantic import ValidationError

from observable_analytics_agent.models.base import TokenUsage
from observable_analytics_agent.observability.pricing import (
    ModelPricing,
    PriceTier,
    PricingTable,
    load_pricing_table,
)

MILLION = 1_000_000
TIER_LIMIT = 100_000
CURRENT_MODELS = {"claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-1"}


def a_tier(limit: int | None, input_price: float = 2.0) -> PriceTier:
    return PriceTier(
        max_prompt_tokens=limit,
        input=input_price,
        output=10.0,
        cache_write_5m=2.5,
        cache_write_1h=4.0,
        cache_read=0.2,
    )


def table_with(model: str, *tiers: PriceTier) -> PricingTable:
    return PricingTable(
        as_of=date(2026, 1, 1),
        source="test",
        currency="USD",
        models={model: ModelPricing(tiers=list(tiers))},
    )


def test_input_and_output_are_priced_per_million_tokens() -> None:
    table = table_with("m", a_tier(None))
    usage = TokenUsage(input_tokens=MILLION, output_tokens=MILLION // 2)

    assert table.cost_usd("m", usage) == pytest.approx(2.0 + 5.0)


def test_cache_writes_and_reads_have_their_own_rates() -> None:
    table = table_with("m", a_tier(None))
    usage = TokenUsage(
        input_tokens=MILLION,
        cache_creation_input_tokens=MILLION,
        cache_creation_1h_input_tokens=400_000,
        cache_read_input_tokens=MILLION,
    )

    expected = 2.0 + 0.6 * 2.5 + 0.4 * 4.0 + 0.2
    assert table.cost_usd("m", usage) == pytest.approx(expected)


def test_the_prompt_length_selects_the_tier_and_counts_cache_tokens() -> None:
    table = table_with("m", a_tier(TIER_LIMIT, input_price=1.0), a_tier(None, input_price=5.0))

    at_the_limit = TokenUsage(input_tokens=TIER_LIMIT)
    over_the_limit = TokenUsage(input_tokens=1, cache_read_input_tokens=TIER_LIMIT)

    assert table.cost_usd("m", at_the_limit) == pytest.approx(TIER_LIMIT * 1.0 / MILLION)
    assert table.cost_usd("m", over_the_limit) == pytest.approx(
        (1 * 5.0 + TIER_LIMIT * 0.2) / MILLION
    )


def test_a_dated_model_id_matches_the_longest_listed_prefix() -> None:
    table = table_with("claude-haiku-5-5", a_tier(None))

    assert table.cost_usd("claude-haiku-5-5-20261001", TokenUsage(input_tokens=MILLION)) == 2.0


def test_a_model_without_a_listed_price_has_no_cost() -> None:
    assert table_with("m", a_tier(None)).cost_usd("other", TokenUsage(input_tokens=1)) is None


@pytest.mark.parametrize(
    "tiers",
    [
        [a_tier(TIER_LIMIT)],
        [a_tier(None), a_tier(None)],
        [a_tier(200_000), a_tier(TIER_LIMIT), a_tier(None)],
    ],
    ids=["last tier is bounded", "unbounded tier in the middle", "tiers out of order"],
)
def test_invalid_tiers_are_rejected(tiers: list[PriceTier]) -> None:
    with pytest.raises(ValidationError):
        ModelPricing(tiers=tiers)


def test_a_price_table_needs_at_least_one_tier() -> None:
    with pytest.raises(ValidationError):
        ModelPricing(tiers=[])


def test_the_packaged_price_file_is_dated_and_covers_the_current_models() -> None:
    table = load_pricing_table()

    assert table.as_of >= date(2026, 10, 1)
    assert table.source.startswith("https://platform.claude.com/docs/")
    assert CURRENT_MODELS <= set(table.models)


def test_haiku_prices_change_above_one_hundred_thousand_prompt_tokens() -> None:
    table = load_pricing_table()
    short_prompt = TokenUsage(input_tokens=TIER_LIMIT)
    long_prompt = TokenUsage(input_tokens=TIER_LIMIT + 1)

    assert table.cost_usd("claude-haiku-5-5", short_prompt) == pytest.approx(0.01)
    assert table.cost_usd("claude-haiku-5-5", long_prompt) == pytest.approx(0.0500005)


def test_a_typical_agent_step_costs_a_fraction_of_a_cent_on_haiku() -> None:
    usage = TokenUsage(input_tokens=2_000, output_tokens=200)

    assert load_pricing_table().cost_usd("claude-haiku-5-5", usage) == pytest.approx(0.0003)
