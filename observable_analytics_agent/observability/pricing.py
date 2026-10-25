"""Token prices from a versioned config file, and the cost of one model call."""

from datetime import date
from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field, model_validator

from observable_analytics_agent.models.base import TokenUsage

DEFAULT_PRICING_PATH = Path(__file__).resolve().parents[2] / "config" / "pricing.json"
TOKENS_PER_PRICE_UNIT = 1_000_000


class PriceTier(BaseModel):
    """Prices in USD per million tokens for prompts of up to `max_prompt_tokens` tokens."""

    max_prompt_tokens: int | None
    input: float
    output: float
    cache_write_5m: float
    cache_write_1h: float
    cache_read: float


class ModelPricing(BaseModel):
    tiers: list[PriceTier] = Field(min_length=1)

    @model_validator(mode="after")
    def tiers_are_ordered_and_the_last_has_no_limit(self) -> Self:
        limits = [tier.max_prompt_tokens for tier in self.tiers]
        if limits[-1] is not None:
            raise ValueError("the last tier must have no prompt token limit")
        bounded = limits[:-1]
        if None in bounded or bounded != sorted(bounded):
            raise ValueError("tiers must be ordered by ascending prompt token limit")
        return self

    def tier_for(self, prompt_tokens: int) -> PriceTier:
        return next(
            tier
            for tier in self.tiers
            if tier.max_prompt_tokens is None or prompt_tokens <= tier.max_prompt_tokens
        )


class PricingTable(BaseModel):
    as_of: date
    source: str
    currency: str
    models: dict[str, ModelPricing]

    def cost_usd(self, model: str, usage: TokenUsage) -> float | None:
        """Cost of one call, or None when the model has no listed price.

        The prompt length that selects the tier counts every input token, cache reads and
        writes included, as the pricing page defines it.
        """
        pricing = self._pricing_for(model)
        if pricing is None:
            return None
        prompt_tokens = (
            usage.input_tokens + usage.cache_creation_input_tokens + usage.cache_read_input_tokens
        )
        tier = pricing.tier_for(prompt_tokens)
        written_for_5m = usage.cache_creation_input_tokens - usage.cache_creation_1h_input_tokens
        priced_tokens = (
            usage.input_tokens * tier.input
            + usage.output_tokens * tier.output
            + written_for_5m * tier.cache_write_5m
            + usage.cache_creation_1h_input_tokens * tier.cache_write_1h
            + usage.cache_read_input_tokens * tier.cache_read
        )
        return priced_tokens / TOKENS_PER_PRICE_UNIT

    def _pricing_for(self, model: str) -> ModelPricing | None:
        """Match a model id exactly, or by the longest listed prefix for dated ids."""
        if model in self.models:
            return self.models[model]
        prefixes = [name for name in self.models if model.startswith(name)]
        return self.models[max(prefixes, key=len)] if prefixes else None


def load_pricing_table(path: Path = DEFAULT_PRICING_PATH) -> PricingTable:
    return PricingTable.model_validate_json(path.read_text(encoding="utf-8"))
