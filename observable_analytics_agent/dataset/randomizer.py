"""Reproducible random choices built only on random(), whose sequence Python guarantees."""

import random
from collections.abc import Sequence


class Randomizer:
    """Draws the same values for the same seed on every Python version."""

    def __init__(self, seed: int) -> None:
        self._generator = random.Random(seed)

    def below(self, upper_bound: int) -> int:
        """Return an integer in [0, upper_bound)."""
        return int(self._generator.random() * upper_bound)

    def between(self, low: int, high: int) -> int:
        """Return an integer in [low, high], both ends included."""
        return low + self.below(high - low + 1)

    def chance(self, probability: float) -> bool:
        return self._generator.random() < probability

    def pick[T](self, options: Sequence[T]) -> T:
        return options[self.below(len(options))]

    def weighted[T](self, options: Sequence[tuple[T, int]]) -> T:
        """Pick a value with probability proportional to its integer weight."""
        threshold = self.below(sum(weight for _, weight in options))
        for value, weight in options[:-1]:
            if threshold < weight:
                return value
            threshold -= weight
        return options[-1][0]
