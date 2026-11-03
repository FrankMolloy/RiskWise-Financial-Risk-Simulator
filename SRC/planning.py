"""Bounded contribution search, independent of Dash and market-data access."""

from collections.abc import Callable
from math import ceil, isfinite


def required_monthly(
    probability: Callable[[float], float],
    target: float,
    upper: float = 5000.0,
    iterations: int = 12,
) -> float | None:
    """Find a feasible £10 increment, or None if the upper bound is infeasible.

    The probability function must be non-decreasing and use identical random
    draws for each contribution. This searches a finite Monte Carlo sample,
    rather than guaranteeing the target probability in the real world.
    """
    if not isfinite(target) or not 0 < target <= 1:
        raise ValueError("target must be in (0, 1]")
    if not isfinite(upper) or upper <= 0 or upper % 10:
        raise ValueError("upper must be a positive multiple of 10")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if probability(0.0) >= target:
        return 0.0
    if probability(upper) < target:
        return None
    lo, hi = 0.0, upper
    for _ in range(iterations):
        mid = (lo + hi) / 2
        if probability(mid) >= target:
            hi = mid
        else:
            lo = mid
    return min(upper, ceil(hi / 10) * 10.0)
