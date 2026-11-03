import numpy as np


def apply_crash(paths: np.ndarray, crash_pct: float, seed: int = 999) -> np.ndarray:
    """Educational stress test: one crash month per path then multiply future balances by (1-crash_pct)."""
    if not np.isfinite(crash_pct) or not 0 <= crash_pct <= 1:
        raise ValueError("crash_pct must be between 0 and 1")
    if paths.ndim != 2 or paths.shape[1] < 1:
        raise ValueError("paths must include an initial balance column")
    if paths.shape[1] == 1:
        return paths.copy()
    if crash_pct <= 0:
        return paths

    out = paths.copy()
    rng = np.random.default_rng(seed)
    n_sims, n_steps = out.shape

    lo = max(1, min(12, n_steps - 2))
    hi = max(lo + 1, n_steps - 12)
    crash_months = rng.integers(low=lo, high=hi, size=n_sims)

    factor = 1.0 - crash_pct
    for i in range(n_sims):
        m = int(crash_months[i])
        out[i, m:] *= factor

    return out


def deterministic_forecast(
    years: int,
    start_balance: float,
    monthly_contribution: float,
    expected_return_annual: float,
    inflation_annual: float,
    fee_annual: float,
) -> np.ndarray:
    """
    One-path deterministic projection using constant monthly growth.
    Returns inflation-adjusted balances (real terms).
    """
    months = int(years * 12)
    r_net_annual = expected_return_annual - fee_annual
    r_m = (1.0 + r_net_annual) ** (1.0 / 12.0) - 1.0
    infl_m = (1.0 + inflation_annual) ** (1.0 / 12.0) - 1.0

    bal = start_balance
    out = np.zeros(months + 1, dtype=float)
    out[0] = bal

    for t in range(1, months + 1):
        bal = bal * (1.0 + r_m) + monthly_contribution
        real = bal / ((1.0 + infl_m) ** t)
        out[t] = real

    return out
