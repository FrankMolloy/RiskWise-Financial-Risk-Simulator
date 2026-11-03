# Mathematical methodology

This describes the implementation, including its inconsistencies. The audit preserves the existing monthly return model and seeded paths; it does not replace them with geometric Brownian motion or historical bootstrapping.

## Monthly simulation

For each path and month, independently:

```text
R_t ~ Normal(mu / 12, sigma / sqrt(12))
B_0 = starting balance
B_t = B_(t-1) * (1 + R_t) + C
V_t = B_t / (1 + inflation)^(t / 12)
```

`mu` is the annual return input, `sigma` annual volatility, `C` a constant nominal monthly contribution, and `V_t` real purchasing power. The engine returns `(simulations, 12 * years + 1)` balances, including the initial balance. Contributions arrive **after** growth, so the first contribution earns no first-month return. Contributions do not rise with inflation. The dashboard starts at £1,000 and assumes 2% inflation.

Dashboard presets are cautious `(5%, 10%)`, balanced `(7%, 15%)` and aggressive `(9%, 20%)` for `(mu, sigma)`. Annual fees are subtracted from `mu`, with the resulting annual input floored at −50%. This approximates fee drag; it does not charge an explicit fee on each balance. Dividing the annual mean by 12 means `mu` is not an effective annual compounded return.

The engine uses a local NumPy `RandomState`, retaining the original MT19937 seeded draws. It no longer reseeds process-global NumPy state, avoiding interference between concurrent callbacks. Dashboard A uses seed 42; B uses 43 unless the same scenario reuses A. Fixed seeds aid reproducibility, not accuracy. Quality selects 1,000, 2,000 or 5,000 paths.

## Summaries and probabilities

- P10, P50 and P90 use NumPy's default linearly interpolated empirical percentiles. Chart bands are calculated separately at each month; a percentile curve is not one investment path.
- P10–P90 contains roughly the central 80% of sampled balances at that month. It is neither a worst-case bound nor a confidence interval for a model parameter.
- Goal probability is `count(V_T >= goal) / number_of_paths`, including equality, at the **final month**. It is not the probability of ever crossing the goal. Goals are in today's money. Non-positive goals disable this metric and return zero internally.
- Min/max are sample extremes, not theoretical loss/gain limits. Finite samples introduce Monte Carlo error; more paths cannot repair incorrect assumptions.

## Historical calibration

`market_data.py` requests daily closing prices from Stooq with a 10-second timeout, sorts by date, and requires at least 200 observations. Successful data is cached for an hour **per Python process**, without sharing across workers or restarts.

```text
daily_log_return = log(close_t / close_(t-1))
mu = mean(daily_log_return) * 252
sigma = sample_std(daily_log_return, ddof=1) * sqrt(252)
```

The dashboard clips `mu` to [−5%, 20%] and `sigma` to [5%, 60%]. Failures fall back to preset assumptions, with a message in the dashboard and solver. There is no fixed historical window, FX conversion or independently verified total-return adjustment.

**Known approximation:** annualised mean log returns are passed directly to the normal **simple-return** engine. No log-to-simple drift conversion is applied. Both comparison scenarios use the same calibrated asset parameters when calibration succeeds; independent seeds can still produce different sample results. Such a comparison is not evidence that one preset outperforms another.

## Stress test and other projections

`apply_crash` chooses one seeded crash index per path. For `S > 1` columns, let `lo = max(1, min(12, S−2))`; it draws from `[lo, max(lo+1, S−12))`, upper bound excluded. Every balance from that index onward is multiplied by `1 − crash_fraction`.

This is a **post-processing haircut**, not a crash return inserted into the monthly recurrence. It also scales contributions arriving after the crash, overstating their losses. Existing whole-year application results are preserved. A zero-horizon path now retains its initial balance, because no crash month exists; a manually supplied one-month path can only crash at month one.

The deterministic overlay uses `(1 + mu − fee)^(1/12) − 1` for monthly growth. It does **not** use the simulator's `(mu − fee)/12` convention and is not its expected-value curve. Its gap from the Monte Carlo median reflects both differing rate conventions and stochastic compounding. It excludes crash stress.

The sequence demo sorts the same monthly return sample ascending and descending, applies end-of-month contributions, and discounts for inflation. It uses the compound-rate conversion for its monthly mean. It illustrates ordering effects, rather than providing a second calibrated forecast. The sensitivity view uses 900 paths per ±1 percentage-point return case with different seeds; differences include sampling noise.

## Contribution solver

The solver searches [£0, £5,000] per month using 900 paths, seed 123, up to 12 bisection steps and fixed crash draws (seed 2001). Reusing draws reduces comparison noise. It checks whether zero already suffices or the upper bound is infeasible. It rounds the feasible upper bracket **up** to a £10 increment; the original nearest-£10 rounding could fall below the target. Settings update after the first Solve click; returning to Dashboard requires solving again.

The probability function is assumed non-decreasing in contributions. With intended parameters this is practical, but unbounded normal returns theoretically permit returns below −100%, negative balances and violations of monotonicity. No returns are clipped. The target applies to a finite sample, not future real-world success, and its sample differs from the dashboard's sample.

## Scope

Returns and volatility are constant, with independent, identically distributed monthly shocks. The model omits serial dependence, fat tails, changing regimes, taxes, transaction costs, withdrawals, multiple correlated assets and currency exposure. Inflation is fixed. It is an educational scenario model, not a validated pricing engine, investment recommendation or calibrated tail-risk system.
