import time
from io import StringIO
from typing import Tuple

import numpy as np
import pandas as pd
import requests

# Simple in-memory cache: symbol -> (timestamp, close_series)
_CACHE = {}


def fetch_stooq_close_prices(symbol: str, cache_seconds: int = 3600) -> pd.Series:
    """
    Fetch daily close prices from Stooq as a pandas Series.

    Example symbols:
      - 'spy.us'  (S&P 500 ETF)
      - 'qqq.us'  (NASDAQ 100 ETF)
      - 'iwm.us'  (Russell 2000 ETF)

    Uses a cache so Dash doesn't refetch on every slider move.
    """
    symbol = symbol.lower().strip()
    now = time.time()

    if symbol in _CACHE:
        ts, close = _CACHE[symbol]
        if now - ts < cache_seconds:
            return close

    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    r = requests.get(url, timeout=10)
    r.raise_for_status()

    df = pd.read_csv(StringIO(r.text))
    if "Date" not in df.columns or "Close" not in df.columns:
        raise ValueError("Unexpected Stooq CSV format")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    close = df.set_index("Date")["Close"].dropna()

    # Ensure enough data
    if len(close) < 200:
        raise ValueError("Not enough historical data returned")

    _CACHE[symbol] = (now, close)
    return close


def estimate_annual_return_vol(close: pd.Series) -> Tuple[float, float]:
    """
    Estimate annualised mean return and volatility from daily log returns.

    Returns: (mu_annual, sigma_annual)
    """
    rets = np.log(close / close.shift(1)).dropna()

    mu_daily = rets.mean()
    sigma_daily = rets.std()

    mu_annual = float(mu_daily * 252)              # ~ trading days per year
    sigma_annual = float(sigma_daily * np.sqrt(252))

    return mu_annual, sigma_annual
