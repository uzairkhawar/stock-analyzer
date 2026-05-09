"""
Utilities — caching, ticker normalization, safe getters, time horizons.
"""
from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

import config

logger = logging.getLogger("stock_analyzer")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ---------------------------------------------------------------------------
# Ticker normalization
# ---------------------------------------------------------------------------
def normalize_ticker(ticker: str, market: str = "US") -> str:
    """Normalize a ticker symbol to the yfinance convention for the given market."""
    t = ticker.strip().upper()
    suffix = config.MARKET_SUFFIX.get(market.upper(), "")
    if suffix and not t.endswith(suffix):
        # Saudi tickers are often given as the 4-digit code only
        t = f"{t}{suffix}"
    return t


def horizon_to_period(horizon: str) -> str:
    """Map a UI horizon label to a yfinance period string."""
    table = {
        "1D": "5d", "1W": "1mo", "1M": "3mo",
        "3M": "6mo", "6M": "1y", "1Y": "2y",
        "3Y": "5y", "5Y": "10y", "MAX": "max",
    }
    return table.get(horizon.upper(), "1y")


def horizon_to_interval(horizon: str) -> str:
    """Map a UI horizon label to a sensible candle interval."""
    table = {
        "1D": "5m", "1W": "30m", "1M": "1d",
        "3M": "1d", "6M": "1d", "1Y": "1d",
        "3Y": "1wk", "5Y": "1wk", "MAX": "1mo",
    }
    return table.get(horizon.upper(), "1d")


# ---------------------------------------------------------------------------
# Safe getters — fundamentals dicts can have many missing keys
# ---------------------------------------------------------------------------
def safe_get(d: Optional[dict], *keys, default: Any = None) -> Any:
    """Try multiple keys; return the first non-None, finite value."""
    if not d:
        return default
    for k in keys:
        v = d.get(k)
        if v is None:
            continue
        try:
            # Filter NaNs
            if isinstance(v, float) and (v != v):
                continue
        except Exception:
            pass
        return v
    return default


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------
def _cache_path(key: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in key)
    return config.CACHE_DIR / f"{safe}.pkl"


def cache_get(key: str, ttl_minutes: Optional[int] = None) -> Optional[Any]:
    """Return cached object if fresh, else None."""
    ttl = ttl_minutes if ttl_minutes is not None else config.CACHE_TTL_MINUTES
    p = _cache_path(key)
    if not p.exists():
        return None
    age = datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)
    if age > timedelta(minutes=ttl):
        return None
    try:
        with p.open("rb") as f:
            return pickle.load(f)
    except Exception as e:  # corrupt cache — drop it
        logger.warning("Cache read failed for %s: %s", key, e)
        try:
            p.unlink()
        except Exception:
            pass
        return None


def cache_set(key: str, value: Any) -> None:
    p = _cache_path(key)
    try:
        with p.open("wb") as f:
            pickle.dump(value, f)
    except Exception as e:
        logger.warning("Cache write failed for %s: %s", key, e)


def cached(key_fn: Callable[..., str], ttl_minutes: Optional[int] = None):
    """Decorator caching the return of a function under a derived key."""
    def deco(fn: Callable):
        def wrap(*args, **kwargs):
            key = key_fn(*args, **kwargs)
            hit = cache_get(key, ttl_minutes)
            if hit is not None:
                return hit
            res = fn(*args, **kwargs)
            if res is not None:
                cache_set(key, res)
            return res
        wrap.__name__ = fn.__name__
        wrap.__doc__ = fn.__doc__
        return wrap
    return deco


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
def pct_change(a: float, b: float) -> Optional[float]:
    """Return (a-b)/b as a percentage, or None if b is 0/None."""
    try:
        if b is None or a is None or b == 0:
            return None
        return (a - b) / b * 100.0
    except Exception:
        return None


def round_or_none(v: Any, n: int = 2) -> Any:
    try:
        return round(float(v), n)
    except Exception:
        return None


def is_market_data_valid(df: Optional[pd.DataFrame], min_rows: int = 30) -> bool:
    return df is not None and not df.empty and len(df) >= min_rows
