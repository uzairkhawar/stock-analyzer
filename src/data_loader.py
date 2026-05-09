"""
Data loader — provider-agnostic interface.

The DataProvider ABC defines the contract. YFinanceProvider is the default
implementation. To add Polygon/Finnhub/Alpha Vantage in v2, implement the same
interface and return Pandas DataFrames with the same column shape.

Expected OHLCV DataFrame columns (case-insensitive friendly, but we standardize):
    Open, High, Low, Close, Volume   (DatetimeIndex, sorted ascending)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pandas as pd

import config
from src.utils import (
    cache_get, cache_set, logger, normalize_ticker,
    horizon_to_period, horizon_to_interval, is_market_data_valid,
)

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
class DataProvider(ABC):
    """Abstract contract for any data provider."""

    @abstractmethod
    def fetch_ohlcv(
        self, ticker: str, period: str = "1y", interval: str = "1d",
    ) -> Optional[pd.DataFrame]:
        ...

    @abstractmethod
    def fetch_info(self, ticker: str) -> Dict:
        """Company info / fundamentals snapshot. May return {}."""

    @abstractmethod
    def fetch_financials(self, ticker: str) -> Dict[str, pd.DataFrame]:
        """Returns dict of DataFrames: income_stmt, balance_sheet, cashflow."""


# ---------------------------------------------------------------------------
# yfinance implementation
# ---------------------------------------------------------------------------
class YFinanceProvider(DataProvider):
    """
    Default provider using yfinance. Free, no API key, but rate-limited
    and quality varies by market (Saudi Tadawul coverage is partial).
    """
    name = "yfinance"

    def _normalize_df(self, df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df is None or df.empty:
            return None
        # yfinance sometimes returns multi-level columns when downloading multiple tickers
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # Standardize column names
        df = df.rename(columns=str.title)
        keep = [c for c in ("Open", "High", "Low", "Close", "Volume", "Adj Close") if c in df.columns]
        df = df[keep].copy()
        df = df.dropna(subset=[c for c in ("Open", "High", "Low", "Close") if c in df.columns])
        df = df.sort_index()
        return df

    def fetch_ohlcv(
        self, ticker: str, period: str = "1y", interval: str = "1d",
    ) -> Optional[pd.DataFrame]:
        if yf is None:
            raise RuntimeError("yfinance is not installed. pip install yfinance")
        cache_key = f"ohlcv_{ticker}_{period}_{interval}"
        cached = cache_get(cache_key)
        if cached is not None:
            return cached
        try:
            df = yf.download(
                ticker, period=period, interval=interval,
                auto_adjust=False, progress=False, threads=False,
            )
            df = self._normalize_df(df)
            if not is_market_data_valid(df):
                logger.warning("No/insufficient OHLCV for %s (%s/%s)", ticker, period, interval)
                return None
            cache_set(cache_key, df)
            return df
        except Exception as e:
            logger.error("fetch_ohlcv(%s) failed: %s", ticker, e)
            return None

    def fetch_info(self, ticker: str) -> Dict:
        if yf is None:
            return {}
        cache_key = f"info_{ticker}"
        cached = cache_get(cache_key, ttl_minutes=24 * 60)
        if cached is not None:
            return cached
        try:
            t = yf.Ticker(ticker)
            # .info is fragile; fall back to .fast_info if needed
            info: Dict = {}
            try:
                info = dict(t.info or {})
            except Exception as e:
                logger.warning("yfinance .info failed for %s: %s", ticker, e)
            try:
                fi = getattr(t, "fast_info", None)
                if fi:
                    for k in ("last_price", "previous_close", "market_cap",
                              "year_high", "year_low", "shares", "currency"):
                        v = getattr(fi, k, None)
                        if v is not None and k not in info:
                            info[k] = v
            except Exception:
                pass
            cache_set(cache_key, info)
            return info
        except Exception as e:
            logger.error("fetch_info(%s) failed: %s", ticker, e)
            return {}

    def fetch_financials(self, ticker: str) -> Dict[str, pd.DataFrame]:
        if yf is None:
            return {}
        cache_key = f"fin_{ticker}"
        cached = cache_get(cache_key, ttl_minutes=24 * 60)
        if cached is not None:
            return cached
        out: Dict[str, pd.DataFrame] = {}
        try:
            t = yf.Ticker(ticker)
            for attr, label in [
                ("income_stmt", "income_stmt"),
                ("balance_sheet", "balance_sheet"),
                ("cashflow", "cashflow"),
            ]:
                try:
                    df = getattr(t, attr, None)
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        out[label] = df
                except Exception:
                    continue
            cache_set(cache_key, out)
        except Exception as e:
            logger.error("fetch_financials(%s) failed: %s", ticker, e)
        return out


# ---------------------------------------------------------------------------
# Convenience / facade
# ---------------------------------------------------------------------------
DEFAULT_PROVIDER: DataProvider = YFinanceProvider()


def load_stock(
    ticker: str, market: str = "US", horizon: str = "1Y",
    provider: Optional[DataProvider] = None,
) -> Dict:
    """
    Load everything we need for one stock in one call.

    Returns a dict with keys: ticker, market, ohlcv (DataFrame),
    info (dict), financials (dict of DataFrames), benchmark_ohlcv (DataFrame|None).
    """
    provider = provider or DEFAULT_PROVIDER
    sym = normalize_ticker(ticker, market)
    period = horizon_to_period(horizon)
    interval = horizon_to_interval(horizon)
    # Always pull at least 2 years of daily for indicator stability
    daily = provider.fetch_ohlcv(sym, period="2y", interval="1d")
    user_view = provider.fetch_ohlcv(sym, period=period, interval=interval) if interval != "1d" else daily
    info = provider.fetch_info(sym)
    financials = provider.fetch_financials(sym)

    bench_sym = config.DEFAULT_BENCHMARK_PER_MARKET.get(market.upper(), "^GSPC")
    bench = provider.fetch_ohlcv(bench_sym, period="2y", interval="1d")

    return {
        "ticker": sym,
        "market": market.upper(),
        "ohlcv": daily,
        "user_view": user_view if user_view is not None else daily,
        "info": info,
        "financials": financials,
        "benchmark_ticker": bench_sym,
        "benchmark_ohlcv": bench,
    }


def batch_load_ohlcv(
    tickers: List[str], market: str = "US",
    period: str = "1y", interval: str = "1d",
    provider: Optional[DataProvider] = None,
) -> Dict[str, pd.DataFrame]:
    """Load OHLCV for many tickers (used by the screener)."""
    provider = provider or DEFAULT_PROVIDER
    out: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        sym = normalize_ticker(t, market)
        df = provider.fetch_ohlcv(sym, period=period, interval=interval)
        if df is not None:
            out[sym] = df
    return out
