"""
Fundamental analysis — extracts ratios and growth from yfinance data.

Many fields will be missing for non-US stocks (especially Tadawul). All getters
fail gracefully and return None where data is unavailable. Functions here do
not score — that happens in scoring.py.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.utils import safe_get, pct_change, logger


def _row_to_value(df: pd.DataFrame, candidates) -> Optional[float]:
    """Look up the most recent annual value for any of the candidate row labels."""
    if df is None or df.empty:
        return None
    for c in candidates:
        if c in df.index:
            row = df.loc[c].dropna()
            if not row.empty:
                try:
                    return float(row.iloc[0])  # most recent column first in yfinance
                except Exception:
                    continue
    return None


def _two_period_growth(df: pd.DataFrame, candidates) -> Optional[float]:
    """Compute year-over-year growth (%) for the first available row."""
    if df is None or df.shape[1] < 2:
        return None
    for c in candidates:
        if c in df.index:
            row = df.loc[c].dropna()
            if len(row) >= 2:
                cur = float(row.iloc[0])
                prev = float(row.iloc[1])
                return pct_change(cur, prev)
    return None


def extract_fundamentals(info: Dict, financials: Dict[str, pd.DataFrame]) -> Dict:
    """
    Build a single dictionary of fundamentals & ratios from raw yfinance outputs.
    Every field may be None — consumers must check.
    """
    inc = financials.get("income_stmt")
    bal = financials.get("balance_sheet")
    cf = financials.get("cashflow")

    revenue = _row_to_value(inc, ["Total Revenue", "Revenue"])
    revenue_prev = None
    revenue_growth = _two_period_growth(inc, ["Total Revenue", "Revenue"])
    net_income = _row_to_value(inc, ["Net Income", "Net Income Common Stockholders"])
    net_income_growth = _two_period_growth(inc, ["Net Income", "Net Income Common Stockholders"])
    gross_profit = _row_to_value(inc, ["Gross Profit"])
    operating_income = _row_to_value(inc, ["Operating Income", "Operating Income or Loss"])
    eps = safe_get(info, "trailingEps")
    eps_growth = safe_get(info, "earningsQuarterlyGrowth")  # quarterly y/y as decimal

    total_debt = _row_to_value(bal, ["Total Debt", "Long Term Debt"])
    cash = _row_to_value(bal, ["Cash And Cash Equivalents", "Cash"])
    total_equity = _row_to_value(bal, ["Stockholders Equity", "Total Stockholder Equity"])

    fcf = _row_to_value(cf, ["Free Cash Flow"])
    if fcf is None:
        ocf = _row_to_value(cf, ["Operating Cash Flow", "Total Cash From Operating Activities"])
        capex = _row_to_value(cf, ["Capital Expenditure", "Capital Expenditures"])
        if ocf is not None and capex is not None:
            fcf = ocf + capex  # capex is negative

    # Margins
    gross_margin = (gross_profit / revenue * 100) if (gross_profit and revenue) else None
    operating_margin = (operating_income / revenue * 100) if (operating_income and revenue) else None
    net_margin = (net_income / revenue * 100) if (net_income and revenue) else None

    # Valuation ratios — prefer info; fall back to derivations
    pe = safe_get(info, "trailingPE")
    forward_pe = safe_get(info, "forwardPE")
    ps = safe_get(info, "priceToSalesTrailing12Months")
    pb = safe_get(info, "priceToBook")
    peg = safe_get(info, "pegRatio")
    dividend_yield = safe_get(info, "dividendYield")
    if dividend_yield is not None and dividend_yield > 1:
        # yfinance sometimes returns % rather than decimal
        dividend_yield = dividend_yield / 100.0

    roe = safe_get(info, "returnOnEquity")
    roa = safe_get(info, "returnOnAssets")
    debt_to_equity = safe_get(info, "debtToEquity")
    current_ratio = safe_get(info, "currentRatio")
    quick_ratio = safe_get(info, "quickRatio")

    market_cap = safe_get(info, "marketCap", "market_cap")
    sector = safe_get(info, "sector")
    industry = safe_get(info, "industry")
    name = safe_get(info, "longName", "shortName")
    currency = safe_get(info, "currency", default="USD")

    return {
        "name": name,
        "sector": sector,
        "industry": industry,
        "currency": currency,
        "market_cap": market_cap,
        # Income
        "revenue": revenue,
        "revenue_growth_yoy_pct": revenue_growth,
        "net_income": net_income,
        "net_income_growth_yoy_pct": net_income_growth,
        "eps": eps,
        "eps_growth_qoq_pct": (eps_growth * 100) if isinstance(eps_growth, (int, float)) else None,
        # Margins
        "gross_margin_pct": gross_margin,
        "operating_margin_pct": operating_margin,
        "net_margin_pct": net_margin,
        # Balance sheet & cashflow
        "total_debt": total_debt,
        "cash": cash,
        "total_equity": total_equity,
        "free_cash_flow": fcf,
        # Ratios
        "pe": pe, "forward_pe": forward_pe, "ps": ps, "pb": pb, "peg": peg,
        "roe": roe * 100 if isinstance(roe, (int, float)) and abs(roe) < 5 else roe,
        "roa": roa * 100 if isinstance(roa, (int, float)) and abs(roa) < 5 else roa,
        "debt_to_equity": debt_to_equity,
        "current_ratio": current_ratio,
        "quick_ratio": quick_ratio,
        "dividend_yield_pct": (dividend_yield * 100) if isinstance(dividend_yield, (int, float)) else None,
    }


def fundamentals_available(f: Dict) -> bool:
    """Did we get enough data to score fundamentals at all?"""
    if not f:
        return False
    keys = ["revenue_growth_yoy_pct", "net_income", "pe", "ps", "roe",
            "gross_margin_pct", "free_cash_flow"]
    return sum(1 for k in keys if f.get(k) is not None) >= 3
