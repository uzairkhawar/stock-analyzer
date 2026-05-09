"""Stock screener with categorical classification."""
from __future__ import annotations
from typing import Dict, List, Optional
import pandas as pd
import config
from src.data_loader import DEFAULT_PROVIDER, DataProvider
from src.fundamentals import extract_fundamentals
from src.indicators import enrich, snapshot, relative_strength
from src.risk_management import build_setups, best_setup
from src.scoring import score_stock
from src.utils import logger, normalize_ticker, is_market_data_valid

S = config.SCREENER


def classify_candidate(snap, setups, df_enriched, score, avg_volume):
    """Categorical screener output."""
    REJ = config.REJECT
    if avg_volume is not None and avg_volume < REJ.min_avg_volume:
        return "Avoid - illiquid"
    acceptable = [s for s in setups if (getattr(s, "acceptable", False) or
                  (isinstance(s, dict) and s.get("acceptable")))]
    if not acceptable:
        rsi_v = snap.get("rsi")
        if rsi_v is not None and rsi_v < 30:
            return "Reversal watchlist"
        return "Avoid - poor risk/reward" if "Uptrend" in (snap.get("trend") or "") else \
               "Avoid - weak technicals"
    style = (acceptable[0].style if hasattr(acceptable[0], "style") else acceptable[0].get("style"))
    trend = (snap.get("trend") or "").lower()
    if style == "breakout": return "Breakout candidate"
    if style == "pullback": return "Pullback candidate"
    if "uptrend" in trend and (score or 0) >= 60: return "Strong trend candidate"
    return "Watchlist"


def passes_technical(snap, df_enriched):
    reasons = []
    px = snap.get("price"); s50 = snap.get("sma_50"); s200 = snap.get("sma_200")
    rsi = snap.get("rsi"); avg_v = (snap.get("volume") or {}).get("avg_volume_20")
    ok = True
    if avg_v is not None and avg_v < S.min_avg_volume:
        ok = False; reasons.append(f"Avg volume {avg_v:,.0f} below min.")
    if px is not None and (px < S.min_price or px > S.max_price):
        ok = False; reasons.append(f"Price {px} outside range.")
    if S.require_above_50_sma and s50 is not None and px is not None and px < s50:
        ok = False; reasons.append("Price below 50-SMA")
    if S.require_above_200_sma and s200 is not None and px is not None and px < s200:
        ok = False; reasons.append("Price below 200-SMA")
    if rsi is not None and not (S.rsi_min <= rsi <= S.rsi_max):
        ok = False; reasons.append(f"RSI {rsi:.1f} out of range.")
    return {"passes": ok, "reasons_failed": reasons}


def passes_relative_strength(stock_close, bench_close, min_rs=S.min_rel_strength):
    if bench_close is None: return {"passes": True, "rs": None}
    rs = relative_strength(stock_close, bench_close, S.rs_lookback_days)
    if rs is None: return {"passes": True, "rs": None}
    return {"passes": rs >= min_rs, "rs": rs}


def screen(tickers=None, market="US", provider=None, require_fundamentals=False,
           benchmark_ticker=None, progress_callback=None):
    provider = provider or DEFAULT_PROVIDER
    market_u = market.upper()
    tickers = tickers or config.DEFAULT_UNIVERSE.get(market_u, [])
    bench_sym = benchmark_ticker or config.DEFAULT_BENCHMARK_PER_MARKET.get(market_u, "^GSPC")
    bench_df = provider.fetch_ohlcv(bench_sym, period="1y", interval="1d")
    bench_close = bench_df["Close"] if isinstance(bench_df, pd.DataFrame) and not bench_df.empty else None

    rows = []
    total = len(tickers)
    for i, raw_t in enumerate(tickers, 1):
        t = normalize_ticker(raw_t, market_u)
        if progress_callback:
            try: progress_callback(i, total, t)
            except Exception: pass
        df = provider.fetch_ohlcv(t, period="2y", interval="1d")
        if not is_market_data_valid(df, min_rows=200):
            rows.append({"ticker": t, "score": None, "classification": "Skipped",
                         "category": "Skipped - data", "fail_reasons": "insufficient data"})
            continue
        try:
            df_e = enrich(df)
            snap = snapshot(df_e)
            rs_info = passes_relative_strength(df_e["Close"], bench_close)
            snap["relative_strength_pct"] = rs_info.get("rs")
            tech = passes_technical(snap, df_e)
            info = provider.fetch_info(t) if require_fundamentals else {}
            fin = provider.fetch_financials(t) if require_fundamentals else {}
            f = extract_fundamentals(info, fin) if require_fundamentals else None
            setups = build_setups(df_e, snap)
            best = best_setup(setups)
            best_d = best.to_dict() if best else None
            sc = score_stock(snap, f, best_d)
            avg_v = (snap.get("volume") or {}).get("avg_volume_20")
            category = classify_candidate(snap, setups, df_e, sc["total"], avg_v)
            rows.append({
                "ticker": t,
                "name": (f or {}).get("name") or info.get("longName") or "",
                "sector": (f or {}).get("sector") or info.get("sector"),
                "market_cap": (f or {}).get("market_cap") or info.get("marketCap"),
                "price": snap.get("price"), "rsi": snap.get("rsi"),
                "trend": snap.get("trend"),
                "rel_strength_pct": rs_info.get("rs"),
                "score": sc["total"], "classification": sc["classification"],
                "category": category, "confidence": sc["confidence"],
                "setup_style": best.style if best else None,
                "rr_tp1": best.risk_reward_tp1 if best else None,
                "pass_technical": tech["passes"],
                "fail_reasons": "; ".join(tech["reasons_failed"]) if tech["reasons_failed"] else "",
            })
        except Exception as e:
            logger.error("Screen error for %s: %s", t, e)
            rows.append({"ticker": t, "score": None, "classification": "Error",
                         "category": "Error", "fail_reasons": str(e)})

    df_out = pd.DataFrame(rows)
    if "score" in df_out.columns:
        df_out = df_out.sort_values(by=["score"], ascending=False, na_position="last")
    return df_out.reset_index(drop=True)
