"""Technical analysis engine - pure functions over OHLCV DataFrames."""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import config
from src.utils import logger

C = config.INDICATORS


def sma(series, period): return series.rolling(window=period, min_periods=period).mean()
def ema(series, period): return series.ewm(span=period, adjust=False, min_periods=period).mean()


def add_moving_averages(df):
    out = df.copy()
    close = out["Close"]
    for p in C.sma_periods: out[f"SMA_{p}"] = sma(close, p)
    for p in C.ema_periods: out[f"EMA_{p}"] = ema(close, p)
    return out


def trend_direction(df):
    last = df.iloc[-1]; px = last["Close"]
    s50 = last.get("SMA_50"); s200 = last.get("SMA_200")
    if pd.isna(s50) or pd.isna(s200): return "Insufficient data"
    if px > s50 > s200: return "Uptrend"
    if px < s50 < s200: return "Downtrend"
    if s50 > s200 and px < s50: return "Uptrend (pullback)"
    if s50 < s200 and px > s50: return "Downtrend (rally)"
    return "Sideways / mixed"


def golden_or_death_cross(df, lookback=60):
    if "SMA_50" not in df or "SMA_200" not in df: return None
    s50 = df["SMA_50"].dropna(); s200 = df["SMA_200"].dropna()
    common = s50.index.intersection(s200.index)
    if len(common) < lookback: return None
    a = s50.loc[common].iloc[-lookback:]; b = s200.loc[common].iloc[-lookback:]
    sign = np.sign(a - b); crosses = sign.diff().fillna(0)
    if (crosses > 0).any(): return "Golden cross (recent)"
    if (crosses < 0).any(): return "Death cross (recent)"
    return None


def rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs_v = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs_v))).fillna(50)


def macd(close, fast=12, slow=26, signal=9):
    f = ema(close, fast); s = ema(close, slow)
    line = f - s; sig = ema(line, signal); hist = line - sig
    return pd.DataFrame({"MACD": line, "MACD_Signal": sig, "MACD_Hist": hist})


def stochastic_rsi(close, period=14):
    r = rsi(close, period)
    lo = r.rolling(period).min(); hi = r.rolling(period).max()
    return ((r - lo) / (hi - lo).replace(0, np.nan)).fillna(0.5) * 100


def rate_of_change(close, period=12): return (close/close.shift(period) - 1.0) * 100.0


def obv(df):
    direction = np.sign(df["Close"].diff().fillna(0))
    return (direction * df["Volume"]).cumsum()


def accumulation_distribution(df):
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / rng
    mfv = mfm.fillna(0) * df["Volume"]
    return mfv.cumsum()


def volume_diagnostics(df):
    avg = df["Volume"].rolling(C.volume_avg_period).mean().iloc[-1]
    last = df["Volume"].iloc[-1]
    ratio = last / avg if avg and not pd.isna(avg) else np.nan
    breakout = bool(ratio is not np.nan and ratio >= C.volume_breakout_multiple)
    o = obv(df); obv_slope = o.diff(20).iloc[-1]
    return {
        "last_volume": float(last) if not pd.isna(last) else None,
        "avg_volume_20": float(avg) if not pd.isna(avg) else None,
        "volume_ratio": float(ratio) if not pd.isna(ratio) else None,
        "volume_breakout": breakout,
        "obv_trend": "rising" if obv_slope and obv_slope > 0 else ("falling" if obv_slope and obv_slope < 0 else "flat"),
    }


def atr(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    prev = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-prev).abs(), (low-prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()


def bollinger_bands(close, period=20, k=2.0):
    mid = close.rolling(period).mean(); std = close.rolling(period).std()
    return pd.DataFrame({"BB_Mid": mid, "BB_Upper": mid + k*std, "BB_Lower": mid - k*std})


def historical_volatility(close, period=30):
    rets = np.log(close/close.shift(1))
    return rets.rolling(period).std() * np.sqrt(252) * 100.0


def swing_pivots(df, lookback=20):
    highs, lows = [], []
    h, l = df["High"], df["Low"]; n = len(df)
    for i in range(lookback, n - lookback):
        if h.iloc[i] == h.iloc[i-lookback:i+lookback+1].max(): highs.append(float(h.iloc[i]))
        if l.iloc[i] == l.iloc[i-lookback:i+lookback+1].min(): lows.append(float(l.iloc[i]))
    return highs, lows


def nearest_levels(price, highs, lows):
    res_above = sorted([h for h in highs if h > price])
    sup_below = sorted([l for l in lows if l < price], reverse=True)
    return {
        "nearest_resistance": res_above[0] if res_above else None,
        "next_resistance": res_above[1] if len(res_above) > 1 else None,
        "third_resistance": res_above[2] if len(res_above) > 2 else None,
        "nearest_support": sup_below[0] if sup_below else None,
        "next_support": sup_below[1] if len(sup_below) > 1 else None,
    }


def market_structure(df, lookback=20):
    highs, lows = swing_pivots(df.tail(200), lookback=max(5, lookback//2))
    if len(highs) < 2 or len(lows) < 2: return "Insufficient structure"
    hh = highs[-1] > highs[-2]; hl = lows[-1] > lows[-2]
    lh = highs[-1] < highs[-2]; ll = lows[-1] < lows[-2]
    if hh and hl: return "Higher highs / higher lows (uptrend structure)"
    if lh and ll: return "Lower highs / lower lows (downtrend structure)"
    return "Mixed / consolidation"


def enrich(df):
    out = add_moving_averages(df)
    out["RSI"] = rsi(out["Close"], C.rsi_period)
    out = pd.concat([out, macd(out["Close"], C.macd_fast, C.macd_slow, C.macd_signal)], axis=1)
    out["StochRSI"] = stochastic_rsi(out["Close"], C.stoch_rsi_period)
    out["ATR"] = atr(out, C.atr_period)
    out = pd.concat([out, bollinger_bands(out["Close"], C.bb_period, C.bb_std)], axis=1)
    out["HV"] = historical_volatility(out["Close"], 30)
    out["OBV"] = obv(out); out["AD"] = accumulation_distribution(out)
    return out


def snapshot(df_enriched):
    if df_enriched is None or df_enriched.empty: return {}
    last = df_enriched.iloc[-1]; px = float(last["Close"])
    s = {
        "price": px,
        "sma_20": float(last.get("SMA_20", np.nan)),
        "sma_50": float(last.get("SMA_50", np.nan)),
        "sma_100": float(last.get("SMA_100", np.nan)),
        "sma_200": float(last.get("SMA_200", np.nan)),
        "ema_20": float(last.get("EMA_20", np.nan)),
        "rsi": float(last.get("RSI", np.nan)),
        "macd": float(last.get("MACD", np.nan)),
        "macd_signal": float(last.get("MACD_Signal", np.nan)),
        "macd_hist": float(last.get("MACD_Hist", np.nan)),
        "stoch_rsi": float(last.get("StochRSI", np.nan)),
        "atr": float(last.get("ATR", np.nan)),
        "bb_upper": float(last.get("BB_Upper", np.nan)),
        "bb_lower": float(last.get("BB_Lower", np.nan)),
        "hv": float(last.get("HV", np.nan)),
        "trend": trend_direction(df_enriched),
        "structure": market_structure(df_enriched),
        "cross": golden_or_death_cross(df_enriched),
    }
    s["volume"] = volume_diagnostics(df_enriched)
    highs, lows = swing_pivots(df_enriched, C.swing_lookback)
    s["levels"] = nearest_levels(px, highs, lows)
    return s


# ============================================================================
# Interpretation helpers (NEW in upgrade)
# ============================================================================
def is_overextended(snap, pct_vs_sma50=15.0, pct_vs_sma20=10.0):
    px = snap.get("price"); s20 = snap.get("sma_20"); s50 = snap.get("sma_50")
    out = {"overextended": False, "reasons": []}
    if px and s50 and not np.isnan(s50):
        d = (px / s50 - 1) * 100
        if d > pct_vs_sma50:
            out["overextended"] = True
            out["reasons"].append(f"Price {d:.1f}% above 50-SMA.")
    if px and s20 and not np.isnan(s20):
        d20 = (px / s20 - 1) * 100
        if d20 > pct_vs_sma20:
            out["overextended"] = True
            out["reasons"].append(f"Price {d20:.1f}% above 20-SMA.")
    return out


def breakout_quality(df_enriched, snap, lookback=20, vol_multiple=1.3):
    if df_enriched is None or len(df_enriched) < lookback + 5:
        return {"quality": "unknown", "score": 0, "reasons": ["Insufficient bars."]}
    close = df_enriched["Close"]; high = df_enriched["High"]; vol = df_enriched["Volume"]
    last_close = float(close.iloc[-1])
    prior_high = float(high.iloc[-(lookback+1):-1].max())
    avg_vol = float(vol.iloc[-(lookback+1):-1].mean())
    last_vol = float(vol.iloc[-1])
    breakout = last_close > prior_high
    vol_ratio = last_vol / avg_vol if avg_vol > 0 else 0.0
    atr_v = float(snap.get("atr") or 0)
    today_range = float(high.iloc[-1] - df_enriched["Low"].iloc[-1])
    expansion = (today_range / atr_v) if atr_v > 0 else 0.0
    prior_close = close.iloc[-(lookback+1):-1]
    compression = atr_v / float(prior_close.mean()) if prior_close.mean() else 0.0
    score = 0; reasons = []
    if breakout:
        score += 1; reasons.append(f"Closed above {lookback}-day high {prior_high:.2f}.")
        if vol_ratio >= vol_multiple:
            score += 2; reasons.append(f"Volume {vol_ratio:.2f}x avg confirms.")
        else:
            reasons.append(f"Volume {vol_ratio:.2f}x avg - does not confirm.")
        if expansion >= 1.2:
            score += 1; reasons.append(f"Range expansion {expansion:.2f}x ATR.")
    else:
        reasons.append("No breakout above recent high.")
    if compression and compression < 0.04:
        score += 1; reasons.append("Prior consolidation visible.")
    qual = "strong" if score >= 3 else ("weak" if score >= 1 else "none")
    return {"quality": qual, "score": int(score), "vol_ratio": vol_ratio,
            "broke_out": bool(breakout), "reasons": reasons}


def pullback_quality(df_enriched, snap):
    if df_enriched is None or len(df_enriched) < 60:
        return {"quality": "unknown", "score": 0, "reasons": ["Insufficient bars."]}
    close = df_enriched["Close"]; last = float(close.iloc[-1])
    s50 = snap.get("sma_50"); ema20 = snap.get("ema_20"); rsi_v = snap.get("rsi")
    score = 0; reasons = []
    if s50 and last > s50:
        score += 1; reasons.append("Price above 50-SMA.")
    else:
        return {"quality": "none", "score": 0, "reasons": ["Not in uptrend per 50-SMA."]}
    if ema20 and ema20 > 0:
        dist = abs(last/ema20 - 1) * 100
        if dist <= 2.0: score += 2; reasons.append(f"Within {dist:.2f}% of 20-EMA.")
        elif dist <= 4.0: score += 1; reasons.append(f"Near 20-EMA ({dist:.2f}%).")
    if rsi_v is not None and 40 <= rsi_v <= 60:
        score += 1; reasons.append(f"RSI {rsi_v:.0f} in pullback zone.")
    vol = df_enriched["Volume"]
    if len(vol) >= 25:
        if vol.iloc[-5:].mean() < vol.iloc[-25:-5].mean() * 0.9:
            score += 1; reasons.append("Volume contracted during pullback.")
    qual = "strong" if score >= 4 else ("acceptable" if score >= 2 else "weak")
    return {"quality": qual, "score": int(score), "reasons": reasons}


def false_breakout_risk(df_enriched, lookback=60):
    n = min(lookback, len(df_enriched) - 21)
    if n <= 0: return {"failed_count": 0, "risk": "unknown"}
    failures = 0
    for i in range(len(df_enriched) - n, len(df_enriched)):
        prior_high = df_enriched["High"].iloc[i-20:i].max()
        if df_enriched["High"].iloc[i] > prior_high and df_enriched["Close"].iloc[i] < prior_high:
            failures += 1
    risk = "high" if failures >= 4 else ("moderate" if failures >= 2 else "low")
    return {"failed_count": int(failures), "risk": risk}


def weekly_trend(df_daily):
    if df_daily is None or len(df_daily) < 100: return {"trend": "unknown"}
    wk = df_daily[["Open","High","Low","Close","Volume"]].resample("W").agg(
        {"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
    if len(wk) < 30: return {"trend": "unknown"}
    wk["SMA20"] = wk["Close"].rolling(20).mean()
    wk["SMA40"] = wk["Close"].rolling(40).mean()
    last = wk.iloc[-1]; px, s20, s40 = last["Close"], last["SMA20"], last["SMA40"]
    if pd.isna(s20) or pd.isna(s40): return {"trend": "unknown"}
    if px > s20 > s40: t = "Uptrend"
    elif px < s20 < s40: t = "Downtrend"
    elif s20 > s40: t = "Uptrend (pullback)"
    else: t = "Mixed"
    return {"trend": t, "weekly_close": float(px),
            "weekly_sma20": float(s20), "weekly_sma40": float(s40)}


def relative_strength(stock_close, bench_close, lookback=63):
    try:
        s = stock_close.dropna().iloc[-lookback:]
        b = bench_close.dropna().iloc[-lookback:]
        if len(s) < 5 or len(b) < 5: return None
        sr = (s.iloc[-1]/s.iloc[0] - 1) * 100
        br = (b.iloc[-1]/b.iloc[0] - 1) * 100
        return float(sr - br)
    except Exception as e:
        logger.warning("relative_strength failed: %s", e); return None
