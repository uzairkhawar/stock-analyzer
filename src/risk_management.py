"""Risk management & trade-setup geometry with realistic rejection rules."""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import config
from src.utils import logger

R = config.RISK


@dataclass
class TradeSetup:
    style: str
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: Optional[float]
    risk_per_share: float
    reward_per_share_tp1: float
    risk_reward_tp1: float
    risk_reward_tp2: float
    rationale: str
    invalidation: str
    acceptable: bool
    rejection_reasons: List[str] = field(default_factory=list)
    stop_width_pct: float = 0.0

    def to_dict(self) -> Dict: return asdict(self)


def _structure_stop(df, lookback=20):
    if df is None or df.empty: return None
    return float(df["Low"].iloc[-lookback:].min())


def _atr_stop(price, atr_value, mult=R.atr_stop_multiplier):
    return float(price - mult * atr_value)


def _evaluate_rejections(*, style, entry, sl, tp1, rr1, stop_width_pct, snap,
                         df_enriched, nearest_res, atr_v):
    REJ = config.REJECT
    reasons: List[str] = []
    if rr1 < R.min_risk_reward:
        reasons.append(f"R:R to TP1 = {rr1:.2f} below min {R.min_risk_reward}.")
    if stop_width_pct > REJ.max_stop_width_pct:
        reasons.append(f"Stop too wide: {stop_width_pct:.1f}% (max {REJ.max_stop_width_pct}%).")
    if style in ("breakout", "aggressive") and nearest_res and atr_v and not np.isnan(atr_v):
        gap = nearest_res - entry
        if 0 < gap < REJ.resistance_proximity_atr * atr_v:
            reasons.append(f"Entry within {gap:.2f} of resistance ({nearest_res:.2f}).")
    px = snap.get("price"); s50 = snap.get("sma_50"); s20 = snap.get("sma_20")
    if px and s50 and not np.isnan(s50):
        ext50 = (px / s50 - 1) * 100
        if ext50 > REJ.overextension_pct_vs_sma50 and style in ("breakout", "aggressive"):
            reasons.append(f"Overextended: price {ext50:.1f}% above 50-SMA.")
    if px and s20 and not np.isnan(s20) and style == "aggressive":
        ext20 = (px / s20 - 1) * 100
        if ext20 > REJ.overextension_pct_vs_sma20:
            reasons.append(f"Overextended: price {ext20:.1f}% above 20-SMA.")
    rsi_v = snap.get("rsi")
    if rsi_v is not None and rsi_v > REJ.rsi_extreme_high and style in ("breakout", "aggressive"):
        reasons.append(f"RSI {rsi_v:.0f} extremely overbought.")
    if style == "breakout":
        v = (snap.get("volume") or {})
        ratio = v.get("volume_ratio")
        if ratio is not None and ratio < 1.0:
            reasons.append(f"Breakout not volume-confirmed (vol ratio {ratio:.2f}).")
    if df_enriched is not None and len(df_enriched) < REJ.min_history_bars:
        reasons.append(f"Only {len(df_enriched)} bars of history.")
    return reasons


def build_setups(df_enriched, snap):
    if df_enriched is None or df_enriched.empty: return []
    price = float(snap["price"])
    atr_v = float(snap.get("atr") or np.nan)
    levels = snap.get("levels", {}) or {}
    nearest_res = levels.get("nearest_resistance")
    next_res = levels.get("next_resistance")
    third_res = levels.get("third_resistance")
    n = R.breakout_lookback
    recent_high = float(df_enriched["High"].iloc[-n:].max())
    ema20 = float(snap.get("ema_20") or df_enriched["EMA_20"].iloc[-1])
    structure_stop = _structure_stop(df_enriched, lookback=n) or (price * 0.92)
    setups: List[TradeSetup] = []

    def _make(style, entry, sl, rationale, invalidation):
        if entry <= 0 or sl <= 0 or sl >= entry: return None
        risk = entry - sl
        tp1 = nearest_res if (nearest_res and nearest_res > entry) else entry + 2 * risk
        tp2 = next_res if (next_res and next_res > tp1) else entry + 3 * risk
        tp3 = third_res if (third_res and third_res > tp2) else entry + 5 * risk
        rr1 = (tp1 - entry) / risk if risk > 0 else 0
        rr2 = (tp2 - entry) / risk if risk > 0 else 0
        stop_width = (risk / entry) * 100 if entry else 0
        rejections = _evaluate_rejections(
            style=style, entry=entry, sl=sl, tp1=tp1, rr1=rr1,
            stop_width_pct=stop_width, snap=snap,
            df_enriched=df_enriched, nearest_res=nearest_res, atr_v=atr_v,
        )
        return TradeSetup(
            style=style, entry=round(entry,2), stop_loss=round(sl,2),
            tp1=round(tp1,2), tp2=round(tp2,2),
            tp3=round(tp3,2) if tp3 else None,
            risk_per_share=round(risk,2),
            reward_per_share_tp1=round(tp1-entry,2),
            risk_reward_tp1=round(rr1,2), risk_reward_tp2=round(rr2,2),
            rationale=rationale, invalidation=invalidation,
            acceptable=(rr1 >= R.min_risk_reward) and not rejections,
            rejection_reasons=rejections, stop_width_pct=round(stop_width,2),
        )

    if recent_high > price:
        bo_entry = recent_high * 1.005
        sl = max(structure_stop, _atr_stop(bo_entry, atr_v) if not np.isnan(atr_v) else 0)
        s = _make("breakout", bo_entry, sl,
                  f"Breakout entry above {n}-day high ({recent_high:.2f}).",
                  f"Daily close back below {recent_high:.2f} or below {round(sl,2)}.")
        if s: setups.append(s)

    if not np.isnan(ema20) and price > ema20 * 0.97:
        pb_entry = ema20
        sl_atr = _atr_stop(pb_entry, atr_v) if not np.isnan(atr_v) else pb_entry * 0.95
        sl = min(sl_atr, structure_stop)
        s = _make("pullback", pb_entry, sl,
                  "Pullback entry near 20-EMA.",
                  f"Daily close below {round(sl,2)}.")
        if s: setups.append(s)

    if nearest_res and nearest_res > price:
        cons_entry = nearest_res * 1.002
        sl = max(structure_stop, _atr_stop(cons_entry, atr_v) if not np.isnan(atr_v) else cons_entry * 0.95)
        s = _make("conservative", cons_entry, sl,
                  "Wait for breakout above nearest resistance and retest.",
                  f"Failed retest below {round(sl,2)}.")
        if s: setups.append(s)

    rsi_v = snap.get("rsi"); trend = snap.get("trend", "")
    if "Uptrend" in (trend or "") and rsi_v and 45 <= rsi_v <= 70:
        agg_entry = price
        sl_atr = _atr_stop(agg_entry, atr_v) if not np.isnan(atr_v) else agg_entry * 0.95
        sl = max(structure_stop, sl_atr)
        s = _make("aggressive", agg_entry, sl,
                  "Market entry - trend up, momentum healthy.",
                  f"Close below {round(sl,2)}.")
        if s: setups.append(s)

    return setups


def best_setup(setups):
    candidates = [s for s in setups if s.acceptable]
    if not candidates:
        return max(setups, key=lambda s: s.risk_reward_tp1) if setups else None
    style_rank = {"pullback": 4, "breakout": 3, "conservative": 2, "aggressive": 1}
    candidates.sort(key=lambda s: (s.risk_reward_tp1, style_rank.get(s.style, 0)), reverse=True)
    return candidates[0]


def position_size(account_equity, setup, risk_pct=R.max_account_risk_pct):
    risk_dollars = account_equity * (risk_pct / 100.0)
    shares = int(risk_dollars // max(setup.risk_per_share, 0.01))
    notional = shares * setup.entry
    return {
        "account_equity": account_equity, "risk_pct": risk_pct,
        "risk_dollars": round(risk_dollars, 2),
        "risk_per_share": setup.risk_per_share,
        "shares": shares, "notional": round(notional, 2),
        "exposure_pct": round((notional/account_equity) * 100, 2) if account_equity else None,
    }


def risk_summary(setups):
    if not setups: return {"acceptable_count": 0, "best": None}
    best = best_setup(setups)
    return {
        "count": len(setups),
        "acceptable_count": sum(1 for s in setups if s.acceptable),
        "best": best.to_dict() if best else None,
    }
