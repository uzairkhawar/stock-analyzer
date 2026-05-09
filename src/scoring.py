"""Transparent 0–100 scoring engine + No-Trade decision."""
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import config
from src.fundamentals import fundamentals_available
from src.utils import logger

W = config.WEIGHTS


def score_trend(snap: Dict) -> Tuple[float, List[str]]:
    max_pts = W.technical_trend
    pts = 0.0
    reasons: List[str] = []
    px = snap.get("price"); s50 = snap.get("sma_50"); s200 = snap.get("sma_200"); ema20 = snap.get("ema_20")

    if px and s200 and not np.isnan(s200):
        if px > s200: pts += 6; reasons.append("Price above 200-SMA (long-term uptrend).")
        else: reasons.append("Price below 200-SMA (long-term downtrend).")
    if px and s50 and not np.isnan(s50):
        if px > s50: pts += 5; reasons.append("Price above 50-SMA.")
        else: reasons.append("Price below 50-SMA.")
    if s50 and s200 and not np.isnan(s50) and not np.isnan(s200):
        if s50 > s200: pts += 4; reasons.append("50-SMA above 200-SMA (bullish stack).")
        else: reasons.append("50-SMA below 200-SMA (bearish stack).")
    if px and ema20 and not np.isnan(ema20):
        if px > ema20: pts += 3; reasons.append("Price above 20-EMA.")

    trend = (snap.get("trend") or "").lower()
    if "uptrend" in trend and "pullback" not in trend:
        pts += 4; reasons.append(f"Trend: {snap.get('trend')}.")
    elif "uptrend (pullback)" in trend:
        pts += 2; reasons.append(f"Trend: {snap.get('trend')}.")
    elif "sideways" in trend:
        pts += 1; reasons.append(f"Trend: {snap.get('trend')}.")
    else:
        reasons.append(f"Trend: {snap.get('trend')}.")

    cross = snap.get("cross")
    if cross and "Golden" in cross:
        pts += 3; reasons.append("Recent golden cross.")
    elif cross and "Death" in cross:
        pts -= 3; reasons.append("Recent death cross.")

    return max(0.0, min(max_pts, pts)), reasons


def score_momentum(snap: Dict) -> Tuple[float, List[str]]:
    max_pts = W.momentum
    pts = 0.0
    reasons: List[str] = []
    rsi = snap.get("rsi")
    if rsi is not None and not np.isnan(rsi):
        if 45 <= rsi <= 70: pts += 6; reasons.append(f"RSI {rsi:.1f} healthy zone.")
        elif 40 <= rsi < 45 or 70 < rsi <= 75: pts += 3; reasons.append(f"RSI {rsi:.1f} borderline.")
        elif rsi > 75: pts += 1; reasons.append(f"RSI {rsi:.1f} overbought.")
        elif rsi < 30: pts += 1; reasons.append(f"RSI {rsi:.1f} oversold.")
        else: reasons.append(f"RSI {rsi:.1f} weak.")
    macd_v = snap.get("macd"); macd_s = snap.get("macd_signal"); macd_h = snap.get("macd_hist")
    if all(v is not None and not np.isnan(v) for v in (macd_v, macd_s, macd_h)):
        if macd_v > macd_s and macd_h > 0: pts += 5; reasons.append("MACD bullish.")
        elif macd_v > macd_s: pts += 3; reasons.append("MACD turning up.")
        else: reasons.append("MACD bearish.")
    sr = snap.get("stoch_rsi")
    if sr is not None and not np.isnan(sr):
        if 20 <= sr <= 80: pts += 2; reasons.append(f"StochRSI {sr:.0f} neutral.")
        elif sr > 80: pts += 1; reasons.append(f"StochRSI {sr:.0f} overbought.")
        elif sr < 20: pts += 1; reasons.append(f"StochRSI {sr:.0f} oversold.")
    rs = snap.get("relative_strength_pct")
    if rs is not None:
        if rs > 5: pts += 2; reasons.append(f"RS +{rs:.1f}% vs benchmark.")
        elif rs < -5: reasons.append(f"RS {rs:.1f}% vs benchmark.")
    return max(0.0, min(max_pts, pts)), reasons


def score_volume(snap: Dict) -> Tuple[float, List[str]]:
    max_pts = W.volume
    pts = 0.0
    reasons: List[str] = []
    v = snap.get("volume", {}) or {}
    ratio = v.get("volume_ratio")
    if ratio is not None:
        if ratio >= 1.5: pts += 6; reasons.append(f"Volume {ratio:.2f}x avg.")
        elif ratio >= 1.0: pts += 4; reasons.append(f"Volume {ratio:.2f}x avg.")
        elif ratio >= 0.7: pts += 2; reasons.append(f"Volume {ratio:.2f}x avg.")
        else: reasons.append(f"Volume {ratio:.2f}x avg (weak).")
    if v.get("volume_breakout"): pts += 3; reasons.append("Volume breakout.")
    obv = v.get("obv_trend")
    if obv == "rising": pts += 4; reasons.append("OBV rising.")
    elif obv == "falling": reasons.append("OBV falling.")
    else: pts += 2; reasons.append("OBV flat.")
    return max(0.0, min(max_pts, pts)), reasons


def score_fundamental(f: Optional[Dict]) -> Tuple[float, List[str]]:
    max_pts = W.fundamental
    if not f or not fundamentals_available(f):
        return 0.0, ["Fundamentals unavailable."]
    pts = 0.0; reasons: List[str] = []
    rg = f.get("revenue_growth_yoy_pct")
    if rg is not None:
        if rg > 15: pts += 5; reasons.append(f"Rev growth {rg:.1f}%.")
        elif rg > 5: pts += 3; reasons.append(f"Rev growth {rg:.1f}%.")
        elif rg > 0: pts += 1; reasons.append(f"Rev growth {rg:.1f}%.")
        else: reasons.append(f"Rev declining {rg:.1f}%.")
    nig = f.get("net_income_growth_yoy_pct")
    if nig is not None:
        if nig > 15: pts += 4; reasons.append(f"NI growth {nig:.1f}%.")
        elif nig > 0: pts += 2; reasons.append(f"NI growth {nig:.1f}%.")
        else: reasons.append(f"NI declining {nig:.1f}%.")
    gm = f.get("gross_margin_pct")
    if gm is not None:
        if gm > 50: pts += 3; reasons.append(f"GM {gm:.1f}%.")
        elif gm > 30: pts += 2; reasons.append(f"GM {gm:.1f}%.")
        elif gm > 15: pts += 1; reasons.append(f"GM {gm:.1f}%.")
    nm = f.get("net_margin_pct")
    if nm is not None:
        if nm > 15: pts += 2; reasons.append(f"NM {nm:.1f}%.")
        elif nm > 5: pts += 1; reasons.append(f"NM {nm:.1f}%.")
        elif nm < 0: reasons.append(f"NM {nm:.1f}%.")
    roe = f.get("roe")
    if roe is not None:
        if roe > 15: pts += 3; reasons.append(f"ROE {roe:.1f}%.")
        elif roe > 8: pts += 2; reasons.append(f"ROE {roe:.1f}%.")
        elif roe < 0: reasons.append(f"ROE {roe:.1f}%.")
    de = f.get("debt_to_equity")
    if de is not None:
        if de < 50: pts += 3; reasons.append(f"D/E {de:.0f}.")
        elif de < 100: pts += 2; reasons.append(f"D/E {de:.0f}.")
        elif de < 200: pts += 1; reasons.append(f"D/E {de:.0f}.")
        else: reasons.append(f"D/E {de:.0f} high.")
    fcf = f.get("free_cash_flow")
    if fcf is not None:
        if fcf > 0: pts += 3; reasons.append("Positive FCF.")
        else: reasons.append("Negative FCF.")
    cr = f.get("current_ratio")
    if cr is not None:
        if cr >= 1.5: pts += 2; reasons.append(f"CR {cr:.2f}.")
        elif cr >= 1: pts += 1; reasons.append(f"CR {cr:.2f}.")
        else: reasons.append(f"CR {cr:.2f} low.")
    return max(0.0, min(max_pts, pts)), reasons


def score_valuation(f: Optional[Dict]) -> Tuple[float, List[str]]:
    max_pts = W.valuation
    if not f or not fundamentals_available(f):
        return 0.0, ["Valuation skipped."]
    pts = 0.0; reasons: List[str] = []
    pe = f.get("pe")
    if pe is not None and pe > 0:
        if pe < 15: pts += 3; reasons.append(f"P/E {pe:.1f}.")
        elif pe < 25: pts += 2; reasons.append(f"P/E {pe:.1f}.")
        elif pe < 40: pts += 1; reasons.append(f"P/E {pe:.1f}.")
        else: reasons.append(f"P/E {pe:.1f} expensive.")
    ps = f.get("ps")
    if ps is not None and ps > 0:
        if ps < 2: pts += 2; reasons.append(f"P/S {ps:.2f}.")
        elif ps < 5: pts += 1; reasons.append(f"P/S {ps:.2f}.")
        else: reasons.append(f"P/S {ps:.2f}.")
    pb = f.get("pb")
    if pb is not None and pb > 0:
        if pb < 1.5: pts += 2; reasons.append(f"P/B {pb:.2f}.")
        elif pb < 4: pts += 1; reasons.append(f"P/B {pb:.2f}.")
        else: reasons.append(f"P/B {pb:.2f}.")
    peg = f.get("peg")
    if peg is not None and peg > 0:
        if peg < 1: pts += 3; reasons.append(f"PEG {peg:.2f}.")
        elif peg < 2: pts += 1; reasons.append(f"PEG {peg:.2f}.")
        else: reasons.append(f"PEG {peg:.2f}.")
    return max(0.0, min(max_pts, pts)), reasons


def score_risk_reward(best_setup: Optional[Dict]) -> Tuple[float, List[str]]:
    max_pts = W.risk_reward
    if not best_setup:
        return 0.0, ["No actionable setup."]
    rr1 = best_setup.get("risk_reward_tp1") or 0
    rr2 = best_setup.get("risk_reward_tp2") or 0
    pts = 0.0; reasons: List[str] = []
    if rr1 >= 3: pts += 6; reasons.append(f"R:R TP1 {rr1:.2f}.")
    elif rr1 >= 2: pts += 5; reasons.append(f"R:R TP1 {rr1:.2f}.")
    elif rr1 >= 1.5: pts += 3; reasons.append(f"R:R TP1 {rr1:.2f}.")
    else: reasons.append(f"R:R TP1 {rr1:.2f} poor.")
    if rr2 >= 3: pts += 4; reasons.append(f"R:R TP2 {rr2:.2f}.")
    elif rr2 >= 2: pts += 2; reasons.append(f"R:R TP2 {rr2:.2f}.")
    return max(0.0, min(max_pts, pts)), reasons


def classify(score: float) -> str:
    th = config.CLASSIFICATION_THRESHOLDS
    if score >= th["Strong Candidate"]: return "Strong Candidate"
    if score >= th["Watchlist Candidate"]: return "Watchlist Candidate"
    if score >= th["Neutral"]: return "Neutral"
    if score >= th["Weak"]: return "Weak"
    return "Avoid"


def confidence_from_components(components: Dict[str, float]) -> str:
    max_map = {"trend": W.technical_trend, "momentum": W.momentum, "volume": W.volume,
               "fundamental": W.fundamental, "valuation": W.valuation, "risk_reward": W.risk_reward}
    ratios = [components[k] / max_map[k] for k in components if max_map.get(k)]
    strong = sum(1 for r in ratios if r >= 0.6)
    if strong >= 5: return "High"
    if strong >= 3: return "Medium"
    return "Low"


def decide_action(*, score_total: float, classification: str,
                  best_setup: Optional[Dict], setups: List[Dict],
                  snap: Dict, market_condition: str,
                  fundamentals_available: bool,
                  history_bars: int) -> Dict:
    """Honest gate: returns {'action', 'tradable', 'reasons'}."""
    reasons: List[str] = []
    REJ = config.REJECT
    if history_bars < REJ.min_history_bars:
        return {"action": "No Trade - missing data", "tradable": False,
                "reasons": [f"Only {history_bars} bars."]}
    avg_v = (snap.get("volume") or {}).get("avg_volume_20")
    if avg_v is not None and avg_v < REJ.min_avg_volume:
        return {"action": "No Trade - illiquid", "tradable": False,
                "reasons": [f"Avg volume {avg_v:,.0f} below {REJ.min_avg_volume:,}."]}
    rs = snap.get("relative_strength_pct")
    weak_market = "Bearish" in (market_condition or "") or "Weak" in (market_condition or "")
    if weak_market and (rs is None or rs < REJ.weak_market_rs_required_pct):
        reasons.append(f"Broad market: {market_condition}.")
        if rs is not None:
            reasons.append(f"RS {rs:.1f}% < required {REJ.weak_market_rs_required_pct}%.")
        return {"action": "No Trade - weak market", "tradable": False, "reasons": reasons}
    if not best_setup:
        return {"action": "No Trade - no clean setup", "tradable": False,
                "reasons": ["No setup met geometry/quality requirements."]}
    rej = best_setup.get("rejection_reasons") or []
    if rej:
        joined = " | ".join(rej)
        if "Overextended" in joined:
            return {"action": "No Trade - overextended", "tradable": False, "reasons": rej}
        if "Entry within" in joined:
            return {"action": "No Trade - entry too close to resistance", "tradable": False, "reasons": rej}
        if "Stop too wide" in joined:
            return {"action": "No Trade - poor risk/reward", "tradable": False, "reasons": rej}
        if "below min" in joined:
            return {"action": "No Trade - poor risk/reward", "tradable": False, "reasons": rej}
        if "RSI" in joined and "overbought" in joined:
            return {"action": "Wait for pullback", "tradable": False, "reasons": rej}
        if "Breakout not volume" in joined:
            return {"action": "Wait for breakout confirmation", "tradable": False, "reasons": rej}
        return {"action": "No Trade - setup rejected", "tradable": False, "reasons": rej}
    style = best_setup.get("style", "")
    if style == "breakout":
        action = "Trade - breakout (confirm with volume)"
    elif style == "pullback":
        action = "Trade - pullback to support"
    elif style == "conservative":
        action = "Wait for breakout confirmation (then retest)"
    elif style == "aggressive":
        action = "Trade - aggressive (smaller size)"
    else:
        action = "Trade"
    if classification in ("Weak", "Avoid"):
        return {"action": "No Trade - score too low", "tradable": False,
                "reasons": [f"Classification '{classification}'."]}
    if classification == "Neutral":
        return {"action": "Wait - neutral score", "tradable": False,
                "reasons": ["Score in neutral band."]}
    if not fundamentals_available:
        reasons.append("Fundamentals unavailable - confidence reduced.")
    return {"action": action, "tradable": True, "reasons": reasons or ["All gates passed."]}


def score_stock(snap: Dict, fundamentals: Optional[Dict],
                best_setup: Optional[Dict]) -> Dict:
    t_pts, t_r  = score_trend(snap)
    m_pts, m_r  = score_momentum(snap)
    v_pts, v_r  = score_volume(snap)
    f_pts, f_r  = score_fundamental(fundamentals)
    val_pts, val_r = score_valuation(fundamentals)
    rr_pts, rr_r = score_risk_reward(best_setup)
    components = {"trend": t_pts, "momentum": m_pts, "volume": v_pts,
                  "fundamental": f_pts, "valuation": val_pts, "risk_reward": rr_pts}
    total = sum(components.values())
    return {
        "components": components,
        "max_components": {"trend": W.technical_trend, "momentum": W.momentum, "volume": W.volume,
                           "fundamental": W.fundamental, "valuation": W.valuation, "risk_reward": W.risk_reward},
        "total": round(total, 1),
        "classification": classify(total),
        "confidence": confidence_from_components(components),
        "reasons": {"trend": t_r, "momentum": m_r, "volume": v_r,
                    "fundamental": f_r, "valuation": val_r, "risk_reward": rr_r},
    }
