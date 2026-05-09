"""
src/signal_engine.py - Final decision-support signal engine.

Maps the analysis (score, decision, snapshot, market condition) plus an optional
backtest 'overall' summary into one of seven signals (closed set):

    Strong Candidate
    Watchlist Candidate
    Wait for Breakout
    Wait for Pullback
    No Trade
    Avoid
    High Risk

Each signal carries: action_guidance, entry/SL/TPs (when applicable),
risk_reward, confidence (Low/Medium/High), setup_status (Acceptable / Not Acceptable),
why bullets, invalidation level, and a list of risk_warnings.

Honesty rules
-------------
- Never produces "Buy" or "Sell" labels. Never claims certainty or guarantees.
- Backtest quality only DOWNGRADES confidence; it never inflates a signal beyond
  what the underlying score and setup geometry support (the user's actual stock
  is not the backtest sample - so backtest is supportive evidence at best).
- Missing data, illiquidity, or weak market routes immediately to Avoid.
- Extreme realized volatility routes a would-be candidate to High Risk.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple


SIGNALS: Tuple[str, ...] = (
    "Strong Candidate",
    "Watchlist Candidate",
    "Wait for Breakout",
    "Wait for Pullback",
    "No Trade",
    "Avoid",
    "High Risk",
)

_CONF: Tuple[str, ...] = ("Low", "Medium", "High")


def _step_down(conf: str, steps: int = 1) -> str:
    try:
        i = max(0, _CONF.index(conf) - steps)
    except ValueError:
        return "Low"
    return _CONF[i]


def _detect_high_risk(snap: Dict) -> List[str]:
    warns: List[str] = []
    hv = snap.get("hv")
    if hv is not None and hv == hv and hv > 60:
        warns.append(f"Annualized HV {hv:.0f}% (high volatility).")
    rsi = snap.get("rsi")
    if rsi is not None and rsi == rsi and rsi > 80:
        warns.append(f"RSI {rsi:.0f} extreme.")
    avg_v = (snap.get("volume") or {}).get("avg_volume_20")
    if avg_v is not None and avg_v < 200_000:
        warns.append(f"Average volume {avg_v:,.0f} below liquidity floor.")
    return warns


def _signal_from_decision(decision: Dict, classification: str, tradable: bool) -> str:
    """Translate decision.action + classification into a 7-signal label."""
    action = (decision or {}).get("action", "") or ""
    al = action.lower()
    if "missing data" in al:               return "Avoid"
    if "illiquid" in al:                   return "Avoid"
    if "weak market" in al:                return "Avoid"
    if "score too low" in al:              return "Avoid"
    if "wait for breakout" in al:          return "Wait for Breakout"
    if "wait for pullback" in al:          return "Wait for Pullback"
    if "overextended" in al:               return "Wait for Pullback"
    if "near resistance" in al or "too close to resistance" in al:
                                            return "Wait for Pullback"
    if "no trade" in al or al.startswith("wait"):
                                            return "No Trade"
    # Tradable
    if tradable and classification == "Strong Candidate":
        return "Strong Candidate"
    if tradable and classification == "Watchlist Candidate":
        return "Watchlist Candidate"
    return "No Trade"


def _confidence_baseline(score_total: float, fundamentals_available: bool) -> str:
    if score_total >= 75 and fundamentals_available:
        return "High"
    if score_total >= 60:
        return "Medium"
    return "Low"


def _apply_backtest_adjustment(
    signal: str, confidence: str,
    bt_overall: Optional[Dict],
    warnings: List[str],
) -> Tuple[str, str]:
    """Apply backtest evidence: only DOWNGRADES, never upgrades."""
    if not bt_overall:
        warnings.append("No backtest evidence supplied.")
        return signal, _step_down(confidence)

    n = int(bt_overall.get("num_trades", 0) or 0)
    exp = bt_overall.get("expectancy_R", 0) or 0
    pf_raw = bt_overall.get("profit_factor", 0)
    pf = pf_raw if isinstance(pf_raw, (int, float)) else 0.0
    dd = bt_overall.get("max_drawdown_R", 0) or 0

    if n == 0:
        warnings.append("Backtest produced no trades.")
        return signal, _step_down(confidence)

    if n < 20:
        warnings.append(f"Backtest sample small ({n} trades).")
        confidence = _step_down(confidence)

    # HARD downgrade: negative expectancy or PF < 1.0 -> Avoid
    if exp <= 0 or (pf and pf < 1.0):
        warnings.append(f"Backtest unprofitable (exp={exp}R, PF={pf}). "
                        "Strategy not validated for this market.")
        return "Avoid", "Low"
    if exp < 0:
        warnings.append(f"Backtest expectancy negative ({exp}R).")
        if signal == "Strong Candidate":
            signal = "Watchlist Candidate"
        confidence = "Low"
    elif exp < 0.10:
        warnings.append(f"Backtest expectancy marginal ({exp:.2f}R).")
        confidence = _step_down(confidence)

    if pf and pf < 1.2:
        warnings.append(f"Backtest profit factor low ({pf}).")
        confidence = _step_down(confidence)

    if dd and dd > 10:
        warnings.append(f"Backtest max drawdown {dd}R is high.")
        confidence = _step_down(confidence)

    return signal, confidence


def compute_signal(*, analysis: Dict,
                   backtest_overall: Optional[Dict] = None) -> Dict:
    """
    Build the final decision-support signal from an analysis result
    (output of analyze_stock) and an optional backtest 'overall' summary.

    Returns a dict with keys:
        signal, action_guidance, confidence, setup_status,
        why (list), risk_warnings (list),
        entry, stop_loss, tp1, tp2, tp3, risk_reward_tp1, invalidation
        (the trade-geometry keys are present only if a setup was selected).
    """
    if not analysis or not analysis.get("ok"):
        return {
            "signal": "Avoid",
            "action_guidance": (analysis or {}).get("error", "Insufficient data."),
            "confidence": "Low",
            "setup_status": "Not Acceptable",
            "why": ["Analysis could not be completed."],
            "risk_warnings": ["missing data"],
        }

    snap = analysis["snapshot"] or {}
    score = analysis["score"] or {}
    decision = analysis.get("decision") or {}
    best = analysis.get("best_setup")
    cls = score.get("classification", "Neutral")
    score_total = score.get("total", 0)

    risk_warnings = _detect_high_risk(snap)
    why: List[str] = []
    market = (analysis.get("market") or "US").upper()
    if backtest_overall:
        exp = backtest_overall.get("expectancy_R", 0) or 0
        pf = backtest_overall.get("profit_factor", 0)
        pf_v = pf if isinstance(pf, (int, float)) else 0
        if exp <= 0 or pf_v < 1.0:
            risk_warnings.append(
                f"{market}: Backtest failed; do not use live without optimization.")
        else:
            risk_warnings.append(f"{market}: Backtest promising but preliminary.")

    base_signal = _signal_from_decision(decision, cls, decision.get("tradable", False))

    # Extreme volatility routes candidates to High Risk
    hv = snap.get("hv")
    if hv is not None and hv == hv and hv > 80 and base_signal in ("Strong Candidate", "Watchlist Candidate"):
        base_signal = "High Risk"
        why.append(f"Realized volatility {hv:.0f}% - too volatile for a clean signal.")

    confidence = _confidence_baseline(score_total, analysis.get("fundamentals_available", False))
    base_signal, confidence = _apply_backtest_adjustment(
        base_signal, confidence, backtest_overall, risk_warnings,
    )

    # Action guidance
    levels = snap.get("levels", {}) or {}
    res1 = levels.get("nearest_resistance")
    sup1 = levels.get("nearest_support")
    ema20 = snap.get("ema_20")
    sma50 = snap.get("sma_50")

    if base_signal == "Wait for Breakout":
        guidance = (f"Wait for daily close above {res1:.2f} on rising volume."
                    if res1 else "Wait for breakout above nearest resistance.")
    elif base_signal == "Wait for Pullback":
        if ema20 and ema20 == ema20:
            guidance = f"Wait for pullback toward {ema20:.2f} (20-EMA)."
        elif sma50 and sma50 == sma50:
            guidance = f"Wait for pullback toward {sma50:.2f} (50-SMA)."
        else:
            guidance = "Wait for pullback toward a moving-average support."
    elif base_signal == "Strong Candidate":
        guidance = ("Setup geometry acceptable. User decision required - "
                    "size with disciplined account risk and respect the stop.")
    elif base_signal == "Watchlist Candidate":
        guidance = "Add to watchlist; await trigger or better setup geometry."
    elif base_signal == "Avoid":
        guidance = "Do not enter; one or more hard gates failed (data, liquidity, or market)."
    elif base_signal == "High Risk":
        guidance = ("Do not size up. Volatility, data quality, or liquidity "
                    "elevate risk beyond a normal swing setup.")
    else:
        guidance = "No clean setup. Wait for confirmation or skip."

    for r in (decision.get("reasons") or [])[:3]:
        why.append(r)
    if not why:
        why.append("See per-component reasons in the scorecard.")

    # Override setup_status / guidance / why for non-tradable signals
    if base_signal == "Avoid":
        setup_status = "Rejected"
        guidance = "Do not enter. Market/backtest conditions failed."
        rej_reason = next((w for w in risk_warnings
                           if "backtest" in w.lower() or "weak" in w.lower()
                           or "illiquid" in w.lower() or "missing" in w.lower()), None)
        if rej_reason and rej_reason not in why:
            why = [rej_reason] + why
    elif base_signal == "No Trade":
        setup_status = "Not Acceptable"
        guidance = "Do not enter."
    else:
        setup_status = "Acceptable" if (best and best.get("acceptable")) else "Not Acceptable"

    out = {
        "signal": base_signal,
        "action_guidance": guidance,
        "confidence": confidence,
        "setup_status": setup_status,
        "why": why,
        "risk_warnings": risk_warnings,
    }

    # Price-vs-entry interpretation: if entry is above current, it's a trigger
    cur_px = snap.get("price")
    if best and cur_px and best.get("entry") and cur_px < best["entry"]:
        out["action_guidance"] = (
            f"Wait for breakout above entry level {best['entry']:.2f}."
        )
        out["entry_label"] = "Breakout trigger"
    elif best and cur_px and best.get("entry"):
        out["entry_label"] = "Entry now"

    # Momentum / volume confirmation warnings
    rsi_v = snap.get("rsi")
    if rsi_v is not None and rsi_v == rsi_v and rsi_v < 50:
        risk_warnings.append("Momentum not confirmed yet (RSI < 50).")
    vol_ratio = (snap.get("volume") or {}).get("volume_ratio")
    if vol_ratio is not None and vol_ratio < 1.0:
        risk_warnings.append("Breakout requires volume confirmation (vol < avg).")
    out["risk_warnings"] = risk_warnings

    if best and base_signal not in ("Avoid", "No Trade"):
        out.update({
            "entry": best.get("entry"),
            "stop_loss": best.get("stop_loss"),
            "tp1": best.get("tp1"),
            "tp2": best.get("tp2"),
            "tp3": best.get("tp3"),
            "risk_reward_tp1": best.get("risk_reward_tp1"),
            "invalidation": best.get("invalidation"),
        })
    elif best and base_signal in ("Avoid", "No Trade"):
        out["reference_levels_only"] = {
            "entry": best.get("entry"),
            "stop_loss": best.get("stop_loss"),
            "tp1": best.get("tp1"),
            "tp2": best.get("tp2"),
            "tp3": best.get("tp3"),
            "note": "Reference levels only - NOT trade levels. Signal is "
                    f"{base_signal}.",
        }
    if sup1 and "invalidation" not in out:
        out["invalidation"] = f"Daily close below {sup1:.2f}."

    return out


def to_markdown(sig: Dict) -> str:
    """Render the signal as a compact markdown block for inclusion in reports."""
    if not sig:
        return ""
    lines = [
        "## Final Signal",
        f"- **Signal:** {sig.get('signal')}",
        f"- **Action Guidance:** {sig.get('action_guidance')}",
        f"- **Confidence:** {sig.get('confidence')}",
        f"- **Setup Status:** {sig.get('setup_status')}",
    ]
    if sig.get("entry") is not None:
        lines += [
            f"- **Entry:** {sig.get('entry')}  |  **Stop:** {sig.get('stop_loss')}",
            f"- **TP1:** {sig.get('tp1')}  |  **TP2:** {sig.get('tp2')}  |  **TP3:** {sig.get('tp3')}",
            f"- **R:R TP1:** {sig.get('risk_reward_tp1')}",
        ]
    if sig.get("invalidation"):
        lines.append(f"- **Invalidation:** {sig['invalidation']}")
    if sig.get("why"):
        lines.append("- **Why:**")
        for w in sig["why"]:
            lines.append(f"  - {w}")
    if sig.get("risk_warnings"):
        lines.append("- **Risk Warnings:**")
        for w in sig["risk_warnings"]:
            lines.append(f"  - {w}")
    return "\n".join(lines)
