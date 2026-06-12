"""Report generator with No-Trade decision and backtest summary integration."""
from __future__ import annotations
import io
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import config
from src.data_loader import load_stock, DataProvider, DEFAULT_PROVIDER
from src.fundamentals import extract_fundamentals, fundamentals_available
from src.indicators import enrich, snapshot, relative_strength
from src.risk_management import build_setups, best_setup, position_size
from src.scoring import score_stock, decide_action
from src.signal_engine import compute_signal, to_markdown as signal_to_md
from src.utils import logger, is_market_data_valid


def build_scenarios(snap, score, setup):
    px = snap.get("price"); levels = snap.get("levels", {}) or {}
    res1 = levels.get("nearest_resistance"); res2 = levels.get("next_resistance")
    sup1 = levels.get("nearest_support"); cls = score["classification"]
    bullish = (f"If price closes decisively above {res1:.2f} on above-average volume, "
               f"the next target is {res2:.2f}." if (res1 and res2) else
               "If price reclaims recent highs on rising volume, the trend continues.")
    base = (f"With price at {px:.2f}, base case is range between {sup1:.2f} and {res1:.2f}."
            if sup1 and res1 else "Base case is consolidation around current levels.")
    bearish = (f"A daily close below {sup1:.2f} would invalidate the bullish setup."
               if sup1 else "Loss of the lower BB on volume would shift bias bearish.")
    if setup:
        invalidation = setup.get("invalidation")
        confirmation = (f"Daily close above {res1:.2f}." if res1 else "Higher high on rising volume.")
    else:
        invalidation = (f"Close below {sup1:.2f}." if sup1 else "Loss of 200-SMA.")
        confirmation = (f"Close above {res1:.2f}." if res1 else "Reclaim of 50-SMA.")
    horizon_map = {"Strong Candidate": "1-3 months", "Watchlist Candidate": "1-2 months",
                   "Neutral": "Wait for confirmation",
                   "Weak": "Avoid until structure improves",
                   "Avoid": "Not a buy candidate"}
    return {"bullish": bullish, "base": base, "bearish": bearish,
            "confirmation": confirmation, "invalidation": invalidation,
            "expected_horizon": horizon_map.get(cls, "Conditional")}


def _market_condition(bench_df):
    if bench_df is None or bench_df.empty or len(bench_df) < 200:
        return "Unknown"
    close = bench_df["Close"]
    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1]
    last = close.iloc[-1]
    if last > sma50 > sma200: return "Bullish (broad uptrend)"
    if last < sma50 < sma200: return "Bearish (broad downtrend)"
    if last > sma200: return "Mixed (above 200-SMA, below 50-SMA)"
    return "Weak (below 200-SMA)"


def analyze_stock(ticker, market="US", horizon="1Y", account_equity=None,
                   provider=None, backtest_overall=None):
    provider = provider or DEFAULT_PROVIDER
    bundle = load_stock(ticker, market, horizon, provider)
    if not is_market_data_valid(bundle["ohlcv"], min_rows=200):
        return {"ok": False, "error": f"Insufficient data for {ticker}.",
                "ticker": bundle["ticker"]}

    df_e = enrich(bundle["ohlcv"]); snap = snapshot(df_e)
    bench_df = bundle.get("benchmark_ohlcv")
    rs = None
    if isinstance(bench_df, pd.DataFrame) and not bench_df.empty:
        rs = relative_strength(df_e["Close"], bench_df["Close"], 63)
    snap["relative_strength_pct"] = rs

    f = extract_fundamentals(bundle["info"], bundle["financials"])
    setups = build_setups(df_e, snap)
    best = best_setup(setups)
    best_d = best.to_dict() if best else None
    sc = score_stock(snap, f, best_d)
    scenarios = build_scenarios(snap, sc, best_d)

    market_condition = _market_condition(bench_df) if isinstance(bench_df, pd.DataFrame) else "Unknown"
    decision = decide_action(
        score_total=sc["total"], classification=sc["classification"],
        best_setup=best_d, setups=[s.to_dict() for s in setups],
        snap=snap, market_condition=market_condition,
        fundamentals_available=fundamentals_available(f),
        history_bars=len(df_e),
    )

    pos_size = position_size(account_equity, best) if (account_equity and best) else None

    profile_map = {
        "Strong Candidate": "Suitable for swing/long-term with risk control.",
        "Watchlist Candidate": "Watchlist - wait for confirmation.",
        "Neutral": "Not actionable.",
        "Weak": "Not suitable for new long positions.",
        "Avoid": "Avoid for long entries."
    }

    result = {
        "ok": True, "generated_at": datetime.now().isoformat(timespec="seconds"),
        "ticker": bundle["ticker"], "market": bundle["market"],
        "name": f.get("name"), "sector": f.get("sector"),
        "industry": f.get("industry"), "currency": f.get("currency"),
        "snapshot": snap, "fundamentals": f,
        "fundamentals_available": fundamentals_available(f),
        "setups": [s.to_dict() for s in setups],
        "best_setup": best_d, "score": sc, "decision": decision,
        "scenarios": scenarios, "position_sizing": pos_size,
        "market_condition": market_condition,
        "benchmark_ticker": bundle["benchmark_ticker"],
        "suitable_profile": profile_map.get(sc["classification"], ""),
        "df_enriched": df_e,
    }
    # Auto-load cached backtest overall for this market when not supplied
    if backtest_overall is None:
        try:
            from src.backtesting import load_cached_overall
            backtest_overall = load_cached_overall(bundle["market"])
        except Exception:
            pass
    result["final_signal"] = compute_signal(
        analysis=result, backtest_overall=backtest_overall,
    )
    return result


def _fmt(v, digits=2, suffix=""):
    try:
        if v is None: return "-"
        if isinstance(v, float):
            if v != v: return "-"
            return f"{v:,.{digits}f}{suffix}"
        return f"{v}{suffix}"
    except Exception:
        return str(v)


def to_markdown(result):
    if not result.get("ok"):
        return f"# Analysis failed\n\n{result.get('error', 'Unknown error')}\n"
    snap = result["snapshot"]; f = result["fundamentals"] or {}
    sc = result["score"]; setups = result["setups"]; best = result["best_setup"]
    sc_components = sc["components"]; sc_max = sc["max_components"]
    scen = result["scenarios"]
    lines: List[str] = []
    name = result.get("name") or result["ticker"]
    lines += [
        f"# Stock Analysis Report - {result['ticker']} ({name})",
        f"_Generated: {result['generated_at']} | Market: {result['market']} | Benchmark: {result['benchmark_ticker']}_",
        "",
        f"**Sector:** {result.get('sector') or '-'} | **Industry:** {result.get('industry') or '-'} | **Currency:** {result.get('currency') or '-'}",
        "",
        "## Verdict",
        f"- **Classification:** {sc['classification']}",
        f"- **Score:** {sc['total']} / 100",
        f"- **Confidence:** {sc['confidence']}",
        f"- **Market condition:** {result['market_condition']}",
        f"- **Suitable for:** {result['suitable_profile']}",
        "",
    ]
    fs = result.get("final_signal")
    if fs:
        lines.append(signal_to_md(fs))
        lines.append("")

    dec = result.get("decision") or {}
    if dec:
        lines += [
            "## Decision",
            f"- **Action:** {dec.get('action')}",
            f"- **Tradable:** {'Yes' if dec.get('tradable') else 'No'}",
        ]
        for r in (dec.get("reasons") or []):
            lines.append(f"  - {r}")
        lines.append("")

    lines += [
        "## Scorecard",
        "| Component | Score | Max |",
        "|---|---:|---:|",
    ]
    for k in ("trend","momentum","volume","fundamental","valuation","risk_reward"):
        lines.append(f"| {k.replace('_',' ').title()} | {sc_components[k]:.1f} | {sc_max[k]} |")
    lines.append(f"| **Total** | **{sc['total']}** | **100** |")
    lines.append("")

    lines.append("### Why this score (per component)")
    for k, label in [("trend","Trend"),("momentum","Momentum"),("volume","Volume"),
                     ("fundamental","Fundamentals"),("valuation","Valuation"),("risk_reward","Risk/Reward")]:
        lines.append(f"**{label}:**")
        for r in sc["reasons"].get(k, []) or ["-"]:
            lines.append(f"- {r}")
        lines.append("")

    vol = snap.get("volume", {}) or {}; lvls = snap.get("levels", {}) or {}
    lines += [
        "## Technical Snapshot",
        f"- Price: **{_fmt(snap.get('price'))}** | Trend: **{snap.get('trend')}** | Structure: {snap.get('structure')}",
        f"- 20-SMA: {_fmt(snap.get('sma_20'))} | 50-SMA: {_fmt(snap.get('sma_50'))} | 200-SMA: {_fmt(snap.get('sma_200'))}",
        f"- RSI(14): {_fmt(snap.get('rsi'),1)} | MACD: {_fmt(snap.get('macd'),3)}",
        f"- ATR(14): {_fmt(snap.get('atr'))} | HV(30): {_fmt(snap.get('hv'),1,'%')}",
        f"- Volume vs avg: {_fmt(vol.get('volume_ratio'),2,'x')} | OBV: {vol.get('obv_trend')}",
        f"- Cross: {snap.get('cross') or 'none'}",
        f"- Support: {_fmt(lvls.get('nearest_support'))} | Resistance: {_fmt(lvls.get('nearest_resistance'))}",
        f"- RS vs benchmark (3M): {_fmt(snap.get('relative_strength_pct'),1,'%')}",
        "",
    ]

    lines += [
        "## Trade Setups",
        "| Style | Entry | Stop | TP1 | TP2 | TP3 | R:R TP1 | Acceptable |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for s in setups:
        lines.append(
            f"| {s['style']} | {_fmt(s['entry'])} | {_fmt(s['stop_loss'])} | "
            f"{_fmt(s['tp1'])} | {_fmt(s['tp2'])} | {_fmt(s.get('tp3'))} | "
            f"{_fmt(s['risk_reward_tp1'])} | {'Yes' if s['acceptable'] else 'No'} |"
        )
        if s.get("rejection_reasons"):
            lines.append(f"  - rejection: {'; '.join(s['rejection_reasons'])}")
    lines.append("")

    if best:
        lines += [
            "### Recommended Setup",
            f"- **Style:** {best['style']}",
            f"- **Entry:** {_fmt(best['entry'])} | **Stop:** {_fmt(best['stop_loss'])}",
            f"- **TP1:** {_fmt(best['tp1'])} | **TP2:** {_fmt(best['tp2'])} | **TP3:** {_fmt(best.get('tp3'))}",
            f"- **R:R TP1:** {_fmt(best['risk_reward_tp1'])} | **R:R TP2:** {_fmt(best['risk_reward_tp2'])}",
            f"- **Rationale:** {best['rationale']}",
            f"- **Invalidation:** {best['invalidation']}",
            "",
        ]

    if result.get("position_sizing"):
        ps = result["position_sizing"]
        lines += [
            "### Position Sizing",
            f"- Account equity: {_fmt(ps['account_equity'])} | Risk %: {ps['risk_pct']}%",
            f"- Risk $: {_fmt(ps['risk_dollars'])} | Risk per share: {_fmt(ps['risk_per_share'])}",
            f"- Suggested shares: **{ps['shares']}** | Notional: {_fmt(ps['notional'])}",
            "",
        ]

    if result.get("fundamentals_available"):
        lines += [
            "## Fundamentals",
            "| Metric | Value |", "|---|---:|",
            f"| Market cap | {_fmt(f.get('market_cap'),0)} |",
            f"| Revenue growth YoY | {_fmt(f.get('revenue_growth_yoy_pct'),1,'%')} |",
            f"| NI growth YoY | {_fmt(f.get('net_income_growth_yoy_pct'),1,'%')} |",
            f"| Gross margin | {_fmt(f.get('gross_margin_pct'),1,'%')} |",
            f"| Net margin | {_fmt(f.get('net_margin_pct'),1,'%')} |",
            f"| ROE | {_fmt(f.get('roe'),1,'%')} |",
            f"| P/E | {_fmt(f.get('pe'),1)} | Forward P/E | {_fmt(f.get('forward_pe'),1)} |",
            f"| P/S | {_fmt(f.get('ps'),2)} | P/B | {_fmt(f.get('pb'),2)} | PEG | {_fmt(f.get('peg'),2)} |",
            f"| D/E | {_fmt(f.get('debt_to_equity'),0)} | Current ratio | {_fmt(f.get('current_ratio'),2)} |",
            f"| FCF | {_fmt(f.get('free_cash_flow'),0)} | Div yield | {_fmt(f.get('dividend_yield_pct'),2,'%')} |",
            "",
        ]
    else:
        lines += ["## Fundamentals", "_Not enough fundamental data was available._", ""]

    bt = result.get("backtest")
    if bt and isinstance(bt, dict):
        o = bt.get("overall", {}) or {}
        lines += [
            "## Backtest summary (preliminary)",
            f"- Strategies: {', '.join(bt.get('strategies', []))}",
            f"- Tickers: {', '.join(bt.get('tickers_succeeded', []))}",
            f"- Trades: {o.get('num_trades', 0)} | Win: {o.get('win_rate_pct','-')}% | "
            f"Expectancy: {o.get('expectancy_R','-')}R | PF: {o.get('profit_factor','-')} | "
            f"Max DD: {o.get('max_drawdown_R','-')}R",
            f"- Verdict: {o.get('verdict','-')}",
        ]
        if o.get("sample_warning"):
            lines.append(f"- {o['sample_warning']}")
        lines.append("")

    lines += [
        "## Scenarios",
        f"- **Bullish:** {scen['bullish']}",
        f"- **Base:** {scen['base']}",
        f"- **Bearish:** {scen['bearish']}",
        f"- **Confirmation:** {scen['confirmation']}",
        f"- **Invalidation:** {scen['invalidation']}",
        f"- **Horizon:** {scen['expected_horizon']}",
        "",
        "## Disclaimer",
        config.DISCLAIMER, "",
    ]
    return "\n".join(lines)


def save_markdown(result, output_dir=None):
    output_dir = output_dir or config.REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{result['ticker'].replace('.', '_')}_report.md"
    path.write_text(to_markdown(result), encoding="utf-8")
    return path


def to_excel_bytes(result):
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        sc = result["score"]
        summary = {
            "Field": ["Ticker","Name","Market","Sector","Industry","Currency",
                      "Classification","Score","Confidence","Market Condition",
                      "Decision","Generated"],
            "Value": [result["ticker"], result.get("name") or "", result["market"],
                      result.get("sector") or "", result.get("industry") or "",
                      result.get("currency") or "", sc["classification"], sc["total"],
                      sc["confidence"], result["market_condition"],
                      (result.get("decision") or {}).get("action", ""),
                      result["generated_at"]],
        }
        pd.DataFrame(summary).to_excel(xl, sheet_name="Summary", index=False)
        comp = sc["components"]; mx = sc["max_components"]
        rows = [{"Component": k, "Score": comp[k], "Max": mx[k]} for k in comp]
        rows.append({"Component": "Total", "Score": sc["total"], "Max": 100})
        pd.DataFrame(rows).to_excel(xl, sheet_name="Scorecard", index=False)
        if result["setups"]:
            pd.DataFrame(result["setups"]).to_excel(xl, sheet_name="Setups", index=False)
        if result.get("fundamentals"):
            f = result["fundamentals"]
            pd.DataFrame([{"Metric": k, "Value": v} for k, v in f.items()]
                         ).to_excel(xl, sheet_name="Fundamentals", index=False)
        rows = []
        for k, items in sc["reasons"].items():
            for r in items: rows.append({"Component": k, "Reason": r})
        pd.DataFrame(rows).to_excel(xl, sheet_name="Reasons", index=False)
        df = result.get("df_enriched")
        if df is not None and not df.empty:
            df.tail(250).to_excel(xl, sheet_name="Indicators")
    out.seek(0)
    return out.read()
