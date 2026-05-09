"""
Streamlit dashboard for the local stock analyzer.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `import config` and `from src.X import ...` when run via `streamlit run app.py`
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

import config
from src.charting import plot_full_analysis
from src.report_generator import (
    analyze_stock, to_markdown, save_markdown, to_excel_bytes,
)
from src.screener import screen
from src.backtesting import run_backtest, to_markdown as bt_to_md


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Stock Analyzer — Decision Support",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

st.title("Stock Analyzer — Local Decision-Support Dashboard")
st.caption("Evidence-based stock analysis. Decision-support only. Not financial advice.")
with st.expander("Important disclaimer", expanded=False):
    st.warning(config.DISCLAIMER)


# ---------------------------------------------------------------------------
# Sidebar — global controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Controls")
    market = st.selectbox("Market", ["US", "SAUDI", "GLOBAL"], index=0)
    horizon = st.selectbox("Timeframe", ["1M", "3M", "6M", "1Y", "3Y", "5Y"], index=3)
    strategy = st.selectbox(
        "Strategy style",
        ["Swing Trade", "Long-Term Investment", "Breakout", "Pullback"],
        index=0,
    )
    risk_profile = st.selectbox(
        "Risk profile", ["Conservative", "Balanced", "Aggressive"], index=1,
    )
    account_equity = st.number_input(
        "Account equity (for sizing)", min_value=0.0, value=10_000.0, step=500.0,
    )
    st.markdown("---")
    st.caption("v1 · yfinance + custom indicators · transparent scoring")


tab_analyze, tab_screen, tab_backtest, tab_paper, tab_methodology = st.tabs(
    ["Analyze", "Screener", "Backtest", "Paper Trading", "Methodology"]
)


# ---------------------------------------------------------------------------
# Analyze tab
# ---------------------------------------------------------------------------
with tab_analyze:
    col1, col2 = st.columns([3, 1])
    with col1:
        ticker_in = st.text_input(
            "Ticker (e.g. AAPL, MSFT, 2222 for Saudi Aramco)",
            value="AAPL",
        ).strip()
    with col2:
        run = st.button("Analyze", use_container_width=True, type="primary")

    use_bt = st.checkbox(
        "Apply latest backtest evidence to confidence",
        value=True,
        help="If you've run the Backtest tab in this session, its overall stats "
             "are used to downgrade the signal's confidence when warranted.",
    )

    if run and ticker_in:
        bt_overall = None
        if use_bt and "last_backtest" in st.session_state:
            bt_overall = (st.session_state["last_backtest"] or {}).get("overall")
        with st.spinner(f"Analyzing {ticker_in}..."):
            try:
                result = analyze_stock(
                    ticker_in, market=market, horizon=horizon,
                    account_equity=account_equity if account_equity > 0 else None,
                    backtest_overall=bt_overall,
                )
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                result = {"ok": False, "error": str(e)}

        if not result.get("ok"):
            st.error(result.get("error", "Analysis failed."))
        else:
            sc = result["score"]
            best = result["best_setup"]
            fs = result.get("final_signal") or {}

            # ---- Final Signal banner (top of result) ----
            sig = fs.get("signal", "No Trade")
            sig_color = {
                "Strong Candidate": "success",
                "Watchlist Candidate": "info",
                "Wait for Breakout": "info",
                "Wait for Pullback": "info",
                "No Trade": "warning",
                "Avoid": "error",
                "High Risk": "error",
            }.get(sig, "info")
            getattr(st, sig_color)(
                f"### Final Signal: **{sig}**  ·  Confidence: **{fs.get('confidence','-')}**  "
                f"·  Setup: **{fs.get('setup_status','-')}**\n\n"
                f"_{fs.get('action_guidance','')}_"
            )
            with st.expander("Why & risk warnings", expanded=False):
                if fs.get("why"):
                    st.markdown("**Why:**")
                    for w in fs["why"]:
                        st.markdown(f"- {w}")
                if fs.get("risk_warnings"):
                    st.markdown("**Risk warnings:**")
                    for w in fs["risk_warnings"]:
                        st.markdown(f"- {w}")
                if fs.get("invalidation"):
                    st.markdown(f"**Invalidation:** {fs['invalidation']}")

            # ---- Top metrics ----
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Score", f"{sc['total']}/100", sc["classification"])
            m2.metric("Confidence", sc["confidence"])
            m3.metric("Trend", result["snapshot"].get("trend") or "—")
            m4.metric("RSI(14)", f"{result['snapshot'].get('rsi'):.1f}"
                      if result['snapshot'].get('rsi') is not None else "—")
            m5.metric("Market", result["market_condition"])

            # ---- Verdict box ----
            cls = sc["classification"]
            color = {"Strong Candidate": "success", "Watchlist Candidate": "info",
                     "Neutral": "warning", "Weak": "warning", "Avoid": "error"}.get(cls, "info")
            getattr(st, color)(
                f"**{cls}** · Score {sc['total']}/100 · Confidence {sc['confidence']}\n\n"
                f"_{result['suitable_profile']}_"
            )

            # Honest decision banner
            dec = result.get("decision") or {}
            if dec:
                if dec.get("tradable"):
                    st.success(f"Decision: **{dec.get('action')}**")
                else:
                    st.warning(f"Decision: **{dec.get('action')}**")
                if dec.get("reasons"):
                    with st.expander("Decision reasoning", expanded=False):
                        for r in dec["reasons"]:
                            st.markdown(f"- {r}")

            # ---- Chart ----
            st.subheader("Chart")
            try:
                chart_path = plot_full_analysis(
                    result["df_enriched"], result["snapshot"], best,
                    result["ticker"],
                )
                st.image(str(chart_path), use_column_width=True)
            except Exception as e:
                st.warning(f"Chart could not be rendered: {e}")

            # ---- Scorecard ----
            st.subheader("Scorecard")
            comp = sc["components"]; mx = sc["max_components"]
            df_score = pd.DataFrame([
                {"Component": k.replace("_", " ").title(),
                 "Score": comp[k], "Max": mx[k],
                 "% of Max": round(comp[k] / mx[k] * 100, 1) if mx[k] else 0}
                for k in comp
            ])
            st.dataframe(df_score, use_container_width=True, hide_index=True)

            with st.expander("Why this score (component reasons)", expanded=False):
                for k in ("trend","momentum","volume","fundamental","valuation","risk_reward"):
                    st.markdown(f"**{k.replace('_',' ').title()}:**")
                    for r in sc["reasons"].get(k, []):
                        st.markdown(f"- {r}")

            # ---- Setups ----
            st.subheader("Trade Setups")
            if result["setups"]:
                df_setups = pd.DataFrame(result["setups"])
                st.dataframe(df_setups, use_container_width=True, hide_index=True)
            else:
                st.info("No actionable long setups in current price action.")

            if best:
                st.markdown("**Recommended setup**")
                cols = st.columns(4)
                cols[0].metric("Style", best["style"])
                cols[1].metric("Entry", f"{best['entry']:.2f}")
                cols[2].metric("Stop", f"{best['stop_loss']:.2f}")
                cols[3].metric("R:R TP1", f"{best['risk_reward_tp1']:.2f}")
                cols2 = st.columns(3)
                cols2[0].metric("TP1", f"{best['tp1']:.2f}")
                cols2[1].metric("TP2", f"{best['tp2']:.2f}")
                cols2[2].metric("TP3", f"{best.get('tp3'):.2f}" if best.get("tp3") else "—")
                st.write(f"**Rationale:** {best['rationale']}")
                st.write(f"**Invalidation:** {best['invalidation']}")

                if result.get("position_sizing"):
                    ps = result["position_sizing"]
                    st.markdown("**Position sizing**")
                    pcols = st.columns(4)
                    pcols[0].metric("Shares", f"{ps['shares']}")
                    pcols[1].metric("Notional", f"{ps['notional']:.2f}")
                    pcols[2].metric("Risk $", f"{ps['risk_dollars']:.2f}")
                    pcols[3].metric("Exposure", f"{ps['exposure_pct']:.1f}%"
                                    if ps.get("exposure_pct") else "—")

            # ---- Fundamentals ----
            st.subheader("Fundamentals")
            if result.get("fundamentals_available"):
                f = result["fundamentals"]
                df_f = pd.DataFrame(
                    [{"Metric": k, "Value": v} for k, v in f.items()]
                )
                st.dataframe(df_f, use_container_width=True, hide_index=True)
            else:
                st.info("Fundamentals not available for this ticker from the provider.")

            # ---- Scenarios ----
            st.subheader("Scenarios")
            scen = result["scenarios"]
            st.markdown(f"- **Bullish:** {scen['bullish']}")
            st.markdown(f"- **Base case:** {scen['base']}")
            st.markdown(f"- **Bearish:** {scen['bearish']}")
            st.markdown(f"- **Confirmation:** {scen['confirmation']}")
            st.markdown(f"- **Invalidation:** {scen['invalidation']}")
            st.markdown(f"- **Expected horizon:** {scen['expected_horizon']}")

            # ---- Exports ----
            st.subheader("Exports")
            md = to_markdown(result)
            colA, colB, colC = st.columns(3)
            with colA:
                st.download_button(
                    "Download Markdown report", data=md,
                    file_name=f"{result['ticker']}_report.md", mime="text/markdown",
                )
            with colB:
                st.download_button(
                    "Download Excel workbook", data=to_excel_bytes(result),
                    file_name=f"{result['ticker']}_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            with colC:
                if st.button("Save report to disk", use_container_width=True):
                    p = save_markdown(result)
                    st.success(f"Saved to {p}")

            with st.expander("Full Markdown report", expanded=False):
                st.markdown(md)


# ---------------------------------------------------------------------------
# Screener tab
# ---------------------------------------------------------------------------
with tab_screen:
    st.subheader("Stock Screener")
    universe_default = ", ".join(config.DEFAULT_UNIVERSE.get(market, []))
    universe_in = st.text_area(
        "Tickers (comma-separated). Defaults to a curated universe per market.",
        value=universe_default, height=100,
    )
    require_fund = st.checkbox(
        "Include fundamentals (slower, hits API more)", value=False,
    )
    if st.button("Run screener", type="primary"):
        tickers = [t.strip() for t in universe_in.split(",") if t.strip()]
        bar = st.progress(0.0, text="Starting...")
        def cb(i, total, t):
            bar.progress(i / total, text=f"{i}/{total} — {t}")
        with st.spinner("Scanning universe..."):
            df = screen(
                tickers=tickers, market=market,
                require_fundamentals=require_fund, progress_callback=cb,
            )
        bar.empty()
        st.success(f"Scanned {len(df)} tickers.")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download CSV", data=df.to_csv(index=False),
            file_name=f"screener_{market}.csv", mime="text/csv",
        )


# ---------------------------------------------------------------------------
# Backtest tab
# ---------------------------------------------------------------------------
with tab_backtest:
    st.subheader("Strategy Backtest (preliminary)")
    st.caption(
        "No look-ahead. Slippage applied symmetrically. Treat results as preliminary "
        "until validated out-of-sample."
    )
    default_us = "AAPL, MSFT, NVDA, JPM, SPY"
    default_sa = "2222.SR, 1120.SR, 2010.SR, 7010.SR"
    bt_market = st.radio("Market", ["US", "SAUDI"], horizontal=True, index=0)
    bt_tickers = st.text_area(
        "Tickers (comma-separated)",
        value=default_us if bt_market == "US" else default_sa,
        height=80,
    )
    bt_strats = st.multiselect(
        "Strategies",
        ["breakout", "pullback", "trend_following"],
        default=["breakout", "pullback", "trend_following"],
    )
    bt_period = st.selectbox("Period", ["3y", "5y", "10y", "max"], index=1)

    if st.button("Run backtest", type="primary"):
        tickers = [t.strip() for t in bt_tickers.split(",") if t.strip()]
        bar = st.progress(0.0, text="Starting...")
        def cb(i, total, t):
            bar.progress(i / total, text=f"{i}/{total} — {t}")
        with st.spinner("Backtesting..."):
            try:
                bt = run_backtest(
                    tickers=tickers, market=bt_market,
                    strategies=bt_strats, period=bt_period,
                    progress_callback=cb,
                )
            except Exception as e:
                st.error(f"Backtest failed: {e}")
                bt = {"overall": {"num_trades": 0}, "failures": [{"error": str(e)}]}
        bar.empty()
        # Cache the latest backtest result for the Analyze tab to use
        st.session_state["last_backtest"] = bt

        o = bt.get("overall", {}) or {}
        if o.get("num_trades", 0) == 0:
            st.error("No trades generated. Check liquidity / data availability for these tickers.")
            if bt.get("failures"):
                st.write("Failures:", bt["failures"])
        else:
            cols = st.columns(5)
            cols[0].metric("Trades", o.get("num_trades"))
            cols[1].metric("Win rate", f"{o.get('win_rate_pct')}%")
            cols[2].metric("Expectancy", f"{o.get('expectancy_R')}R")
            cols[3].metric("Profit factor", o.get("profit_factor"))
            cols[4].metric("Max DD (R)", o.get("max_drawdown_R"))
            st.info(f"**Verdict:** {o.get('verdict')}")
            if o.get("sample_warning"):
                st.warning(o["sample_warning"])

            if bt.get("by_strategy"):
                st.markdown("**By strategy**")
                st.dataframe(pd.DataFrame(bt["by_strategy"]).T, use_container_width=True)
            if bt.get("by_ticker"):
                st.markdown("**By ticker**")
                st.dataframe(pd.DataFrame(bt["by_ticker"]).T, use_container_width=True)
            if bt.get("trades"):
                with st.expander(f"All {len(bt['trades'])} trades", expanded=False):
                    st.dataframe(pd.DataFrame(bt["trades"]), use_container_width=True)

            md = bt_to_md(bt)
            st.download_button(
                "Download backtest report (Markdown)", data=md,
                file_name=f"backtest_{bt_market}.md", mime="text/markdown",
            )

        if bt.get("failures"):
            with st.expander("Failures / skipped tickers", expanded=False):
                st.write(bt["failures"])


# ---------------------------------------------------------------------------
# Paper Trading tab
# ---------------------------------------------------------------------------
with tab_paper:
    from src.execution.paper_broker import PaperBroker
    from src.risk.risk_limits import RiskConfig, position_size, check_order, ALLOWED_SIGNALS

    st.subheader("Paper Trading (simulator only - no live orders)")

    if "paper_broker" not in st.session_state:
        st.session_state["paper_broker"] = PaperBroker(10_000.0)
    pb: PaperBroker = st.session_state["paper_broker"]

    c = st.columns(3)
    starting = c[0].number_input("Starting balance", min_value=100.0,
                                  value=pb.starting_cash, step=500.0)
    risk_pct = c[1].number_input("Risk per trade %", min_value=0.1,
                                  max_value=10.0, value=1.0, step=0.1)
    max_pos = c[2].number_input("Max open positions", min_value=1,
                                 max_value=20, value=5, step=1)
    if starting != pb.starting_cash and not pb.positions:
        st.session_state["paper_broker"] = PaperBroker(starting); pb = st.session_state["paper_broker"]

    c2 = st.columns([2, 1, 1])
    pt_ticker = c2[0].text_input("Ticker", value="AAPL").strip()
    pt_market = c2[1].selectbox("Market", ["US", "SAUDI"], index=0, key="pt_market")
    if c2[2].button("Load Latest Signal", use_container_width=True):
        try:
            from src.report_generator import analyze_stock
            r = analyze_stock(pt_ticker, market=pt_market)
            st.session_state["pt_signal"] = r.get("final_signal") if r.get("ok") else None
            st.session_state["pt_signal_err"] = None if r.get("ok") else r.get("error")
        except Exception as e:
            st.session_state["pt_signal"] = None
            st.session_state["pt_signal_err"] = str(e)

    sig = st.session_state.get("pt_signal")
    if st.session_state.get("pt_signal_err"):
        st.error(st.session_state["pt_signal_err"])
    if sig:
        st.info(f"Signal: **{sig.get('signal')}** | Confidence: {sig.get('confidence')} "
                f"| Setup: {sig.get('setup_status')}")
        entry = sig.get("entry"); sl = sig.get("stop_loss")
        cfg = RiskConfig(risk_per_trade_pct=risk_pct, max_open_positions=int(max_pos))
        qty = position_size(pb.portfolio_value(), entry or 0, sl or 0, risk_pct)
        rps = abs((entry or 0) - (sl or 0)) if (entry and sl) else 0
        capital_used = qty * (entry or 0)
        max_risk = qty * rps
        # Trade plan cards
        r1 = st.columns(5)
        r1[0].metric("Entry", f"{entry:.2f}" if entry else "-")
        r1[1].metric("Stop Loss", f"{sl:.2f}" if sl else "-")
        r1[2].metric("TP1", f"{sig.get('tp1'):.2f}" if sig.get("tp1") else "-")
        r1[3].metric("TP2", f"{sig.get('tp2'):.2f}" if sig.get("tp2") else "-")
        r1[4].metric("TP3", f"{sig.get('tp3'):.2f}" if sig.get("tp3") else "-")
        r2 = st.columns(4)
        r2[0].metric("Risk / share", f"${rps:.2f}")
        r2[1].metric("Quantity", qty)
        r2[2].metric("Capital used", f"${capital_used:,.2f}")
        r2[3].metric("Max risk", f"${max_risk:,.2f}")
        chk = check_order(signal=sig, account_equity=pb.portfolio_value(),
                          qty=qty, open_positions=pb.open_count(),
                          daily_pnl_pct=pb.daily_pnl_pct(), cfg=cfg)
        if not chk["allowed"]:
            st.warning("Approve Paper Trade BLOCKED: " + "; ".join(chk["reasons"]))
        if st.button("Approve Paper Trade", type="primary",
                      disabled=not chk["allowed"]):
            res = pb.submit_order(pt_ticker, qty, entry, sl,
                                  sig.get("tp1"), sig.get("tp2"), sig.get("tp3"),
                                  meta={"signal": sig.get("signal")})
            if res.get("ok"): st.success(f"Opened paper position: {res['position']}")
            else: st.error(res.get("error"))

    st.markdown("---")
    pv_col, btn_col = st.columns([3, 1])
    pv_col.metric("Portfolio value", f"${pb.portfolio_value():,.2f}",
                   f"{pb.daily_pnl_pct():.2f}% from start")
    if btn_col.button("Update Market Prices", use_container_width=True,
                       disabled=not pb.positions):
        try:
            from src.data_loader import DEFAULT_PROVIDER
            new_marks = {}
            for tk in pb.positions:
                df = DEFAULT_PROVIDER.fetch_ohlcv(tk, period="5d", interval="1d")
                if df is not None and not df.empty:
                    new_marks[tk] = float(df["Close"].iloc[-1])
            pb.mark_to_market(new_marks)
            st.success(f"Updated {len(new_marks)} marks.")
        except Exception as e:
            st.error(f"Price update failed: {e}")

    st.markdown("**Open positions**")
    if pb.positions:
        rows = []
        for tk, p in pb.positions.items():
            mark = pb._marks.get(tk, p["entry"])
            cur_val = p["qty"] * mark
            upnl = (mark - p["entry"]) * p["qty"]
            upnl_pct = (mark / p["entry"] - 1) * 100 if p["entry"] else 0
            rows.append({
                "Ticker": tk.upper(), "Qty": p["qty"],
                "Entry": round(p["entry"], 2), "Mark": round(mark, 2),
                "Current Value": round(cur_val, 2),
                "Unrealized P&L": round(upnl, 2),
                "Unrealized %": round(upnl_pct, 2),
                "Stop": p["stop_loss"], "TP1": p.get("tp1"),
                "Opened": p["opened_at"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        close_t = st.selectbox("Close position", list(pb.positions.keys()),
                                key="pt_close_ticker")
        close_px = st.number_input("Exit price", min_value=0.01,
                                    value=float(pb._marks.get(close_t, pb.positions[close_t]["entry"])))
        if st.button("Close Position"):
            r = pb.close_position(close_t, close_px, reason="manual")
            (st.success if r.get("ok") else st.error)(r)
    else:
        st.caption("No open paper positions.")

    st.markdown("**Trade log**")
    if pb.trade_log:
        log_rows = []
        for e in pb.trade_log:
            log_rows.append({
                "Action": e.get("action"),
                "Ticker": (e.get("ticker") or "").upper(),
                "Qty": e.get("qty"),
                "Entry": e.get("entry"),
                "Exit": e.get("exit"),
                "Realized P&L": e.get("pnl"),
                "Reason": e.get("reason", ""),
                "Timestamp": e.get("opened_at") or e.get("closed_at"),
            })
        st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No trades yet.")


# ---------------------------------------------------------------------------
# Methodology tab
# ---------------------------------------------------------------------------
with tab_methodology:
    st.subheader("Methodology — How the score is computed")
    st.markdown(f"""
**Total = 100 points, broken down per `config.WEIGHTS`:**

- **Trend** ({config.WEIGHTS.technical_trend} pts): price vs 20/50/200 SMA stacking, 50/200 cross, classification.
- **Momentum** ({config.WEIGHTS.momentum} pts): RSI healthy zone, MACD posture, StochRSI, relative strength bonus.
- **Volume** ({config.WEIGHTS.volume} pts): volume vs 20-day average, breakout day, OBV trend.
- **Fundamentals** ({config.WEIGHTS.fundamental} pts): revenue and net income growth, margins, ROE, leverage, free cash flow, liquidity.
- **Valuation** ({config.WEIGHTS.valuation} pts): P/E, P/S, P/B, PEG.
- **Risk/Reward** ({config.WEIGHTS.risk_reward} pts): quality of best available trade setup (R:R to TP1, TP2).

**Classification thresholds:**
- ≥ 75 → Strong Candidate
- ≥ 60 → Watchlist Candidate
- ≥ 45 → Neutral
- ≥ 30 → Weak
- < 30 → Avoid

**Confidence** is independent of raw score: it counts how many of the six components scored ≥ 60% of max. Five+ → High; three–four → Medium; otherwise Low. This penalizes one-dimensional setups (e.g. great price action but ugly fundamentals).

**Trade-setup geometry** is derived from swing-pivot S/R combined with ATR(14) and 20-day structural lows. Every setup is scored on R:R to TP1 and rejected from "acceptable" status if R:R < {config.RISK.min_risk_reward}.

**Limitations:**
- yfinance fundamentals are best-effort; many Tadawul tickers will not have ratios.
- TASI index symbol coverage on yfinance is partial — falls back to S&P 500 if missing.
- All indicators are computed on daily candles regardless of UI horizon (daily is more reliable for trend/MA/RSI signals than intraday).
- This tool does not predict prices. Scenarios are conditional descriptions of what *would* confirm or invalidate the current setup.
""")


st.markdown("---")
st.caption(config.DISCLAIMER)
