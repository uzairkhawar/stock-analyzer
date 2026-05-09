# Stock Analyzer — Local Decision-Support Dashboard

A professional-grade, **local** stock analysis tool. It pulls real market data, computes a wide set of technical and fundamental signals, scores each stock with a transparent 0–100 model, and proposes entry / stop-loss / take-profit setups with explicit risk/reward and invalidation rules.

**Decision support, not financial advice.** This tool never says "buy" or "sell" — it classifies setups (Strong Candidate / Watchlist / Neutral / Weak / Avoid), explains the reasoning per component, and tells you exactly what would invalidate the thesis.

---

## Features (v1)

- **Data**: yfinance for US, Saudi (Tadawul `.SR` suffix), and global tickers. Provider-agnostic interface so Polygon / Finnhub / Alpha Vantage can be added in v2.
- **Technical engine**: SMA/EMA (20/50/100/200), trend classification, golden/death cross, RSI, MACD, StochRSI, Rate of Change, ATR, Bollinger Bands, historical volatility, OBV, A/D, swing-pivot S/R, market structure (HH/HL etc.).
- **Fundamentals engine**: revenue/net-income growth, margins, ROE/ROA, debt/equity, current ratio, free cash flow, P/E, forward P/E, P/S, P/B, PEG, dividend yield. Works where the provider exposes them; returns "unavailable" cleanly otherwise.
- **Scoring**: 100-point weighted model. Every point is explainable.
- **Trade setups**: Breakout, Pullback, Conservative (retest), Aggressive. Each has explicit entry, ATR/structure stop, three TPs from S/R, R:R, and invalidation level. Setups below 1:2 R:R are flagged unacceptable.
- **Position sizing**: fixed-fractional risk (default 1% of account equity).
- **Scenarios**: Bullish / Base / Bearish with explicit confirmation and invalidation conditions and an expected horizon.
- **Market filter**: compares vs S&P 500 (US) or TASI index (Saudi, where available). Reports broad market posture so you can avoid aggressive longs in weak tape.
- **Charting**: multi-panel candlestick + MAs + S/R + entry/SL/TP + volume + RSI + MACD, exported as PNG.
- **Screener**: scans a curated universe (US default 50 names; Saudi default 10) with technical/fundamental/risk filters and ranks by score.
- **Streamlit UI** with three tabs: Analyze, Screener, Methodology.
- **Exports**: Markdown report, Excel workbook (Summary / Scorecard / Setups / Fundamentals / Reasons / Indicators), saved chart PNG.
- **Caching**: file-based pickle cache in `data/cache/` to reduce redundant API calls.

---

## Installation

Requires Python 3.10+.

```bash
cd stock_analyzer
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

A browser tab opens at `http://localhost:8501`. Pick a ticker, set your timeframe and risk profile, click **Analyze**.

Examples:
- US: `AAPL`, `MSFT`, `NVDA`, `JPM`
- Saudi: `2222` (Aramco), `1120` (Al Rajhi), `2010` (SABIC). The `.SR` suffix is added automatically when Market = SAUDI.

To use the screener, switch to the **Screener** tab, optionally edit the universe text box, and click **Run screener**.

---

## Project Structure

```
stock_analyzer/
├── app.py                  # Streamlit dashboard
├── config.py               # Global config (weights, thresholds, paths, universes)
├── requirements.txt
├── README.md
├── data/                   # cached payloads (auto-created)
├── reports/                # saved markdown reports (auto-created)
├── charts/                 # saved PNG charts (auto-created)
└── src/
    ├── data_loader.py      # DataProvider ABC + YFinanceProvider
    ├── indicators.py       # All technical indicators + snapshot
    ├── fundamentals.py     # Ratio extraction from yfinance
    ├── scoring.py          # 100-point transparent scoring engine
    ├── risk_management.py  # Entry/SL/TP geometry, sizing
    ├── charting.py         # PNG chart with full overlays
    ├── screener.py         # Universe scan + ranking
    ├── report_generator.py # End-to-end pipeline + Markdown/Excel export
    └── utils.py            # Cache, ticker normalization, safe getters
```

---

## Methodology

### Scoring — 100 points total

| Component        | Max | What it measures |
|------------------|----:|------------------|
| Trend            | 25  | Price vs 20/50/200 SMAs, stack order, golden/death cross, classification |
| Momentum         | 15  | RSI healthy zone, MACD line vs signal + histogram sign, StochRSI, optional relative-strength bonus |
| Volume           | 15  | Volume vs 20-day avg, breakout day flag, OBV trend |
| Fundamentals     | 25  | Revenue + net income growth YoY, gross/net margin, ROE, debt/equity, FCF sign, current ratio |
| Valuation        | 10  | P/E, P/S, P/B, PEG bands |
| Risk/Reward      | 10  | Quality of best trade setup (R:R to TP1 and TP2) |

**Classification:** ≥75 Strong · ≥60 Watchlist · ≥45 Neutral · ≥30 Weak · <30 Avoid.

**Confidence** is computed *independently* from the breadth of agreement: how many of the 6 components scored ≥60% of their max. This penalizes one-dimensional setups.

### Trade setups

Four styles are constructed from price/structure/ATR:

| Style          | Entry idea                                | Stop-loss logic |
|----------------|-------------------------------------------|-----------------|
| Breakout       | Slightly above 20-day high                | max(structure low, entry − 1.5×ATR) |
| Pullback       | Near 20-EMA in confirmed uptrend          | min(1.5×ATR stop, structure low) |
| Conservative   | Above nearest resistance after retest     | max(structure low, ATR stop) |
| Aggressive     | Market entry if uptrend & RSI 45–70       | max(structure low, 1.5×ATR) |

TPs use the nearest 1–3 swing-high resistances; if missing, fall back to measured-move multiples of risk (2R, 3R, 5R).

A setup is **acceptable** only if R:R to TP1 ≥ 2.0 (configurable in `config.RISK.min_risk_reward`).

---

## Sample Report Format

Each Markdown report contains:

1. Header — ticker, name, sector, industry, currency, generation timestamp
2. **Verdict** — classification, total score, confidence, market condition, suitable profile
3. **Scorecard** — six components plus reasons per component
4. **Technical Snapshot** — price, trend, structure, MAs, RSI, MACD, ATR, HV, volume diagnostics, S/R levels, relative strength
5. **Trade Setups** table + recommended setup + position sizing
6. **Fundamentals** table (when available)
7. **Scenarios** — bullish / base / bearish + confirmation + invalidation + expected horizon
8. **Disclaimer**

A sample for AAPL would look like (truncated):

```
# Stock Analysis Report — AAPL (Apple Inc.)
_Generated: 2026-04-30T... · Market: US · Benchmark: ^GSPC_

## Verdict
- Classification: Watchlist Candidate
- Score: 64.5 / 100
- Confidence: Medium
- Market condition: Bullish (broad uptrend)
- Suitable for: Watchlist — wait for confirmation. Suitable for patient investors.

## Scorecard
| Component | Score | Max |
| Trend     | 19.0  | 25  |
| Momentum  | 11.0  | 15  |
| Volume    |  9.0  | 15  |
| Fundamental | 18.0 | 25 |
| Valuation |  4.0  | 10  |
| Risk_Reward | 3.5 | 10  |
| Total     | 64.5  | 100 |

### Why this score
Trend: Price above 200-SMA (long-term uptrend). 50-SMA above 200-SMA (bullish stack). ...

## Trade Setups
| Style    | Entry | Stop | TP1 | TP2 | TP3 | R:R TP1 | Acceptable |
| pullback | 184.20| 178.10| 192.50 | 198.40 | 210.00 | 1.37 | No |
| breakout | 192.50| 184.10| 198.40 | 210.00 | 224.00 | 0.70 | No |
...

## Scenarios
- Bullish: If price closes decisively above 192.50 on above-average volume, ...
- Base case: With price at 187.30, base case is range-bound action between 178.50 and 192.50.
- Bearish: A daily close below 178.50 would invalidate the bullish setup ...
- Confirmation: Daily close above 192.50.
- Invalidation: Daily close back below 192.50 or below 178.10.
- Expected horizon: 1–2 months.
```

---

## Error Handling

- **Invalid tickers**: returns a clean `{ok: False, error: ...}` payload; UI shows error.
- **Missing fundamentals**: scoring skips fundamental + valuation; verdict notes the omission.
- **Insufficient OHLCV** (<200 daily bars): analysis aborts with explanation. Indicators that need 200 bars (SMA-200) are flagged.
- **API rate limits**: file-based cache (default 30 min TTL for OHLCV, 24 h for info/financials) reduces hits. yfinance failures are caught and logged.
- **Partial Saudi index coverage**: TASI symbol fallback to S&P 500 is automatic; all derived figures stay valid.

---

## Backtesting

A strict, no-look-ahead historical backtest is included for the breakout, pullback, and trend-following strategies. At each decision bar, only data up to that bar is used to compute indicators, swing-pivot levels, stops, and targets. Entries fill on the **next bar's open**, with symmetric slippage applied to both fills (default 0.10% per side). SL is the max of the recent 20-day swing low and 1.5×ATR below entry. TP1 is the nearest swing-high resistance (or 2R fallback); TP2 = 3R; TP3 = 5R. Setups failing the minimum 1:2 R:R or with stops wider than 8% of entry are discarded.

Results report number of trades, win rate, expectancy in R, profit factor, max drawdown (R), average holding days, best/worst trade, and a per-ticker buy-and-hold comparison. A sample-size warning fires below 20 trades. The aggregate verdict explicitly says when the strategy is unprofitable on the sample, marginal, mixed, or acceptable-but-still-preliminary — never "guaranteed".

Run from the **Backtest** tab in the Streamlit app, or programmatically:

```python
from src.backtesting import run_backtest, to_markdown
result = run_backtest(["AAPL","MSFT","NVDA","JPM","SPY"], market="US", period="5y")
print(to_markdown(result))
```

Limitations: trades are sized in R-multiples (not dollars), entries assume open-fill execution, and Saudi tickers depend entirely on yfinance's coverage of Tadawul listings — many will return no data.

## Roadmap — v2 and beyond

- **v2 — Fundamentals & coverage**
  - Plug Polygon, Finnhub, Twelve Data, Alpha Vantage as alternative providers (provider implements the existing `DataProvider` ABC).
  - Direct Tadawul scraping for fundamentals + index data.
  - Sector/peer comparison view (rank stock vs sector average for each ratio).
- **v2 — Backtesting**
  - Walk-forward backtest of any setup style on the universe; report hit-rate, expectancy, max drawdown.
  - Calibrate `min_risk_reward` and classification thresholds from realized data.
- **v2 — AI-generated narrative**
  - LLM-rewritten report section that turns the structured findings into a natural-language brief, with the *exact* numbers as inputs (no hallucinated levels).
- **v2 — Risk extensions**
  - VaR / CVaR estimates from historical-simulation.
  - Correlation matrix and portfolio-level exposure check.
- **v2 — UI**
  - Interactive Plotly chart with toggleable overlays.
  - Saved-watchlist and alerts (price, RSI, MA crossovers).

---

## Quant-review notes (self-critique)

These are known limitations to address before relying on the tool for live decisions:

1. **Indicator computation uses daily candles only.** For the "1D / 1W" UI horizons the chart shrinks but the indicators do not switch to intraday. This is intentional (intraday MA/RSI are noisier and more rate-limited) but worth knowing.
2. **Swing pivots** are computed with a fixed lookback. They lag — they cannot identify the *current* candle as a pivot. For very recent S/R levels, use the chart visually as well.
3. **yfinance fundamentals** are pulled on a best-effort basis. For some tickers, `info` returns empty or `priceToBook` is missing. The scoring model gracefully degrades but you may see "fundamentals unavailable" for newer or less-covered listings.
4. **No backtest validation** in v1. The classification thresholds (75/60/45/30) are heuristic. Treat the score as an *ordinal* signal, not a calibrated probability.
5. **TASI index** is not consistently exposed by yfinance. The market-condition figure for Saudi may fall back to S&P 500.
6. **Walk-forward stability** of the score over time has not been measured. Different volatility regimes will likely require re-tuning the RSI healthy band and ATR stop multiplier.
7. **Position sizing** is fixed-fractional. It does not account for correlation across open positions or volatility-targeting.

---

## License & Disclaimer

Personal/educational use. Adapt freely.

> **DISCLAIMER:** This tool is decision-support software for educational and research purposes. It is not financial advice, not a recommendation to buy or sell any security, and not a guarantee of future performance. Markets carry risk of substantial loss. Always perform independent research and consult a licensed advisor before making investment decisions.
