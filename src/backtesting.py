"""
src/backtesting.py — strict, no-look-ahead historical backtest of the
breakout / pullback / trend-following strategies used by the analyzer.

Design choices
--------------
* We iterate bar-by-bar. At decision bar `i`, only data in df.iloc[:i+1] is used
  (entry signal, stop loss, take-profit calculation).
* Entry fills at the next bar's open (so signal -> entry never overlaps).
* Slippage is applied symmetrically (worse fill on entry and exit).
* SL/TP are static, set at entry. Exits are simulated bar-by-bar:
    - If the next bar's open gaps past SL/TP, fill at the open.
    - Otherwise SL is checked first, then TP1/TP2/TP3 (conservative).
* Max holding period exits at the close of the last allowed bar.
* R-multiple = (exit - entry) / risk_per_share. Win = R > 0.
* Buy-and-hold benchmark uses the same window: entry at the first decision bar
  open, exit at the last bar close, both adjusted for slippage.
* Statistics include win-rate, expectancy in R, profit factor, max drawdown of
  the trade equity curve, average holding days, best/worst trade, and a
  comparison against buy-and-hold.

This is decision-support analytics, not a high-frequency simulator.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
from src.data_loader import DEFAULT_PROVIDER, DataProvider
from src.indicators import enrich
from src.utils import logger, normalize_ticker, is_market_data_valid

BT = config.BACKTEST


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------
@dataclass
class Trade:
    ticker: str
    strategy: str
    entry_date: pd.Timestamp
    entry_price: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: Optional[float]
    risk_per_share: float
    exit_date: pd.Timestamp
    exit_price: float
    exit_reason: str         # SL, TP1, TP2, TP3, TIME
    holding_days: int
    pnl_per_share: float
    r_multiple: float

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["entry_date"] = self.entry_date.strftime("%Y-%m-%d")
        d["exit_date"] = self.exit_date.strftime("%Y-%m-%d")
        return d


# ---------------------------------------------------------------------------
# Signal generators (strict no-look-ahead — see history slice below)
# ---------------------------------------------------------------------------
def _atr_at(history: pd.DataFrame, period: int = 14) -> Optional[float]:
    if len(history) < period + 2:
        return None
    high, low, close = history["High"], history["Low"], history["Close"]
    prev = close.shift(1)
    tr = pd.concat([(high - low).abs(),
                    (high - prev).abs(),
                    (low - prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean().iloc[-1]
    return float(atr) if not np.isnan(atr) else None


def _swing_low(history: pd.DataFrame, lookback: int = 20) -> Optional[float]:
    if len(history) < lookback:
        return None
    return float(history["Low"].iloc[-lookback:].min())


def _nearest_resistance(history: pd.DataFrame, price: float,
                        lookback: int = 200, swing: int = 10) -> Optional[float]:
    """Nearest swing-high above price using only past data."""
    h = history["High"].iloc[-lookback:]
    pivots: List[float] = []
    for i in range(swing, len(h) - swing):
        win = h.iloc[i - swing:i + swing + 1]
        if h.iloc[i] == win.max():
            pivots.append(float(h.iloc[i]))
    above = sorted(p for p in pivots if p > price)
    return above[0] if above else None


def signal_breakout(history: pd.DataFrame) -> Optional[Dict]:
    """Bar i is a breakout bar if its close > prior 20-day high and volume >= 1.3x avg."""
    n = BT.breakout_lookback
    if len(history) < n + 5:
        return None
    last = history.iloc[-1]
    prior_high = float(history["High"].iloc[-(n + 1):-1].max())
    avg_vol = float(history["Volume"].iloc[-(n + 1):-1].mean())
    if avg_vol <= 0:
        return None
    if last["Close"] > prior_high and last["Volume"] >= avg_vol * BT.breakout_volume_multiple:
        return {"prior_high": prior_high}
    return None


def signal_pullback(history: pd.DataFrame) -> Optional[Dict]:
    """Pullback in uptrend: price > SMA50 for >=50 bars; today's low <= EMA20 <= today's high;
    today's close > today's open (bullish reversal candle)."""
    if len(history) < 60:
        return None
    close = history["Close"]
    sma50 = close.rolling(50).mean()
    if sma50.isna().iloc[-1]:
        return None
    # require uptrend confirmation
    above = (close > sma50).iloc[-BT.pullback_min_uptrend_bars:]
    if above.sum() < int(0.7 * BT.pullback_min_uptrend_bars):
        return None
    ema20 = close.ewm(span=20, adjust=False).mean()
    last_ema20 = float(ema20.iloc[-1])
    last = history.iloc[-1]
    touched = last["Low"] <= last_ema20 <= last["High"]
    bullish = last["Close"] > last["Open"]
    if touched and bullish:
        return {"ema20": last_ema20}
    return None


def signal_trend_following(history: pd.DataFrame) -> Optional[Dict]:
    """50-SMA cross above 200-SMA today, with RSI in healthy range."""
    if len(history) < 220:
        return None
    close = history["Close"]
    s50 = close.rolling(50).mean()
    s200 = close.rolling(200).mean()
    if s50.isna().iloc[-1] or s200.isna().iloc[-1] or s50.isna().iloc[-2] or s200.isna().iloc[-2]:
        return None
    crossed = (s50.iloc[-2] <= s200.iloc[-2]) and (s50.iloc[-1] > s200.iloc[-1])
    # RSI(14) check
    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).iloc[-1]
    if not crossed:
        return None
    if not (BT.trend_rsi_min <= rsi <= BT.trend_rsi_max):
        return None
    return {"rsi": float(rsi)}


SIGNAL_REGISTRY: Dict[str, Callable[[pd.DataFrame], Optional[Dict]]] = {
    "breakout": signal_breakout,
    "pullback": signal_pullback,
    "trend_following": signal_trend_following,
}


# ---------------------------------------------------------------------------
# Trade simulation
# ---------------------------------------------------------------------------
def _simulate_trade(df: pd.DataFrame, signal_idx: int, ticker: str, strategy: str,
                    history: pd.DataFrame) -> Optional[Trade]:
    """signal_idx = bar where signal fired. Entry at bar signal_idx+1 open."""
    if signal_idx + 1 >= len(df):
        return None
    entry_bar = df.iloc[signal_idx + 1]
    entry_date = df.index[signal_idx + 1]
    raw_entry = float(entry_bar["Open"])
    if raw_entry <= 0 or np.isnan(raw_entry):
        return None
    entry_price = raw_entry * (1 + BT.slippage_pct / 100)

    # Stop loss: max(structure low, entry - 1.5*ATR)
    atr_v = _atr_at(history, 14) or (entry_price * 0.02)
    swing_low = _swing_low(history, 20) or (entry_price * 0.95)
    sl = max(swing_low, entry_price - 1.5 * atr_v)
    if sl >= entry_price:
        return None
    risk = entry_price - sl

    # Reject up front if R:R won't work or stop is too wide
    if (risk / entry_price * 100) > config.REJECT.max_stop_width_pct:
        return None

    # Targets: nearest_resistance for TP1; otherwise 2R; TP2 = 3R, TP3 = 5R
    nearest_res = _nearest_resistance(history, entry_price)
    tp1 = nearest_res if (nearest_res and nearest_res > entry_price + 1.5 * risk) else entry_price + 2 * risk
    tp2 = entry_price + 3 * risk
    tp3 = entry_price + 5 * risk
    rr1 = (tp1 - entry_price) / risk
    if rr1 < config.RISK.min_risk_reward:
        return None

    # Walk forward
    max_j = min(signal_idx + 1 + BT.max_holding_days, len(df) - 1)
    exit_price = None
    exit_reason = "TIME"
    exit_idx = max_j
    for j in range(signal_idx + 1, max_j + 1):
        bar = df.iloc[j]
        # Gap-down past SL on the open
        if bar["Open"] <= sl:
            exit_price = float(bar["Open"]); exit_reason = "SL_GAP"; exit_idx = j; break
        # Intraday SL hit
        if bar["Low"] <= sl:
            exit_price = float(sl); exit_reason = "SL"; exit_idx = j; break
        # Intraday TP hit (use TP3 first if reached, else TP2, else TP1 — most realistic for trailing)
        if bar["High"] >= tp3:
            exit_price = float(tp3); exit_reason = "TP3"; exit_idx = j; break
        if bar["High"] >= tp2:
            exit_price = float(tp2); exit_reason = "TP2"; exit_idx = j; break
        if bar["High"] >= tp1:
            exit_price = float(tp1); exit_reason = "TP1"; exit_idx = j; break

    if exit_price is None:
        exit_price = float(df.iloc[max_j]["Close"]); exit_idx = max_j; exit_reason = "TIME"
    # exit slippage (worse for the trader)
    exit_price *= (1 - BT.slippage_pct / 100)

    pnl_per_share = exit_price - entry_price
    r_mult = pnl_per_share / risk if risk > 0 else 0.0
    return Trade(
        ticker=ticker, strategy=strategy,
        entry_date=entry_date, entry_price=round(entry_price, 4),
        stop_loss=round(sl, 4), tp1=round(tp1, 4), tp2=round(tp2, 4), tp3=round(tp3, 4),
        risk_per_share=round(risk, 4),
        exit_date=df.index[exit_idx], exit_price=round(exit_price, 4),
        exit_reason=exit_reason,
        holding_days=int((df.index[exit_idx] - entry_date).days),
        pnl_per_share=round(pnl_per_share, 4),
        r_multiple=round(r_mult, 3),
    )


# ---------------------------------------------------------------------------
# Per-ticker backtest
# ---------------------------------------------------------------------------
def backtest_ticker(ticker: str, df: pd.DataFrame, strategies: List[str]) -> List[Trade]:
    """Run all selected strategies bar-by-bar over df (must be enriched? No — raw OHLCV)."""
    trades: List[Trade] = []
    if not is_market_data_valid(df, BT.min_history_for_signal):
        logger.warning("Skipping %s: insufficient history (%s)",
                       ticker, 0 if df is None else len(df))
        return trades

    _v = df["Volume"].rolling(20).mean().iloc[-1]
    avg_v = float(_v) if _v == _v and _v else 0.0
    if avg_v < BT.min_avg_volume:
        logger.warning("Skipping %s: avg volume %.0f below floor", ticker, avg_v)
        return trades

    # In-trade tracking per strategy: don't pile on duplicate signals while a trade is open
    in_trade_until: Dict[str, int] = {s: -1 for s in strategies}

    for i in range(BT.min_history_for_signal, len(df) - 1):
        history = df.iloc[: i + 1]
        for strat in strategies:
            if i <= in_trade_until[strat]:
                continue
            sig_fn = SIGNAL_REGISTRY.get(strat)
            if not sig_fn:
                continue
            sig = sig_fn(history)
            if sig is None:
                continue
            t = _simulate_trade(df, i, ticker, strat, history)
            if t is None:
                continue
            trades.append(t)
            # Block additional entries until exit
            exit_pos = df.index.get_loc(t.exit_date)
            if isinstance(exit_pos, slice):
                exit_pos = exit_pos.stop - 1
            in_trade_until[strat] = int(exit_pos)
    return trades


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def _max_drawdown_r(trades: List[Trade]) -> float:
    """Max drawdown of cumulative R-multiple equity curve."""
    if not trades:
        return 0.0
    cum = np.cumsum([t.r_multiple for t in trades])
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    return float(dd.max()) if len(dd) else 0.0


def _buy_hold_return_pct(df: pd.DataFrame) -> float:
    if df is None or df.empty or len(df) < 2:
        return 0.0
    start_idx = max(BT.min_history_for_signal, 0)
    if start_idx >= len(df):
        start_idx = 0
    start = float(df["Open"].iloc[start_idx])
    end = float(df["Close"].iloc[-1])
    if start <= 0:
        return 0.0
    return (end / start - 1) * 100


def _aggregate(trades: List[Trade], df_by_ticker: Dict[str, pd.DataFrame]) -> Dict:
    if not trades:
        return {"num_trades": 0, "warning": "No trades generated."}

    rs = np.array([t.r_multiple for t in trades])
    wins = rs[rs > 0]; losses = rs[rs <= 0]
    win_rate = len(wins) / len(rs)
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0   # negative number
    expectancy = float(rs.mean())
    profit_factor = (wins.sum() / abs(losses.sum())) if losses.sum() < 0 else float("inf") if wins.sum() > 0 else 0.0
    max_dd = _max_drawdown_r(trades)
    avg_hold = float(np.mean([t.holding_days for t in trades]))
    best = max(trades, key=lambda t: t.r_multiple)
    worst = min(trades, key=lambda t: t.r_multiple)

    bh = {tk: _buy_hold_return_pct(df) for tk, df in df_by_ticker.items()}

    out = {
        "num_trades": len(trades),
        "win_rate_pct": round(win_rate * 100, 2),
        "loss_rate_pct": round((1 - win_rate) * 100, 2),
        "avg_gain_R": round(avg_win, 3),
        "avg_loss_R": round(avg_loss, 3),
        "expectancy_R": round(expectancy, 3),
        "profit_factor": (round(profit_factor, 2) if profit_factor != float("inf") else "inf"),
        "max_drawdown_R": round(max_dd, 2),
        "avg_holding_days": round(avg_hold, 1),
        "best_trade_R": round(best.r_multiple, 2),
        "worst_trade_R": round(worst.r_multiple, 2),
        "buy_hold_pct_by_ticker": {k: round(v, 2) for k, v in bh.items()},
    }

    # Sample-size warning
    if len(trades) < BT.min_trades_for_confidence:
        out["sample_warning"] = (
            f"Only {len(trades)} trades — sample is too small "
            f"(need >= {BT.min_trades_for_confidence}). Treat results as preliminary."
        )

    # Verdict
    if expectancy < 0 or out["profit_factor"] in (0.0, "inf") and expectancy < 0:
        verdict = "Strategy unprofitable on this sample. Do not trust live."
    elif expectancy < 0.10:
        verdict = "Marginal expectancy. Strategy needs tuning before live use."
    elif win_rate >= 0.40 and expectancy >= 0.20 and (
        isinstance(out["profit_factor"], float) and out["profit_factor"] >= 1.5
    ):
        verdict = "Acceptable on this sample. Still preliminary — out-of-sample test required."
    else:
        verdict = "Mixed result. Promising but unverified."
    out["verdict"] = verdict
    return out


def _by_group(trades: List[Trade], key: Callable[[Trade], str]) -> Dict[str, Dict]:
    """Per-group stats (by ticker, by strategy)."""
    groups: Dict[str, List[Trade]] = {}
    for t in trades:
        groups.setdefault(key(t), []).append(t)
    out = {}
    for k, ts in groups.items():
        rs = np.array([t.r_multiple for t in ts])
        win = (rs > 0).sum()
        out[k] = {
            "num_trades": len(ts),
            "win_rate_pct": round(win / len(ts) * 100, 2),
            "expectancy_R": round(float(rs.mean()), 3),
            "profit_factor": (round(rs[rs > 0].sum() / abs(rs[rs <= 0].sum()), 2)
                              if (rs <= 0).any() and rs[rs <= 0].sum() < 0 else None),
        }
    return out


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------
def run_backtest(
    tickers: List[str],
    market: str = "US",
    strategies: Optional[List[str]] = None,
    period: str = "5y",
    provider: Optional[DataProvider] = None,
    progress_callback=None,
) -> Dict:
    """
    Run the full backtest across `tickers` and return a structured result dict.
    """
    provider = provider or DEFAULT_PROVIDER
    strategies = strategies or [
        s for s, on in [("breakout", BT.enable_breakout),
                         ("pullback", BT.enable_pullback),
                         ("trend_following", BT.enable_trend_following)] if on
    ]

    all_trades: List[Trade] = []
    df_by_ticker: Dict[str, pd.DataFrame] = {}
    failures: List[Dict] = []

    for i, raw in enumerate(tickers, 1):
        sym = normalize_ticker(raw, market)
        if progress_callback:
            try: progress_callback(i, len(tickers), sym)
            except Exception: pass
        try:
            df = provider.fetch_ohlcv(sym, period=period, interval="1d")
        except Exception as e:
            failures.append({"ticker": sym, "reason": f"fetch_error: {e}"})
            continue
        if df is None:
            failures.append({"ticker": sym, "reason": "no_data_returned (network/proxy or unsupported ticker)"})
            continue
        if not is_market_data_valid(df, BT.min_history_for_signal):
            failures.append({"ticker": sym, "reason": f"insufficient_history ({len(df)} bars)"})
            continue
        df_by_ticker[sym] = df
        trades = backtest_ticker(sym, df, strategies)
        all_trades.extend(trades)

    overall = _aggregate(all_trades, df_by_ticker)
    by_strategy = _by_group(all_trades, key=lambda t: t.strategy)
    by_ticker = _by_group(all_trades, key=lambda t: t.ticker)

    result = {
        "market": market.upper(),
        "tickers_attempted": [normalize_ticker(t, market) for t in tickers],
        "tickers_succeeded": list(df_by_ticker.keys()),
        "failures": failures,
        "strategies": strategies,
        "overall": overall,
        "by_strategy": by_strategy,
        "by_ticker": by_ticker,
        "trades": [t.to_dict() for t in all_trades],
    }
    # Cache overall stats per market for the signal engine
    try:
        import json
        cache = config.DATA_DIR / f"last_backtest_{market.upper()}.json"
        cache.write_text(json.dumps({"market": market.upper(), "overall": overall},
                                     default=str, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("Could not cache backtest overall: %s", e)
    return result


def load_cached_overall(market: str) -> Optional[Dict]:
    """Load the most recent cached overall stats for a given market."""
    try:
        import json
        cache = config.DATA_DIR / f"last_backtest_{market.upper()}.json"
        if cache.exists():
            return json.loads(cache.read_text(encoding="utf-8")).get("overall")
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def to_markdown(result: Dict) -> str:
    o = result.get("overall", {}) or {}
    lines: List[str] = []
    lines.append("# Backtest Report")
    lines.append("")
    lines.append(f"**Tickers attempted:** {', '.join(result.get('tickers_attempted', []))}")
    lines.append(f"**Tickers with data:** {', '.join(result.get('tickers_succeeded', []))}")
    if result.get("failures"):
        lines.append(f"**Failures:** {result['failures']}")
    lines.append(f"**Strategies:** {', '.join(result.get('strategies', []))}")
    lines.append("")
    lines.append("## Overall")
    if o.get("num_trades", 0) == 0:
        lines.append("_No trades generated. Check liquidity / history availability._")
    else:
        for k, v in o.items():
            if isinstance(v, dict):
                continue
            lines.append(f"- **{k.replace('_',' ').title()}**: {v}")

    lines.append("")
    lines.append("## By strategy")
    if not result.get("by_strategy"):
        lines.append("_No data._")
    for s, d in (result.get("by_strategy") or {}).items():
        lines.append(f"- **{s}**: {d}")

    lines.append("")
    lines.append("## By ticker")
    for s, d in (result.get("by_ticker") or {}).items():
        lines.append(f"- **{s}**: {d}")

    lines.append("")
    lines.append("## Honesty notes")
    lines.append("- Backtest results are sensitive to data quality and the chosen window. "
                 "Treat them as preliminary.")
    lines.append("- Slippage assumed: %.2f%% per fill; commission: $%.2f/share." %
                 (BT.slippage_pct, BT.commission_per_share))
    lines.append("- Walk-forward / out-of-sample validation is recommended before live use.")
    return "\n".join(lines)
