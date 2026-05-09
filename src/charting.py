"""
Charting — produce a multi-panel chart with candles, MAs, S/R, entry/SL/TP,
volume, RSI and MACD. Uses matplotlib so the PNG export works headless.

For interactive in-app charting we use Plotly (built inline in app.py); this
module specializes in static, presentation-quality PNGs for reports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

import config


def _draw_candles(ax, df: pd.DataFrame) -> None:
    """Manual candlestick drawing (avoids mplfinance dependency)."""
    width_body = 0.6
    width_wick = 0.1
    for i, (idx, row) in enumerate(df.iterrows()):
        open_, close = row["Open"], row["Close"]
        high, low = row["High"], row["Low"]
        color = "#26a69a" if close >= open_ else "#ef5350"
        # Wick
        ax.add_patch(Rectangle(
            (i - width_wick / 2, low), width_wick, high - low,
            facecolor=color, edgecolor=color, linewidth=0))
        # Body
        body_low = min(open_, close)
        body_h = max(abs(close - open_), high * 0.0005)
        ax.add_patch(Rectangle(
            (i - width_body / 2, body_low), width_body, body_h,
            facecolor=color, edgecolor=color, linewidth=0))
    ax.set_xlim(-1, len(df))


def plot_full_analysis(
    df: pd.DataFrame, snap: Dict, setup: Optional[Dict],
    ticker: str, output_path: Optional[Path] = None,
    last_n: int = 180,
) -> Path:
    """Render and save the multi-panel chart. Returns the output path."""
    if df is None or df.empty:
        raise ValueError("Empty dataframe")
    df = df.tail(last_n).copy()

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.05)
    ax_price = fig.add_subplot(gs[0])
    ax_vol   = fig.add_subplot(gs[1], sharex=ax_price)
    ax_rsi   = fig.add_subplot(gs[2], sharex=ax_price)
    ax_macd  = fig.add_subplot(gs[3], sharex=ax_price)

    # ---- Price panel ----
    _draw_candles(ax_price, df)
    for p, color in [(20, "#1976d2"), (50, "#fb8c00"), (200, "#6a1b9a")]:
        col = f"SMA_{p}"
        if col in df.columns and df[col].notna().any():
            ax_price.plot(np.arange(len(df)), df[col].values,
                          label=f"SMA{p}", color=color, linewidth=1.2)

    # Nearest support / resistance only (reduce clutter)
    levels = (snap or {}).get("levels", {}) or {}
    for label, val, style in [
        ("Resistance", levels.get("nearest_resistance"), {"color": "#d32f2f", "linestyle": "--"}),
        ("Support",    levels.get("nearest_support"),    {"color": "#388e3c", "linestyle": "--"}),
    ]:
        if val:
            ax_price.axhline(val, linewidth=1, alpha=0.7, **style)
            ax_price.text(len(df) - 1, val, f" {label} {val:.2f}",
                          fontsize=8, color=style["color"], va="center")

    # Current price annotation
    cur_price = float(df["Close"].iloc[-1])
    ax_price.axhline(cur_price, color="#000", linewidth=0.8, alpha=0.6,
                     label=f"Current {cur_price:.2f}")

    # Trade setup overlay (relabel based on price vs entry)
    if setup:
        e = setup["entry"]; sl = setup["stop_loss"]
        tp1 = setup["tp1"]; tp2 = setup["tp2"]; tp3 = setup.get("tp3")
        entry_label = ("Breakout trigger" if e > cur_price else "Entry now")
        ax_price.axhspan(e * 0.998, e * 1.002, color="#1976d2", alpha=0.15,
                         label=f"{entry_label} {e:.2f}")
        ax_price.axhline(sl, color="#c62828", linewidth=1.4, label=f"Stop {sl:.2f}")
        ax_price.axhline(tp1, color="#2e7d32", linewidth=1.0, linestyle="-.", label=f"TP1 {tp1:.2f}")
        ax_price.axhline(tp2, color="#2e7d32", linewidth=1.0, linestyle="--", label=f"TP2 {tp2:.2f}")
        if tp3:
            ax_price.axhline(tp3, color="#2e7d32", linewidth=1.0, linestyle=":",  label=f"TP3 {tp3:.2f}")

    ax_price.set_title(f"{ticker} — Technical Analysis", fontsize=13, fontweight="bold")
    ax_price.set_ylabel("Price")
    ax_price.grid(alpha=0.25)
    ax_price.legend(loc="upper left", fontsize=8, ncol=3)

    # ---- Volume ----
    colors = ["#26a69a" if c >= o else "#ef5350"
              for o, c in zip(df["Open"], df["Close"])]
    ax_vol.bar(np.arange(len(df)), df["Volume"].values, color=colors, alpha=0.8, width=0.7)
    if "Volume" in df.columns:
        vavg = df["Volume"].rolling(20).mean()
        ax_vol.plot(np.arange(len(df)), vavg.values, color="#1565c0", linewidth=1.0, label="Vol 20-MA")
    ax_vol.set_ylabel("Volume")
    ax_vol.grid(alpha=0.25)
    ax_vol.legend(loc="upper left", fontsize=8)

    # ---- RSI ----
    if "RSI" in df.columns:
        ax_rsi.plot(np.arange(len(df)), df["RSI"].values, color="#7b1fa2", linewidth=1.1)
        ax_rsi.axhline(70, color="#d32f2f", linestyle="--", linewidth=0.8)
        ax_rsi.axhline(30, color="#388e3c", linestyle="--", linewidth=0.8)
        ax_rsi.set_ylim(0, 100)
        ax_rsi.set_ylabel("RSI(14)")
        ax_rsi.grid(alpha=0.25)

    # ---- MACD ----
    if "MACD" in df.columns:
        ax_macd.plot(np.arange(len(df)), df["MACD"].values, color="#1976d2", linewidth=1.1, label="MACD")
        ax_macd.plot(np.arange(len(df)), df["MACD_Signal"].values, color="#fb8c00", linewidth=1.1, label="Signal")
        hist = df["MACD_Hist"].values
        hcolors = ["#2e7d32" if h >= 0 else "#c62828" for h in hist]
        ax_macd.bar(np.arange(len(df)), hist, color=hcolors, alpha=0.5, width=0.8)
        ax_macd.axhline(0, color="#666", linewidth=0.6)
        ax_macd.set_ylabel("MACD")
        ax_macd.grid(alpha=0.25)
        ax_macd.legend(loc="upper left", fontsize=8)

    # X-axis: show date ticks at sparse positions
    n = len(df)
    ticks = list(range(0, n, max(1, n // 8)))
    ax_macd.set_xticks(ticks)
    ax_macd.set_xticklabels([df.index[t].strftime("%Y-%m-%d") for t in ticks],
                            rotation=30, ha="right", fontsize=8)
    for ax in (ax_price, ax_vol, ax_rsi):
        plt.setp(ax.get_xticklabels(), visible=False)

    out = output_path or (config.CHARTS_DIR / f"{ticker.replace('.', '_')}_analysis.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out
