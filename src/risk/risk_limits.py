"""Pre-trade risk gate for paper trades."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional


ALLOWED_SIGNALS = {"Strong Candidate", "Watchlist Candidate"}


@dataclass
class RiskConfig:
    risk_per_trade_pct: float = 1.0
    max_open_positions: int = 5
    daily_loss_limit_pct: float = 3.0


def position_size(account_equity: float, entry: float, stop_loss: float,
                  risk_pct: float) -> int:
    if entry is None or stop_loss is None or entry <= 0 or stop_loss <= 0:
        return 0
    rps = abs(entry - stop_loss)
    if rps <= 0:
        return 0
    risk_amount = account_equity * (risk_pct / 100.0)
    return int(math.floor(risk_amount / rps))


def check_order(*, signal: Dict, account_equity: float, qty: int,
                open_positions: int, daily_pnl_pct: float,
                cfg: RiskConfig) -> Dict:
    """Returns {'allowed': bool, 'reasons': [...]} for a paper order."""
    reasons: List[str] = []
    sig_name = (signal or {}).get("signal", "")
    entry = (signal or {}).get("entry")
    sl = (signal or {}).get("stop_loss")

    if sig_name not in ALLOWED_SIGNALS:
        reasons.append(f"Signal '{sig_name}' not tradable.")
    if entry is None:
        reasons.append("Entry price missing.")
    if sl is None:
        reasons.append("Stop loss missing.")
    if qty < 1:
        reasons.append(f"Quantity {qty} < 1.")
    if open_positions >= cfg.max_open_positions:
        reasons.append(f"Max open positions reached ({cfg.max_open_positions}).")
    if daily_pnl_pct <= -cfg.daily_loss_limit_pct:
        reasons.append(f"Daily loss limit hit ({daily_pnl_pct:.2f}%).")
    if entry and sl:
        rps = abs(entry - sl)
        risk_amount = qty * rps
        if risk_amount > account_equity * (cfg.risk_per_trade_pct / 100.0) * 1.05:
            reasons.append(
                f"Risk ${risk_amount:.2f} exceeds {cfg.risk_per_trade_pct}% of equity."
            )
    return {"allowed": not reasons, "reasons": reasons}
