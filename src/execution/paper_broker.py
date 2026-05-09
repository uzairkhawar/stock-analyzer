"""Local paper-trading simulator. NO network, NO real orders."""
from __future__ import annotations
from datetime import datetime
from typing import Dict, List, Optional
from src.execution.broker_base import BrokerBase


class PaperBroker(BrokerBase):
    def __init__(self, starting_cash: float = 10_000.0):
        self.starting_cash = float(starting_cash)
        self.current_cash = float(starting_cash)
        self.positions: Dict[str, Dict] = {}
        self.trade_log: List[Dict] = []
        self._marks: Dict[str, float] = {}

    def submit_order(self, ticker: str, qty: int, entry: float,
                     stop_loss: float, tp1: Optional[float] = None,
                     tp2: Optional[float] = None, tp3: Optional[float] = None,
                     meta: Optional[Dict] = None) -> Dict:
        if qty < 1: return {"ok": False, "error": "qty<1"}
        cost = qty * entry
        if cost > self.current_cash:
            return {"ok": False, "error": f"insufficient cash ({cost:.2f}>{self.current_cash:.2f})"}
        if ticker in self.positions:
            return {"ok": False, "error": "position already open"}
        self.current_cash -= cost
        pos = {"ticker": ticker, "qty": qty, "entry": entry,
               "stop_loss": stop_loss, "tp1": tp1, "tp2": tp2, "tp3": tp3,
               "opened_at": datetime.now().isoformat(timespec="seconds"),
               "meta": meta or {}}
        self.positions[ticker] = pos
        self.trade_log.append({"action": "OPEN", **pos})
        self._marks[ticker] = entry
        return {"ok": True, "position": pos}

    def close_position(self, ticker: str, exit_price: float, reason: str = "manual") -> Dict:
        if ticker not in self.positions:
            return {"ok": False, "error": "no open position"}
        pos = self.positions.pop(ticker)
        proceeds = pos["qty"] * exit_price
        self.current_cash += proceeds
        pnl = (exit_price - pos["entry"]) * pos["qty"]
        rec = {"action": "CLOSE", "ticker": ticker, "qty": pos["qty"],
               "entry": pos["entry"], "exit": exit_price, "pnl": pnl,
               "reason": reason,
               "closed_at": datetime.now().isoformat(timespec="seconds")}
        self.trade_log.append(rec)
        self._marks.pop(ticker, None)
        return {"ok": True, "trade": rec}

    def mark_to_market(self, prices: Dict[str, float]) -> None:
        for t, p in prices.items():
            if t in self.positions:
                self._marks[t] = float(p)

    def portfolio_value(self) -> float:
        mv = sum(self.positions[t]["qty"] * self._marks.get(t, self.positions[t]["entry"])
                 for t in self.positions)
        return float(self.current_cash + mv)

    def open_count(self) -> int:
        return len(self.positions)

    def daily_pnl_pct(self) -> float:
        if self.starting_cash <= 0: return 0.0
        return (self.portfolio_value() - self.starting_cash) / self.starting_cash * 100.0
