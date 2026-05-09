"""Abstract broker interface. NO live execution implementations belong here."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Optional


class BrokerBase(ABC):
    @abstractmethod
    def submit_order(self, ticker: str, qty: int, entry: float,
                     stop_loss: float, tp1: Optional[float] = None,
                     tp2: Optional[float] = None, tp3: Optional[float] = None,
                     meta: Optional[Dict] = None) -> Dict: ...

    @abstractmethod
    def close_position(self, ticker: str, exit_price: float, reason: str = "manual") -> Dict: ...

    @abstractmethod
    def mark_to_market(self, prices: Dict[str, float]) -> None: ...

    @abstractmethod
    def portfolio_value(self) -> float: ...
