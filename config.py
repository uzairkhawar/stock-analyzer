"""Global configuration for the stock analyzer."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
CHARTS_DIR: Path = PROJECT_ROOT / "charts"
for _d in (DATA_DIR, REPORTS_DIR, CHARTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

MARKET_SUFFIX: Dict[str, str] = {"US": "", "SAUDI": ".SR", "GLOBAL": ""}
BENCHMARKS: Dict[str, str] = {"US": "^GSPC", "US_TECH": "^IXIC", "US_DOW": "^DJI",
                              "SAUDI": "^TASI.SR", "GLOBAL": "^GSPC"}
DEFAULT_BENCHMARK_PER_MARKET: Dict[str, str] = {"US": "^GSPC", "SAUDI": "^TASI.SR", "GLOBAL": "^GSPC"}

DEFAULT_UNIVERSE: Dict[str, List[str]] = {
    "US": ["AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","AVGO","JPM","V","MA","UNH","JNJ",
           "PG","HD","XOM","CVX","WMT","KO","PEP","COST","MRK","LLY","ABBV","ORCL","CRM","ADBE",
           "NFLX","AMD","INTC","QCOM","CSCO","TXN","IBM","GE","BA","CAT","DE","GS","MS","BAC",
           "WFC","T","VZ","DIS","MCD"],
    "SAUDI": ["2222.SR","1120.SR","2010.SR","7010.SR","1180.SR","2350.SR","4030.SR","4002.SR",
              "1211.SR","2380.SR"],
}

@dataclass(frozen=True)
class IndicatorConfig:
    sma_periods: tuple = (20, 50, 100, 200)
    ema_periods: tuple = (20, 50, 100, 200)
    rsi_period: int = 14
    rsi_healthy_low: float = 45.0
    rsi_healthy_high: float = 70.0
    rsi_overbought: float = 75.0
    rsi_oversold: float = 30.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    stoch_rsi_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    swing_lookback: int = 20
    volume_avg_period: int = 20
    volume_breakout_multiple: float = 1.5

INDICATORS = IndicatorConfig()

@dataclass(frozen=True)
class ScoringWeights:
    technical_trend: int = 25
    momentum: int = 15
    volume: int = 15
    fundamental: int = 25
    valuation: int = 10
    risk_reward: int = 10
    def total(self) -> int:
        return (self.technical_trend + self.momentum + self.volume
                + self.fundamental + self.valuation + self.risk_reward)

WEIGHTS = ScoringWeights()
assert WEIGHTS.total() == 100

CLASSIFICATION_THRESHOLDS: Dict[str, int] = {
    "Strong Candidate": 75, "Watchlist Candidate": 60,
    "Neutral": 45, "Weak": 30,
}

@dataclass(frozen=True)
class RiskConfig:
    atr_stop_multiplier: float = 1.5
    min_risk_reward: float = 2.0
    max_account_risk_pct: float = 1.0
    pullback_ema_period: int = 20
    breakout_lookback: int = 20

RISK = RiskConfig()

@dataclass(frozen=True)
class ScreenerConfig:
    min_avg_volume: int = 200_000
    min_price: float = 2.0
    max_price: float = 10_000.0
    require_above_50_sma: bool = True
    require_above_200_sma: bool = True
    rsi_min: float = 45.0
    rsi_max: float = 70.0
    min_rel_strength: float = 0.0
    rs_lookback_days: int = 63

SCREENER = ScreenerConfig()

@dataclass(frozen=True)
class RejectionConfig:
    overextension_pct_vs_sma50: float = 15.0
    overextension_pct_vs_sma20: float = 10.0
    max_stop_width_pct: float = 8.0
    resistance_proximity_atr: float = 1.0
    rsi_extreme_high: float = 78.0
    weak_market_rs_required_pct: float = 5.0
    min_history_bars: int = 200
    min_avg_volume: int = 200_000

REJECT = RejectionConfig()

@dataclass(frozen=True)
class BacktestConfig:
    slippage_pct: float = 0.10
    commission_per_share: float = 0.0
    max_holding_days: int = 60
    min_history_for_signal: int = 200
    min_trades_for_confidence: int = 20
    min_avg_volume: int = 200_000
    risk_per_trade_pct: float = 1.0
    initial_equity: float = 10_000.0
    enable_breakout: bool = True
    enable_pullback: bool = True
    enable_trend_following: bool = True
    breakout_lookback: int = 20
    breakout_volume_multiple: float = 1.3
    pullback_min_uptrend_bars: int = 50
    trend_rsi_min: float = 50.0
    trend_rsi_max: float = 70.0

BACKTEST = BacktestConfig()

CACHE_TTL_MINUTES: int = 30
CACHE_DIR: Path = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DISCLAIMER: str = (
    "DISCLAIMER: This tool is decision-support software for educational and "
    "research purposes. It is not financial advice, not a recommendation to "
    "buy or sell any security, and not a guarantee of future performance. "
    "Markets carry risk of substantial loss. Always perform independent "
    "research and consult a licensed advisor before making investment decisions."
)
