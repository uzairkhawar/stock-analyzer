import json, logging
from datetime import datetime
from pathlib import Path
import config
from src.report_generator import analyze_stock
from src.notifications.telegram_notifier import TelegramNotifier
from src.data_loader import DEFAULT_PROVIDER
from src.signal_ledger import resolve_pending

logger = logging.getLogger("stock_analyzer.alerts")
STATE_PATH = config.DATA_DIR / "alert_state.json"
WATCHLIST_PATH = config.PROJECT_ROOT / "watchlist.json"
ALERT_SIGNALS = {"Strong Candidate","Wait for Breakout"}  # tightened based on 2-week performance

def _load_watchlist():
    if WATCHLIST_PATH.exists():
        return json.loads(WATCHLIST_PATH.read_text(encoding="utf-8-sig"))
    return {"US": [], "SAUDI": []}

def _load_state():
    if STATE_PATH.exists():
        try: return json.loads(STATE_PATH.read_text(encoding="utf-8-sig"))
        except Exception: return {}
    return {}

def _save_state(s): STATE_PATH.write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")

def _safe(v, fb="-"):
    if v is None: return fb
    try:
        if v != v: return fb
    except Exception: pass
    return v

def _format_message(ticker, market, sig, cp):
    lines = [f"*{sig.get('signal','?')}*  -  *{ticker}* ({market})",
             f"Confidence: {_safe(sig.get('confidence'))}  |  Setup: {_safe(sig.get('setup_status'))}"]
    if sig.get("entry") is not None:
        lines += [f"Entry: `{_safe(sig.get('entry'))}`   Stop: `{_safe(sig.get('stop_loss'))}`",
                  f"TP1: `{_safe(sig.get('tp1'))}`  TP2: `{_safe(sig.get('tp2'))}`  TP3: `{_safe(sig.get('tp3'))}`",
                  f"R:R TP1: `{_safe(sig.get('risk_reward_tp1'))}`"]
    if cp: lines.append(f"Gates: {cp.get('satisfied_count')}/{cp.get('total_gates')} satisfied")
    if sig.get("invalidation"): lines.append(f"Invalidation: {sig.get('invalidation')}")
    if sig.get("action_guidance"): lines.append(f"_{sig.get('action_guidance')}_")
    if sig.get("risk_warnings"): lines.append("_Warnings: " + " | ".join(sig["risk_warnings"][:3]) + "_")
    lines.append("_Decision-support only. Not financial advice._")
    return "\n".join(lines)

def _should_alert(prev, sig, cp):
    name = sig.get("signal")
    if name not in ALERT_SIGNALS: return False
    if not prev: return True
    if prev.get("last_signal") != name: return True
    # gates dedup removed - only label changes fire re-alerts
    return False

def _history_lookup(ticker):
    try:
        df = DEFAULT_PROVIDER.fetch_ohlcv(ticker, period="3mo", interval="1d")
        if df is not None and not df.empty:
            return df
    except Exception: pass
    return None

def run_once(notifier=None):
    notifier = notifier or TelegramNotifier()
    wl = _load_watchlist(); state = _load_state()
    sent, skipped, failed = 0, 0, []
    for market, tickers in wl.items():
        for t in tickers:
            try:
                r = analyze_stock(t, market=market)
                if not r.get("ok"):
                    failed.append({"ticker": t, "error": r.get("error")}); continue
                sig = r.get("final_signal") or {}; cp = r.get("conviction_path")
                key = f"{market}:{t.upper()}"; prev = state.get(key)
                if _should_alert(prev, sig, cp):
                    if notifier.send(_format_message(t, market, sig, cp)):
                        sent += 1
                        try:
                            from src.signal_ledger import record as _ledger_record
                            _ledger_record(ticker=t, market=market, signal=sig,
                                           price=(r.get("snapshot") or {}).get("price"))
                        except Exception as e:
                            logger.warning("Ledger record failed: %s", e)
                        state[key] = {"last_signal": sig.get("signal"),
                            "last_gates": (cp or {}).get("satisfied_count"),
                            "last_ts": datetime.utcnow().isoformat(timespec="seconds")+"Z"}
                else: skipped += 1
            except Exception as e:
                logger.error("alert for %s failed: %s", t, e)
                failed.append({"ticker": t, "error": str(e)})
    _save_state(state)
    resolved = 0
    try: resolved = resolve_pending(_history_lookup)
    except Exception as e: logger.warning("Ledger resolution failed: %s", e)
    return {"sent": sent, "skipped": skipped, "failed": failed,
            "resolved": resolved,
            "ts": datetime.utcnow().isoformat(timespec="seconds")+"Z"}
