"""Append-only signal ledger with range-walking resolver."""
import hashlib, json
from datetime import datetime
from pathlib import Path
import config

LEDGER_PATH = config.DATA_DIR / "signal_ledger.jsonl"
LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)

def _canonical(d): return json.dumps(d, sort_keys=True, default=str, separators=(",",":"))

def record(*, ticker, market, signal, price=None):
    payload = {"ts": datetime.utcnow().isoformat(timespec="seconds")+"Z",
        "ticker": ticker.upper(), "market": market.upper(),
        "signal": signal.get("signal"), "confidence": signal.get("confidence"),
        "setup_status": signal.get("setup_status"),
        "entry": signal.get("entry"), "stop_loss": signal.get("stop_loss"),
        "tp1": signal.get("tp1"), "tp2": signal.get("tp2"), "tp3": signal.get("tp3"),
        "price_at_signal": price, "outcome": None, "resolved_at": None}
    payload["content_hash"] = hashlib.sha256(_canonical({k:v for k,v in payload.items() if k!="content_hash"}).encode()).hexdigest()[:16]
    with LEDGER_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")
    return payload

def load_all():
    if not LEDGER_PATH.exists(): return []
    out = []
    for line in LEDGER_PATH.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line: continue
        try: out.append(json.loads(line))
        except Exception: continue
    return out

def resolve_pending(history_lookup):
    """history_lookup(ticker) -> DataFrame indexed by date with High/Low/Close columns."""
    rows = load_all(); changed = 0
    by_ticker = {}
    for r in rows:
        if r.get("outcome"): continue
        by_ticker.setdefault(r.get("ticker"), []).append(r)
    for ticker, group in by_ticker.items():
        try: df = history_lookup(ticker)
        except Exception: df = None
        if df is None or df.empty:
            continue
        for r in group:
            entry = r.get("entry"); sl = r.get("stop_loss"); tp1 = r.get("tp1")
            if not (entry and sl):
                r["outcome"] = "no_levels"
                r["resolved_at"] = datetime.utcnow().isoformat(timespec="seconds")+"Z"
                changed += 1; continue
            try:
                entry_ts = datetime.fromisoformat((r.get("ts") or "").rstrip("Z"))
            except Exception:
                continue
            sub = df[df.index >= entry_ts]
            for idx, bar in sub.iterrows():
                low = float(bar["Low"]); high = float(bar["High"])
                if low <= sl:
                    r["outcome"] = "hit_stop"; break
                if tp1 and high >= tp1:
                    r["outcome"] = "hit_tp1"; break
            if not r.get("outcome"): continue
            r["resolved_at"] = datetime.utcnow().isoformat(timespec="seconds")+"Z"
            changed += 1
    if changed:
        LEDGER_PATH.write_text("\n".join(json.dumps(r, default=str) for r in rows) + "\n", encoding="utf-8")
    return changed

def track_record():
    rows = load_all(); by_signal = {}
    for r in rows:
        s = r.get("signal","unknown")
        b = by_signal.setdefault(s, {"count":0,"resolved":0,"hit_tp1":0,"hit_stop":0})
        b["count"] += 1
        oc = r.get("outcome")
        if oc in ("hit_tp1","hit_stop"): b["resolved"] += 1; b[oc] += 1
    for s, b in by_signal.items():
        b["hit_rate_pct"] = round(b["hit_tp1"]/b["resolved"]*100,1) if b["resolved"] else None
    return {"total_signals": len(rows), "by_signal": by_signal, "ledger_path": str(LEDGER_PATH)}
