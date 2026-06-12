"""Daily report: today's signals + resolutions + rolling stats. Sent via Telegram."""
import sys, pathlib
from datetime import datetime, timedelta
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    from dotenv import load_dotenv; load_dotenv(ROOT / ".env")
except Exception: pass

from src.signal_ledger import load_all
from src.notifications.telegram_notifier import TelegramNotifier

rows = load_all()
now = datetime.utcnow()
today_start = datetime(now.year, now.month, now.day)

def ts(r, key="ts"):
    try: return datetime.fromisoformat((r.get(key) or "").rstrip("Z"))
    except Exception: return None

today_signals = [r for r in rows if (ts(r) or datetime.min) >= today_start]
today_resolved = [r for r in rows if (ts(r,"resolved_at") or datetime.min) >= today_start]

cutoff = now - timedelta(days=30)
rolling = [r for r in rows if (ts(r) or datetime.min) >= cutoff]
resolved = [r for r in rolling if r.get("outcome") in ("hit_tp1","hit_stop")]
wins = [r for r in resolved if r["outcome"] == "hit_tp1"]
wr = round(len(wins)/len(resolved)*100, 1) if resolved else None

lines = [f"*Daily Report - {now.strftime('%Y-%m-%d')}*", ""]
lines += [f"*Today*: {len(today_signals)} new signals, {len(today_resolved)} resolved", ""]
if today_signals:
    lines.append("*New Today*")
    for r in today_signals[-10:]:
        lines.append(f"- {r.get('ticker')}: {r.get('signal')}")
    lines.append("")
if today_resolved:
    lines.append("*Resolved Today*")
    for r in today_resolved[-10:]:
        oc = "WIN" if r.get("outcome")=="hit_tp1" else "LOSS"
        lines.append(f"- {r.get('ticker')}: {r.get('signal')} -> {oc}")
    lines.append("")
lines += [
    "*Rolling 30 days*",
    f"Signals: {len(rolling)} | Resolved: {len(resolved)}",
    f"Hit rate (TP1 vs SL): {wr}%" if wr is not None else "Hit rate: not enough data",
    f"Total signals all-time: {len(rows)}",
    "",
    "_Honest stats from immutable ledger. Decision-support only._",
]
msg = "\n".join(lines)
TelegramNotifier().send(msg)
print(msg)
