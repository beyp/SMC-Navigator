import csv
from pathlib import Path

from smc_navigator.simulator.trade import Trade


HEADER = [
    "trade_id", "timestamp", "exchange", "symbol", "timeframe", "direction", "entry_price", "stop_loss",
    "take_profit", "position_size", "risk_amount", "confidence_score", "status", "exit_price", "pnl",
    "pnl_pct", "entry_fee", "exit_fee", "total_fees", "reason",
]


def append_trade(path: str | Path, trade: Trade) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        if not exists:
            writer.writeheader()
        writer.writerow(trade.__dict__)
