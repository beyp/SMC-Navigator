from dataclasses import dataclass
from datetime import datetime
from typing import Literal

Status = Literal["OPEN", "WIN", "LOSS", "CLOSED_MANUAL", "EXPIRED"]


@dataclass
class Trade:
    trade_id: str
    timestamp: datetime
    exchange: str
    symbol: str
    timeframe: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    risk_amount: float
    confidence_score: int
    status: Status
    exit_price: float | None
    pnl: float | None
    pnl_pct: float | None
    reason: str

    def close(self, exit_price: float, status: Status) -> None:
        self.exit_price = exit_price
        if self.direction == "LONG":
            raw_pnl = (exit_price - self.entry_price) * self.position_size
        else:
            raw_pnl = (self.entry_price - exit_price) * self.position_size
        self.pnl = raw_pnl
        self.pnl_pct = (raw_pnl / (self.entry_price * self.position_size) * 100) if self.position_size else 0.0
        self.status = status
