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
    entry_fee: float
    exit_fee: float
    total_fees: float
    reason: str

    def close(self, exit_price: float, status: Status, entry_fee: float = 0.0, exit_fee: float = 0.0) -> None:
        self.exit_price = exit_price
        self.entry_fee = entry_fee
        self.exit_fee = exit_fee
        self.total_fees = entry_fee + exit_fee

        if self.direction == "LONG":
            gross_pnl = (exit_price - self.entry_price) * self.position_size
        else:
            gross_pnl = (self.entry_price - exit_price) * self.position_size

        net_pnl = gross_pnl - self.total_fees
        self.pnl = net_pnl
        self.pnl_pct = (net_pnl / (self.entry_price * self.position_size) * 100) if self.position_size else 0.0
        self.status = status
