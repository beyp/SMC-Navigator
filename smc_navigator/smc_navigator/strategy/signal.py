from dataclasses import dataclass
from datetime import datetime
from typing import Literal

Direction = Literal["LONG", "SHORT", "NONE"]


@dataclass
class Signal:
    symbol: str
    timestamp: datetime
    direction: Direction
    confidence_score: int
    reason: list[str]
    entry_price: float
    suggested_stop_loss: float
    suggested_take_profit: float
