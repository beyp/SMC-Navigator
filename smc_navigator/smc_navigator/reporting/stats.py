from dataclasses import dataclass

from smc_navigator.simulator.trade import Trade


@dataclass
class TradeStats:
    total_trades: int
    wins: int
    losses: int
    winrate: float
    total_pnl: float
    average_pnl: float
    max_drawdown: float
    total_fees: float
    average_fees: float


def compute_trade_stats(trades: list[Trade]) -> TradeStats:
    total_trades = len(trades)
    wins = sum(1 for t in trades if t.status == "WIN")
    losses = sum(1 for t in trades if t.status == "LOSS")
    winrate = (wins / total_trades * 100) if total_trades else 0.0

    pnls = [float(t.pnl or 0.0) for t in trades]
    fee_values = [float(getattr(t, "total_fees", 0.0) or 0.0) for t in trades]
    total_pnl = sum(pnls)
    average_pnl = (total_pnl / total_trades) if total_trades else 0.0
    total_fees = sum(fee_values)
    average_fees = (total_fees / total_trades) if total_trades else 0.0

    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        drawdown = peak - equity
        max_drawdown = max(max_drawdown, drawdown)

    return TradeStats(total_trades, wins, losses, winrate, total_pnl, average_pnl, max_drawdown, total_fees, average_fees)
