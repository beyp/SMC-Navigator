from dataclasses import asdict, dataclass
import json
from pathlib import Path

from smc_navigator.simulator.trade import Trade


@dataclass
class TradeStats:
    total_trades: int
    wins: int
    losses: int
    winrate: float
    gross_profit: float
    gross_loss: float
    net_pnl_after_fees: float
    total_fees_paid: float
    average_pnl_per_trade: float
    profit_factor: float
    expectancy: float
    max_drawdown: float
    average_holding_candles: float
    pnl_by_symbol: dict[str, float]
    pnl_by_direction: dict[str, float]


def compute_trade_stats(trades: list[Trade]) -> TradeStats:
    total_trades = len(trades)
    wins = sum(1 for t in trades if (t.pnl or 0.0) > 0)
    losses = sum(1 for t in trades if (t.pnl or 0.0) < 0)
    winrate = (wins / total_trades * 100) if total_trades else 0.0

    gross_profit = sum((t.gross_pnl if t.gross_pnl > 0 else 0.0) for t in trades)
    gross_loss = sum((t.gross_pnl if t.gross_pnl < 0 else 0.0) for t in trades)
    net_pnl_after_fees = sum(float(t.pnl or 0.0) for t in trades)
    total_fees_paid = sum(float(t.total_fees or 0.0) for t in trades)
    average_pnl_per_trade = (net_pnl_after_fees / total_trades) if total_trades else 0.0
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss < 0 else 0.0
    expectancy = average_pnl_per_trade

    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for t in trades:
        equity += float(t.pnl or 0.0)
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)

    average_holding_candles = sum(t.holding_candles for t in trades) / total_trades if total_trades else 0.0

    pnl_by_symbol: dict[str, float] = {}
    pnl_by_direction: dict[str, float] = {"LONG": 0.0, "SHORT": 0.0}
    for t in trades:
        pnl = float(t.pnl or 0.0)
        pnl_by_symbol[t.symbol] = pnl_by_symbol.get(t.symbol, 0.0) + pnl
        pnl_by_direction[t.direction] = pnl_by_direction.get(t.direction, 0.0) + pnl

    return TradeStats(total_trades, wins, losses, winrate, gross_profit, gross_loss, net_pnl_after_fees, total_fees_paid, average_pnl_per_trade, profit_factor, expectancy, max_drawdown, average_holding_candles, pnl_by_symbol, pnl_by_direction)


def save_backtest_summary(stats: TradeStats, reports_dir: str | Path) -> None:
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    payload = asdict(stats)
    json_path = reports_path / "backtest_summary.json"
    csv_path = reports_path / "backtest_summary.csv"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    flat_rows = [
        ("total_trades", payload["total_trades"]),
        ("wins", payload["wins"]),
        ("losses", payload["losses"]),
        ("winrate", payload["winrate"]),
        ("gross_profit", payload["gross_profit"]),
        ("gross_loss", payload["gross_loss"]),
        ("net_pnl_after_fees", payload["net_pnl_after_fees"]),
        ("total_fees_paid", payload["total_fees_paid"]),
        ("average_pnl_per_trade", payload["average_pnl_per_trade"]),
        ("profit_factor", payload["profit_factor"]),
        ("expectancy", payload["expectancy"]),
        ("max_drawdown", payload["max_drawdown"]),
        ("average_holding_candles", payload["average_holding_candles"]),
        ("pnl_by_symbol", json.dumps(payload["pnl_by_symbol"])),
        ("pnl_by_direction", json.dumps(payload["pnl_by_direction"])),
    ]
    csv_path.write_text("metric,value\n" + "\n".join(f"{k},{v}" for k, v in flat_rows) + "\n", encoding="utf-8")
