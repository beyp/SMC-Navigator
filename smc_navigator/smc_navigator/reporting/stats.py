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
    pnl_by_tag: dict[str, float]


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

    equity=peak=max_drawdown=0.0
    for t in trades:
        equity += float(t.pnl or 0.0)
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)

    average_holding_candles = sum(t.holding_candles for t in trades) / total_trades if total_trades else 0.0
    pnl_by_symbol, pnl_by_direction, pnl_by_tag = {}, {"LONG":0.0,"SHORT":0.0}, {}
    for t in trades:
        pnl=float(t.pnl or 0.0)
        pnl_by_symbol[t.symbol]=pnl_by_symbol.get(t.symbol,0.0)+pnl
        pnl_by_direction[t.direction]=pnl_by_direction.get(t.direction,0.0)+pnl
        for tag in [x.strip() for x in (t.tags or '').split('|') if x.strip()]:
            pnl_by_tag[tag]=pnl_by_tag.get(tag,0.0)+pnl

    return TradeStats(total_trades,wins,losses,winrate,gross_profit,gross_loss,net_pnl_after_fees,total_fees_paid,average_pnl_per_trade,profit_factor,expectancy,max_drawdown,average_holding_candles,pnl_by_symbol,pnl_by_direction,pnl_by_tag)


def save_backtest_summary(stats: TradeStats, reports_dir: str | Path) -> None:
    reports_path = Path(reports_dir); reports_path.mkdir(parents=True, exist_ok=True)
    payload = asdict(stats)
    (reports_path / "backtest_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows=[(k,v if not isinstance(v,dict) else json.dumps(v)) for k,v in payload.items()]
    (reports_path / "backtest_summary.csv").write_text("metric,value\n"+"\n".join(f"{k},{v}" for k,v in rows)+"\n", encoding="utf-8")
