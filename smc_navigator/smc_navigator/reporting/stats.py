from dataclasses import asdict, dataclass
import json
from pathlib import Path

import pandas as pd

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
    performance_by_month: dict[str, float]
    performance_by_regime: dict[str, float]
    cagr: float
    sharpe_approx: float
    ulcer_index: float
    recovery_factor: float
    yearly_returns: dict[str, float]


def compute_trade_stats(trades: list[Trade], initial_capital: float = 100.0) -> TradeStats:
    total_trades = len(trades)
    wins = sum(1 for t in trades if (t.pnl or 0.0) > 0)
    losses = sum(1 for t in trades if (t.pnl or 0.0) < 0)
    winrate = (wins / total_trades * 100) if total_trades else 0.0
    gross_profit = sum((t.gross_pnl if t.gross_pnl > 0 else 0.0) for t in trades)
    gross_loss = sum((t.gross_pnl if t.gross_pnl < 0 else 0.0) for t in trades)
    net = sum(float(t.pnl or 0.0) for t in trades)
    fees = sum(float(t.total_fees or 0.0) for t in trades)
    avg = (net / total_trades) if total_trades else 0.0
    pf = (gross_profit / abs(gross_loss)) if gross_loss < 0 else 0.0

    eq = 0.0
    peak = 0.0
    mdd = 0.0
    drawdowns_pct: list[float] = []
    ret_series: list[float] = []

    hold = sum(t.holding_candles for t in trades) / total_trades if total_trades else 0.0
    by_symbol, by_dir, by_tag, by_month, by_regime, by_year = {}, {"LONG": 0.0, "SHORT": 0.0}, {}, {}, {}, {}

    sorted_trades = sorted(trades, key=lambda t: t.timestamp)
    for t in sorted_trades:
        pnl = float(t.pnl or 0.0)
        eq += pnl
        peak = max(peak, eq)
        dd = peak - eq
        mdd = max(mdd, dd)
        dd_pct = (dd / max(initial_capital + peak, 1e-9)) * 100
        drawdowns_pct.append(dd_pct)
        ret_series.append(pnl / max(initial_capital, 1e-9))

        by_symbol[t.symbol] = by_symbol.get(t.symbol, 0.0) + pnl
        by_dir[t.direction] = by_dir.get(t.direction, 0.0) + pnl
        month = pd.Timestamp(t.timestamp).strftime("%Y-%m")
        year = pd.Timestamp(t.timestamp).strftime("%Y")
        by_month[month] = by_month.get(month, 0.0) + pnl
        by_year[year] = by_year.get(year, 0.0) + pnl
        regime = "trend" if "trend_continuation" in (t.tags or "") else "range"
        by_regime[regime] = by_regime.get(regime, 0.0) + pnl
        for tag in [x.strip() for x in (t.tags or "").split("|") if x.strip()]:
            by_tag[tag] = by_tag.get(tag, 0.0) + pnl

    # approximate long-term metrics
    years = max(1.0, len(by_year))
    end_capital = initial_capital + net
    cagr = ((end_capital / max(initial_capital, 1e-9)) ** (1 / years) - 1) if end_capital > 0 else -1.0
    sharpe = (pd.Series(ret_series).mean() / pd.Series(ret_series).std()) if len(ret_series) > 1 and pd.Series(ret_series).std() > 0 else 0.0
    ulcer_index = (pd.Series(drawdowns_pct).pow(2).mean() ** 0.5) if drawdowns_pct else 0.0
    recovery_factor = (net / mdd) if mdd > 0 else 0.0

    return TradeStats(total_trades, wins, losses, winrate, gross_profit, gross_loss, net, fees, avg, pf, avg, mdd, hold, by_symbol, by_dir, by_tag, by_month, by_regime, cagr, float(sharpe), float(ulcer_index), float(recovery_factor), by_year)


def save_backtest_summary(stats: TradeStats, reports_dir: str | Path) -> None:
    rp = Path(reports_dir)
    rp.mkdir(parents=True, exist_ok=True)
    payload = asdict(stats)
    (rp / "backtest_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = [(k, v if not isinstance(v, dict) else json.dumps(v)) for k, v in payload.items()]
    (rp / "backtest_summary.csv").write_text("metric,value\n" + "\n".join(f"{k},{v}" for k, v in rows) + "\n", encoding="utf-8")
