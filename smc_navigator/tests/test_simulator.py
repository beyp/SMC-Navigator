from datetime import datetime, timezone

import pandas as pd

from smc_navigator.simulator.engine import evaluate_trade_outcome, run_backtest_for_symbol
from smc_navigator.simulator.trade import Trade


def test_trade_pnl_calculation_long_win() -> None:
    trade = Trade(
        trade_id="t1",
        timestamp=datetime.now(timezone.utc),
        exchange="kraken",
        symbol="ETH/EUR",
        timeframe="15m",
        direction="LONG",
        entry_price=100,
        stop_loss=99,
        take_profit=102,
        position_size=1,
        risk_amount=1,
        confidence_score=70,
        status="OPEN",
        exit_price=None,
        pnl=None,
        pnl_pct=None,
        reason="test",
    )

    future = pd.DataFrame([
        {"high": 101, "low": 100, "close": 100.5},
        {"high": 102.5, "low": 100.5, "close": 102.2},
    ])
    result = evaluate_trade_outcome(trade, future)
    assert result.status == "WIN"
    assert result.pnl == 2


def test_backtest_avoids_lookahead_bias() -> None:
    config = {
        "exchange": "kraken",
        "timeframe": "15m",
        "starting_capital": 100,
        "risk_per_trade_pct": 1,
        "default_stop_loss_pct": 1.0,
        "default_take_profit_pct": 2.0,
    }

    rows = []
    ts = pd.Timestamp("2026-01-01T00:00:00Z")
    for i in range(75):
        price = 100 + i * 0.1
        rows.append(
            {
                "timestamp": ts + pd.Timedelta(minutes=15 * i),
                "open": price,
                "high": price + 0.3,
                "low": price - 0.3,
                "close": price,
                "volume": 1000,
                "ema_9": price + 0.05,
                "ema_26": price - 0.05,
                "ema_50": price - 0.1,
                "rsi_14": 55,
                "vwap": price,
                "support": price - 0.5,
                "resistance": price + 0.5,
            }
        )

    df = pd.DataFrame(rows)
    trades = run_backtest_for_symbol(config, "ETH/EUR", df, "/tmp/test_journal.csv", warmup=60, max_holding_candles=5)

    assert len(trades) >= 1
    first_trade = trades[0]
    assert "signal_index=" in first_trade.reason
