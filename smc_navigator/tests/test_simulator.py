from datetime import datetime, timezone

import pandas as pd

from smc_navigator.simulator.engine import evaluate_trade_outcome, run_backtest_for_symbol
from smc_navigator.simulator.trade import Trade


def test_trade_pnl_calculation_long_win() -> None:
    trade = Trade(
        trade_id="t1", timestamp=datetime.now(timezone.utc), exchange="kraken", symbol="ETH/EUR", timeframe="15m",
        direction="LONG", entry_price=100, stop_loss=99, take_profit=102, position_size=1, risk_amount=1,
        confidence_score=70, status="OPEN", exit_price=None, pnl=None, pnl_pct=None, gross_pnl=0.0,
        entry_fee=0.0, exit_fee=0.0, total_fees=0.0, holding_candles=0, rr_ratio=2.0, reason="test",
    )
    future = pd.DataFrame([{"high": 101, "low": 100, "close": 100.5}, {"high": 102.5, "low": 100.5, "close": 102.2}])
    result = evaluate_trade_outcome(trade, future, taker_fee_pct=0.1, spread_pct=0.05)
    assert result.status == "WIN"
    assert result.pnl is not None and result.pnl < 2


def test_backtest_filters_and_cooldown() -> None:
    config = {
        "exchange": "kraken", "timeframe": "15m", "starting_capital": 100, "risk_per_trade_pct": 1,
        "default_stop_loss_pct": 1.0, "default_take_profit_pct": 2.0,
        "maker_fee_pct": 0.16, "taker_fee_pct": 0.26, "spread_pct": 0.05,
        "min_confidence_score": 90, "enable_long_trades": True, "enable_short_trades": True,
        "min_rr_ratio": 1.5, "max_trades_per_symbol": 5, "cooldown_candles_after_trade": 2,
        "max_distance_from_vwap_pct": 100.0, "max_distance_from_ema26_pct": 100.0,
    }
    rows = []
    ts = pd.Timestamp("2026-01-01T00:00:00Z")
    for i in range(75):
        price = 100 + i * 0.1
        rows.append({"timestamp": ts + pd.Timedelta(minutes=15 * i), "open": price, "high": price + 0.3, "low": price - 0.3, "close": price, "volume": 1000, "ema_9": price + 0.05, "ema_26": price - 0.05, "ema_50": price - 0.1, "rsi_14": 55, "vwap": price, "support": price - 0.5, "resistance": price + 0.5})
    df = pd.DataFrame(rows)
    trades = run_backtest_for_symbol(config, "ETH/EUR", df, "/tmp/test_journal.csv", warmup=60, max_holding_candles=5)
    assert len(trades) == 0  # blocked by confidence filter


def test_fees_impact_on_pnl() -> None:
    base = Trade(
        trade_id="t2", timestamp=datetime.now(timezone.utc), exchange="kraken", symbol="ETH/EUR", timeframe="15m",
        direction="LONG", entry_price=100, stop_loss=99, take_profit=102, position_size=1, risk_amount=1,
        confidence_score=70, status="OPEN", exit_price=None, pnl=None, pnl_pct=None, gross_pnl=0.0,
        entry_fee=0.0, exit_fee=0.0, total_fees=0.0, holding_candles=0, rr_ratio=2.0, reason="test",
    )
    future = pd.DataFrame([{"high": 102.5, "low": 100.5, "close": 102.2}])
    no_fee = evaluate_trade_outcome(base, future, taker_fee_pct=0.0, spread_pct=0.0)

    with_fee = Trade(**{**base.__dict__, "trade_id": "t3", "status": "OPEN", "exit_price": None, "pnl": None})
    with_fee = evaluate_trade_outcome(with_fee, future, taker_fee_pct=0.3, spread_pct=0.05)
    assert with_fee.pnl is not None and no_fee.pnl is not None
    assert with_fee.pnl < no_fee.pnl


def test_min_rr_and_cooldown_filters() -> None:
    config = {
        "exchange": "kraken", "timeframe": "15m", "starting_capital": 100, "risk_per_trade_pct": 1,
        "default_stop_loss_pct": 2.0, "default_take_profit_pct": 1.0,
        "maker_fee_pct": 0.16, "taker_fee_pct": 0.26, "spread_pct": 0.0,
        "min_confidence_score": 0, "enable_long_trades": True, "enable_short_trades": True,
        "min_rr_ratio": 2.0, "max_trades_per_symbol": 5, "cooldown_candles_after_trade": 3,
        "max_distance_from_vwap_pct": 100.0, "max_distance_from_ema26_pct": 100.0,
    }
    rows=[]
    ts = pd.Timestamp("2026-01-01T00:00:00Z")
    for i in range(90):
        price=100+i*0.1
        rows.append({"timestamp": ts + pd.Timedelta(minutes=15*i), "open": price, "high": price+0.5, "low": price-0.5, "close": price, "volume":1000, "ema_9": price+0.05, "ema_26": price-0.05, "ema_50": price-0.1, "rsi_14":55, "vwap": price, "support": price-0.2, "resistance": price+0.2})
    df=pd.DataFrame(rows)
    trades=run_backtest_for_symbol(config, "ETH/EUR", df, "/tmp/test_journal2.csv", warmup=60, max_holding_candles=2)
    assert len(trades) == 0  # RR below threshold (tp<sl distance)
