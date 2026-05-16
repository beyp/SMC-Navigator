from datetime import datetime, timezone

from smc_navigator.reporting.stats import compute_trade_stats
from smc_navigator.simulator.trade import Trade


def _trade(status: str, pnl: float) -> Trade:
    return Trade(
        trade_id="x",
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
        confidence_score=60,
        status=status,
        exit_price=100 + pnl,
        pnl=pnl,
        pnl_pct=pnl,
        reason="test",
    )


def test_compute_trade_stats() -> None:
    trades = [_trade("WIN", 2.0), _trade("LOSS", -1.0), _trade("WIN", 1.0)]
    stats = compute_trade_stats(trades)
    assert stats.total_trades == 3
    assert stats.wins == 2
    assert stats.losses == 1
    assert round(stats.winrate, 2) == 66.67
    assert stats.total_pnl == 2.0
    assert round(stats.average_pnl, 4) == round(2.0 / 3.0, 4)
    assert stats.max_drawdown >= 0
