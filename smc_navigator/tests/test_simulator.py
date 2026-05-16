from datetime import datetime, timezone

import pandas as pd

from smc_navigator.simulator.engine import simulate_signal
from smc_navigator.strategy.signal import Signal


def test_trade_pnl_calculation(tmp_path) -> None:
    config = {
        "exchange": "kraken",
        "timeframe": "15m",
        "starting_capital": 100,
        "risk_per_trade_pct": 1,
    }
    signal = Signal(
        symbol="ETH/EUR",
        timestamp=datetime.now(timezone.utc),
        direction="LONG",
        confidence_score=70,
        reason=["test"],
        entry_price=100,
        suggested_stop_loss=99,
        suggested_take_profit=102,
    )
    df = pd.DataFrame(
        [
            {"high": 101, "low": 99.5},
            {"high": 102.5, "low": 100.5},
        ]
    )
    journal_path = tmp_path / "journal.csv"
    trade = simulate_signal(config, signal, df, str(journal_path))
    assert trade is not None
    assert trade.status == "WIN"
    assert trade.pnl is not None and trade.pnl > 0
