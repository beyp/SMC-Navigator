import pandas as pd

from smc_navigator.strategy.rules import evaluate_signal


def test_no_trade_signal_generation() -> None:
    df = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-01-01T00:00:00Z"),
                "close": 100,
                "ema_9": 100,
                "ema_26": 100,
                "ema_50": 100,
                "rsi_14": 50,
                "vwap": 120,
                "support": 80,
                "resistance": 130,
            }
        ]
    )
    signal = evaluate_signal("ETH/EUR", df, 1.0, 2.0)
    assert signal.direction == "NONE"
    assert signal.confidence_score == 20
    assert isinstance(signal.tags, list)
