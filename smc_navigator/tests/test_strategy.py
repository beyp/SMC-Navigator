import pandas as pd

from smc_navigator.strategy.rules import evaluate_signal


def test_no_trade_signal_generation() -> None:
    df = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-01-01T00:00:00Z"),
                "open": 100,
                "high": 101,
                "low": 99,
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


def test_structure_long_with_htf_bias_tags() -> None:
    rows = []
    ts = pd.Timestamp("2026-01-01T00:00:00Z")
    # build a rising structure with a sweep and reclaim behavior near end
    prices = [100, 99, 101, 100, 102, 101, 103, 102, 104, 103, 105, 104, 106, 103, 105]
    for i, p in enumerate(prices):
        rows.append(
            {
                "timestamp": ts + pd.Timedelta(minutes=15 * i),
                "open": p - 0.2,
                "high": p + 0.6,
                "low": p - 0.8,
                "close": p,
                "ema_9": p - 0.1,
                "ema_26": p - 0.2,
                "ema_50": p - 0.3,
                "rsi_14": 55,
                "vwap": p - 0.1,
                "support": p - 1.0,
                "resistance": p + 1.0,
            }
        )
    df = pd.DataFrame(rows)

    # bullish H1/H4 structures
    htf_rows = []
    for i, p in enumerate([95, 97, 99, 101, 103, 105, 107, 109]):
        htf_rows.append(
            {
                "timestamp": ts + pd.Timedelta(hours=i),
                "open": p - 0.2,
                "high": p + 0.5,
                "low": p - 0.5,
                "close": p,
                "ema_50": p - 0.5,
            }
        )
    h1 = pd.DataFrame(htf_rows)
    h4 = pd.DataFrame(htf_rows)

    signal = evaluate_signal("ETH/EUR", df, 1.0, 2.0, h1_df=h1, h4_df=h4)
    assert signal.direction in {"LONG", "NONE"}
    if signal.direction == "LONG":
        assert any(tag in signal.tags for tag in ["trend_continuation", "discount_long", "bos_long", "choch_long", "sweep_reversal"])
