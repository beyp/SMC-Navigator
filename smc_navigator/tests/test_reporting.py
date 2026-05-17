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
        entry_fee=0.2,
        exit_fee=0.2,
        total_fees=0.4,
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


def test_plot_symbol_chart_handles_nan_indicators(tmp_path, monkeypatch) -> None:
    import pandas as pd

    from smc_navigator.reporting import charts

    captured = {}

    def fake_plot(*args, **kwargs):
        captured["called"] = True
        captured["addplot"] = kwargs.get("addplot")

    monkeypatch.setattr(charts.mpf, "plot", fake_plot)

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=10, freq="15min", tz="UTC"),
            "open": [100 + i for i in range(10)],
            "high": [101 + i for i in range(10)],
            "low": [99 + i for i in range(10)],
            "close": [100.5 + i for i in range(10)],
            "volume": [1000 for _ in range(10)],
            "ema_9": [float("nan") for _ in range(10)],
            "ema_26": [float("nan") for _ in range(10)],
            "ema_50": [float("nan") for _ in range(10)],
            "vwap": [float("nan") for _ in range(10)],
            "support": [float("nan") for _ in range(10)],
            "resistance": [float("nan") for _ in range(10)],
        }
    )

    charts.plot_symbol_chart(df=df, symbol="ETH/EUR", output_path=tmp_path / "chart.png", trade=None, confidence_score=50)

    assert captured.get("called") is True


def test_fee_stats_present() -> None:
    trades = [_trade("WIN", 1.0), _trade("LOSS", -0.5)]
    stats = compute_trade_stats(trades)
    assert stats.total_fees == 0.8
    assert stats.average_fees == 0.4
