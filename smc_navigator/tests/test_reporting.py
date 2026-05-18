from datetime import datetime, timezone

from smc_navigator.reporting.stats import compute_trade_stats
from smc_navigator.simulator.trade import Trade


def _trade(status: str, pnl: float, gross_pnl: float | None = None) -> Trade:
    gp = pnl if gross_pnl is None else gross_pnl
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
        gross_pnl=gp,
        entry_fee=0.2,
        exit_fee=0.2,
        total_fees=0.4,
        holding_candles=3,
        rr_ratio=2.0,
        reason="test",
        tags="trend_following|high_confidence",
    )


def test_compute_trade_stats() -> None:
    trades = [_trade("WIN", 2.0), _trade("LOSS", -1.0), _trade("WIN", 1.0)]
    stats = compute_trade_stats(trades)
    assert stats.total_trades == 3
    assert stats.wins == 2
    assert stats.losses == 1
    assert round(stats.winrate, 2) == 66.67
    assert stats.net_pnl_after_fees == 2.0
    assert round(stats.average_pnl_per_trade, 4) == round(2.0 / 3.0, 4)
    assert stats.max_drawdown >= 0


def test_fee_stats_present() -> None:
    trades = [_trade("WIN", 1.0, gross_pnl=1.4), _trade("LOSS", -0.5, gross_pnl=-0.1)]
    stats = compute_trade_stats(trades)
    assert stats.total_fees_paid == 0.8
    assert stats.average_holding_candles == 3
    assert "ETH/EUR" in stats.pnl_by_symbol


def test_plot_symbol_chart_handles_string_ohlc(monkeypatch, tmp_path) -> None:
    import pytest

    pd = pytest.importorskip("pandas")
    charts = pytest.importorskip("smc_navigator.reporting.charts")

    called = {"ok": False}

    def fake_plot(*args, **kwargs):
        called["ok"] = True

    monkeypatch.setattr(charts.mpf, "plot", fake_plot)

    df = pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00Z", "2026-01-01T00:15:00Z", "2026-01-01T00:30:00Z"],
            "open": ["100", "101", "bad"],
            "high": ["101", "102", "103"],
            "low": ["99", "100", "101"],
            "close": ["100.5", "101.5", "102.5"],
            "volume": ["1000", "1100", "1200"],
            "ema_9": ["100", "101", "102"],
            "ema_26": ["99", "100", "101"],
        }
    )

    charts.plot_symbol_chart(df, "ETH/EUR", tmp_path / "string_ohlc.png", trade=None, confidence_score=50)
    assert called["ok"] is True
