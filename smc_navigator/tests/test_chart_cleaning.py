import pytest


def test_plot_symbol_chart_handles_string_ohlc_from_cache(monkeypatch, tmp_path) -> None:
    pd = pytest.importorskip("pandas")
    charts = pytest.importorskip("smc_navigator.reporting.charts")

    called = {"ok": False}

    def fake_plot(*args, **kwargs):
        called["ok"] = True

    monkeypatch.setattr(charts.mpf, "plot", fake_plot)

    # Simulate cache/parquet-loaded object dtypes
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


def test_plot_symbol_chart_handles_timestamp_in_index(monkeypatch, tmp_path) -> None:
    pd = pytest.importorskip("pandas")
    charts = pytest.importorskip("smc_navigator.reporting.charts")

    called = {"ok": False}

    def fake_plot(*args, **kwargs):
        called["ok"] = True

    monkeypatch.setattr(charts.mpf, "plot", fake_plot)

    df = pd.DataFrame(
        {
            "open": ["100", "101"],
            "high": ["101", "102"],
            "low": ["99", "100"],
            "close": ["100.5", "101.5"],
            "volume": ["1000", "1100"],
        },
        index=pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T00:15:00Z"]),
    )
    df.index.name = "timestamp"

    charts.plot_symbol_chart(df, "BTC/EUR", tmp_path / "index_ts.png", trade=None, confidence_score=40)
    assert called["ok"] is True
