import pandas as pd

from smc_navigator.market_data.candles import fetch_candles_df
from smc_navigator.exchanges.base import BaseExchange


class FakeExchange(BaseExchange):
    def fetch_ohlcv(self, symbol: str, timeframe: str = "15m", limit: int = 200, since: int | None = None, max_fetch_batches: int = 1):
        return [
            [2000, 2, 2.2, 1.8, 2.1, 10],
            [1000, 1, 1.2, 0.8, 1.1, 10],
            [2000, 2, 2.2, 1.8, 2.1, 10],
            [3000, 3, 3.2, 2.8, 3.1, 10],
        ]


class KrakenExchange(BaseExchange):
    def fetch_ohlcv(self, symbol: str, timeframe: str = "15m", limit: int = 200, since: int | None = None, max_fetch_batches: int = 1):
        assert timeframe == "1d"
        ts = lambda s: int(pd.Timestamp(s).timestamp() * 1000)
        return [
            [ts("2025-01-03T00:00:00Z"), 100, 110, 90, 105, 10],
            [ts("2025-01-20T00:00:00Z"), 106, 120, 100, 115, 20],
            [ts("2025-02-02T00:00:00Z"), 116, 125, 110, 120, 30],
            [ts("2025-02-15T00:00:00Z"), 121, 130, 118, 128, 40],
        ]


def test_fetch_candles_deduplicate_and_sort() -> None:
    ex = FakeExchange()
    df, _ = fetch_candles_df(ex, "ETH/EUR", "15m", limit=10, since="2025-01-01T00:00:00Z", max_fetch_batches=2, refresh_market_data=True)
    assert len(df) == 3
    assert list(df["timestamp"]) == sorted(df["timestamp"].tolist())


def test_fetch_candles_until_filter() -> None:
    ex = FakeExchange()
    df, _ = fetch_candles_df(ex, "ETH/EUR", "15m", until="1970-01-01T00:00:02Z", refresh_market_data=True)
    assert len(df) == 2


def test_kraken_monthly_resample_from_daily() -> None:
    ex = KrakenExchange()
    df, src = fetch_candles_df(ex, "ETH/EUR", "1M", refresh_market_data=True)
    assert src in {"api", "cache+api"}
    assert len(df) == 2
    jan = df.iloc[0]
    feb = df.iloc[1]
    assert jan["open"] == 100
    assert jan["high"] == 120
    assert jan["low"] == 90
    assert jan["close"] == 115
    assert jan["volume"] == 30
    assert feb["open"] == 116
    assert feb["high"] == 130
    assert feb["low"] == 110
    assert feb["close"] == 128
    assert feb["volume"] == 70
