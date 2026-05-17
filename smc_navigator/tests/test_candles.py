from smc_navigator.market_data.candles import fetch_candles_df
from smc_navigator.exchanges.base import BaseExchange


class FakeExchange(BaseExchange):
    def fetch_ohlcv(self, symbol: str, timeframe: str = "15m", limit: int = 200, since: int | None = None, max_fetch_batches: int = 1):
        # includes duplicate timestamp and unsorted rows
        return [
            [2000, 2, 2.2, 1.8, 2.1, 10],
            [1000, 1, 1.2, 0.8, 1.1, 10],
            [2000, 2, 2.2, 1.8, 2.1, 10],
            [3000, 3, 3.2, 2.8, 3.1, 10],
        ]


def test_fetch_candles_deduplicate_and_sort() -> None:
    ex = FakeExchange()
    df = fetch_candles_df(ex, "ETH/EUR", "15m", limit=10, since="2025-01-01T00:00:00Z", max_fetch_batches=2)
    assert len(df) == 3
    assert list(df["timestamp"]) == sorted(df["timestamp"].tolist())


def test_fetch_candles_until_filter() -> None:
    ex = FakeExchange()
    df = fetch_candles_df(ex, "ETH/EUR", "15m", until="1970-01-01T00:00:02Z")
    assert len(df) == 2
