import ccxt

from smc_navigator.exchanges.base import BaseExchange


class BinanceExchange(BaseExchange):
    def __init__(self) -> None:
        self.client = ccxt.binance({"enableRateLimit": True})

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "15m",
        limit: int = 200,
        since: int | None = None,
        max_fetch_batches: int = 1,
    ) -> list[list[float]]:
        all_rows: list[list[float]] = []
        cursor = since
        for _ in range(max_fetch_batches):
            batch = self.client.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
            if not batch:
                break
            all_rows.extend(batch)
            if len(all_rows) >= limit * max_fetch_batches:
                break
            cursor = int(batch[-1][0]) + 1
        unique = {int(r[0]): r for r in all_rows}
        return [unique[k] for k in sorted(unique.keys())]
