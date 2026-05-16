import ccxt

from smc_navigator.exchanges.base import BaseExchange


class KrakenExchange(BaseExchange):
    def __init__(self) -> None:
        self.client = ccxt.kraken({"enableRateLimit": True})

    def fetch_ohlcv(self, symbol: str, timeframe: str = "15m", limit: int = 200) -> list[list[float]]:
        return self.client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
