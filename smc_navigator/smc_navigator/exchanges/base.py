from abc import ABC, abstractmethod


class BaseExchange(ABC):
    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "15m",
        limit: int = 200,
        since: int | None = None,
        max_fetch_batches: int = 1,
    ) -> list[list[float]]:
        raise NotImplementedError
