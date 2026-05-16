from abc import ABC, abstractmethod


class BaseExchange(ABC):
    @abstractmethod
    def fetch_ohlcv(self, symbol: str, timeframe: str = "15m", limit: int = 200) -> list[list[float]]:
        raise NotImplementedError
