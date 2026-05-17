import pandas as pd

from smc_navigator.exchanges.base import BaseExchange


def fetch_candles_df(
    exchange: BaseExchange,
    symbol: str,
    timeframe: str,
    limit: int = 200,
    since: str | None = None,
    max_fetch_batches: int = 1,
    until: str | None = None,
) -> pd.DataFrame:
    since_ms = int(pd.Timestamp(since).timestamp() * 1000) if since else None
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since_ms, max_fetch_batches=max_fetch_batches)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    if until:
        df = df[df["timestamp"] <= pd.Timestamp(until)]
    return df.reset_index(drop=True)
