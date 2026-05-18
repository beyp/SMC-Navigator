from pathlib import Path

import pandas as pd

from smc_navigator.exchanges.base import BaseExchange


def _cache_path(exchange_name: str, symbol: str, timeframe: str) -> Path:
    safe_symbol = symbol.replace("/", "_")
    return Path("data/cache") / exchange_name.lower() / safe_symbol / f"{timeframe}.parquet"


def fetch_candles_df(
    exchange: BaseExchange,
    symbol: str,
    timeframe: str,
    limit: int = 200,
    since: str | None = None,
    max_fetch_batches: int = 1,
    until: str | None = None,
    refresh_market_data: bool = False,
) -> tuple[pd.DataFrame, str]:
    ex_name = exchange.__class__.__name__.replace("Exchange", "").lower()
    cache_file = _cache_path(ex_name, symbol, timeframe)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    cached_df = pd.DataFrame()
    source = "api"
    if cache_file.exists() and not refresh_market_data:
        try:
            cached_df = pd.read_parquet(cache_file)
            if not cached_df.empty:
                cached_df["timestamp"] = pd.to_datetime(cached_df["timestamp"], utc=True)
                source = "cache+api"
        except Exception:
            cached_df = pd.DataFrame()

    since_ts = pd.Timestamp(since) if since else None
    since_ms = int(since_ts.timestamp() * 1000) if since_ts is not None else None

    if not cached_df.empty and not refresh_market_data:
        last_cached = cached_df["timestamp"].max()
        last_cached_ms = int(last_cached.timestamp() * 1000) + 1
        if since_ms is None or last_cached_ms > since_ms:
            since_ms = last_cached_ms

    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since_ms, max_fetch_batches=max_fetch_batches)
    fresh_df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    if not fresh_df.empty:
        fresh_df["timestamp"] = pd.to_datetime(fresh_df["timestamp"], unit="ms", utc=True)

    if cached_df.empty and fresh_df.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]), "none"

    merged = pd.concat([cached_df, fresh_df], ignore_index=True) if not cached_df.empty else fresh_df
    merged = merged.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    if since:
        merged = merged[merged["timestamp"] >= pd.Timestamp(since)]
    if until:
        merged = merged[merged["timestamp"] <= pd.Timestamp(until)]
    merged = merged.reset_index(drop=True)

    merged.to_parquet(cache_file, index=False)
    if cache_file.exists() and fresh_df.empty:
        source = "cache"
    elif cache_file.exists() and not fresh_df.empty and not cached_df.empty:
        source = "cache+api"
    else:
        source = "api"
    return merged, source
