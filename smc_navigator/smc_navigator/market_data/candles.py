from pathlib import Path
import time
import logging

import ccxt
import pandas as pd

from smc_navigator.exchanges.base import BaseExchange

LOGGER = logging.getLogger(__name__)


def _cache_path(exchange_name: str, symbol: str, timeframe: str) -> Path:
    safe_symbol = symbol.replace("/", "_")
    return Path("data/cache") / exchange_name.lower() / safe_symbol / f"{timeframe}.parquet"


def _fetch_with_retry(
    exchange: BaseExchange,
    symbol: str,
    timeframe: str,
    limit: int,
    since_ms: int | None,
    max_fetch_batches: int,
    request_delay_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
) -> list[list[float]]:
    for attempt in range(max_retries + 1):
        try:
            if request_delay_seconds > 0:
                time.sleep(request_delay_seconds)
            return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since_ms, max_fetch_batches=max_fetch_batches)
        except (ccxt.DDoSProtection, ccxt.RateLimitExceeded):
            if attempt >= max_retries:
                raise
            time.sleep(retry_backoff_seconds * (2 ** attempt))
    return []


def _resample_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    daily = df.copy().set_index("timestamp").sort_index()
    monthly = daily.resample("MS").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["open", "high", "low", "close"]).reset_index()
    return monthly


def fetch_candles_df(
    exchange: BaseExchange,
    symbol: str,
    timeframe: str,
    limit: int = 200,
    since: str | None = None,
    max_fetch_batches: int = 1,
    until: str | None = None,
    refresh_market_data: bool = False,
    request_delay_seconds: float = 0.0,
    max_retries: int = 0,
    retry_backoff_seconds: float = 1.0,
) -> tuple[pd.DataFrame, str]:
    ex_name = exchange.__class__.__name__.replace("Exchange", "").lower()
    effective_timeframe = timeframe
    resample_to_monthly = False
    if ex_name == "kraken" and timeframe == "1M":
        LOGGER.info("Kraken does not support 1M directly; resampling 1d to 1M")
        effective_timeframe = "1d"
        resample_to_monthly = True

    cache_file = _cache_path(ex_name, symbol, timeframe)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    cached_df = pd.DataFrame()
    if cache_file.exists():
        try:
            cached_df = pd.read_parquet(cache_file)
            if not cached_df.empty:
                cached_df["timestamp"] = pd.to_datetime(cached_df["timestamp"], utc=True)
        except Exception:
            cached_df = pd.DataFrame()

    since_ts = pd.Timestamp(since) if since else None
    since_ms = int(since_ts.timestamp() * 1000) if since_ts is not None else None

    if not refresh_market_data and not cached_df.empty:
        merged = cached_df.copy()
        if since:
            merged = merged[merged["timestamp"] >= pd.Timestamp(since)]
        if until:
            merged = merged[merged["timestamp"] <= pd.Timestamp(until)]
        merged = merged.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        return merged, "cache"

    if not cached_df.empty and refresh_market_data:
        last_cached = cached_df["timestamp"].max()
        last_cached_ms = int(last_cached.timestamp() * 1000) + 1
        if since_ms is None or last_cached_ms > since_ms:
            since_ms = last_cached_ms

    raw = _fetch_with_retry(
        exchange, symbol, effective_timeframe, limit, since_ms, max_fetch_batches,
        request_delay_seconds=request_delay_seconds,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    fresh_df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    if not fresh_df.empty:
        fresh_df["timestamp"] = pd.to_datetime(fresh_df["timestamp"], unit="ms", utc=True)
    if resample_to_monthly:
        fresh_df = _resample_to_monthly(fresh_df)

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
    return merged, "cache+api" if not cached_df.empty else "api"
