import pandas as pd



def add_indicators(df: pd.DataFrame, sr_lookback: int = 20) -> pd.DataFrame:
    data = df.copy()
    data["ema_9"] = data["close"].ewm(span=9, adjust=False).mean()
    data["ema_26"] = data["close"].ewm(span=26, adjust=False).mean()
    data["ema_50"] = data["close"].ewm(span=50, adjust=False).mean()

    delta = data["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    data["rsi_14"] = 100 - (100 / (1 + rs))

    typical_price = (data["high"] + data["low"] + data["close"]) / 3
    cumulative_tp_vol = (typical_price * data["volume"]).cumsum()
    cumulative_vol = data["volume"].cumsum().replace(0, pd.NA)
    data["vwap"] = cumulative_tp_vol / cumulative_vol

    data["support"] = data["low"].rolling(sr_lookback).min()
    data["resistance"] = data["high"].rolling(sr_lookback).max()

    return data
