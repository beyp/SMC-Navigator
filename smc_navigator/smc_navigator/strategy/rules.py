import pandas as pd

from smc_navigator.risk.sl_tp import calculate_sl_tp
from smc_navigator.strategy.signal import Signal


def evaluate_signal(symbol: str, df: pd.DataFrame, sl_pct: float, tp_pct: float, h1_close: float | None = None, h1_ema50: float | None = None) -> Signal:
    row = df.iloc[-1]
    price = float(row["close"])
    reasons: list[str] = []
    tags: list[str] = []

    long_cond = row["close"] > row["ema_26"] and row["ema_9"] > row["ema_26"] and 45 <= row["rsi_14"] <= 70
    short_cond = row["close"] < row["ema_26"] and row["ema_9"] < row["ema_26"] and 30 <= row["rsi_14"] <= 55

    near_support = abs(price - row["support"]) / price <= 0.01 if pd.notna(row["support"]) else False
    near_resistance = abs(price - row["resistance"]) / price <= 0.01 if pd.notna(row["resistance"]) else False
    near_vwap = abs(price - row["vwap"]) / price <= 0.005 if pd.notna(row["vwap"]) else False

    htf_long_ok = True if (h1_close is None or h1_ema50 is None) else h1_close > h1_ema50
    htf_short_ok = True if (h1_close is None or h1_ema50 is None) else h1_close < h1_ema50

    direction = "NONE"
    score = 20
    if long_cond and (near_support or near_vwap) and htf_long_ok:
        direction = "LONG"
        reasons.append("Trend and momentum support long setup")
        tags.append("trend_following")
        score = 75
    elif short_cond and (near_resistance or near_vwap) and htf_short_ok:
        direction = "SHORT"
        reasons.append("Trend and momentum support short setup")
        tags.append("trend_following")
        score = 71
    else:
        reasons.append("Conditions not met")

    if score >= 70:
        tags.append("high_confidence")

    sl, tp = calculate_sl_tp(price, direction, sl_pct, tp_pct)
    return Signal(symbol=symbol, timestamp=row["timestamp"].to_pydatetime(), direction=direction, confidence_score=score, reason=reasons, tags=tags, entry_price=price, suggested_stop_loss=sl, suggested_take_profit=tp)
