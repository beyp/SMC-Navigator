import pandas as pd

from smc_navigator.risk.sl_tp import calculate_sl_tp
from smc_navigator.strategy.signal import Signal



def evaluate_signal(symbol: str, df: pd.DataFrame, sl_pct: float, tp_pct: float) -> Signal:
    row = df.iloc[-1]
    price = float(row["close"])
    reasons: list[str] = []
    score = 0

    long_cond = row["close"] > row["ema_26"] and row["ema_9"] > row["ema_26"] and 45 <= row["rsi_14"] <= 70
    short_cond = row["close"] < row["ema_26"] and row["ema_9"] < row["ema_26"] and 30 <= row["rsi_14"] <= 55

    near_support = abs(price - row["support"]) / price <= 0.01 if pd.notna(row["support"]) else False
    near_resistance = abs(price - row["resistance"]) / price <= 0.01 if pd.notna(row["resistance"]) else False
    near_vwap = abs(price - row["vwap"]) / price <= 0.005 if pd.notna(row["vwap"]) else False

    direction = "NONE"
    if long_cond and (near_support or near_vwap):
        direction = "LONG"
        reasons.append("Trend and momentum support long setup")
        score = 72
    elif short_cond and (near_resistance or near_vwap):
        direction = "SHORT"
        reasons.append("Trend and momentum support short setup")
        score = 68
    else:
        reasons.append("Conditions not met")
        score = 20

    sl, tp = calculate_sl_tp(price, direction, sl_pct, tp_pct)

    return Signal(
        symbol=symbol,
        timestamp=row["timestamp"].to_pydatetime(),
        direction=direction,
        confidence_score=score,
        reason=reasons,
        entry_price=price,
        suggested_stop_loss=sl,
        suggested_take_profit=tp,
    )
