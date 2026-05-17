import pandas as pd

from smc_navigator.risk.sl_tp import calculate_sl_tp
from smc_navigator.strategy.signal import Signal


def _swing_points(df: pd.DataFrame, lookback: int = 2) -> tuple[list[int], list[int]]:
    highs, lows = [], []
    if len(df) < (lookback * 2 + 1):
        return highs, lows
    for i in range(lookback, len(df) - lookback):
        h = df.iloc[i]["high"]
        l = df.iloc[i]["low"]
        if h >= df.iloc[i - lookback : i + lookback + 1]["high"].max():
            highs.append(i)
        if l <= df.iloc[i - lookback : i + lookback + 1]["low"].min():
            lows.append(i)
    return highs, lows


def _trend_state(df: pd.DataFrame) -> str:
    highs, lows = _swing_points(df)
    if len(highs) < 2 or len(lows) < 2:
        return "NEUTRAL"
    h1, h2 = df.iloc[highs[-2]]["high"], df.iloc[highs[-1]]["high"]
    l1, l2 = df.iloc[lows[-2]]["low"], df.iloc[lows[-1]]["low"]
    if h2 > h1 and l2 > l1:
        return "BULLISH"  # HH/HL
    if h2 < h1 and l2 < l1:
        return "BEARISH"  # LH/LL
    return "NEUTRAL"


def evaluate_signal(
    symbol: str,
    df: pd.DataFrame,
    sl_pct: float,
    tp_pct: float,
    h1_close: float | None = None,
    h1_ema50: float | None = None,
    h1_df: pd.DataFrame | None = None,
    h4_df: pd.DataFrame | None = None,
) -> Signal:
    row = df.iloc[-1]
    price = float(row["close"])
    reasons: list[str] = []
    tags: list[str] = []

    # HTF bias (H1/H4 market structure trend)
    h1_trend = _trend_state(h1_df) if h1_df is not None and not h1_df.empty else ("BULLISH" if (h1_close and h1_ema50 and h1_close > h1_ema50) else "BEARISH" if (h1_close and h1_ema50 and h1_close < h1_ema50) else "NEUTRAL")
    h4_trend = _trend_state(h4_df) if h4_df is not None and not h4_df.empty else "NEUTRAL"

    highs, lows = _swing_points(df)
    recent_high = float(df.iloc[highs[-1]]["high"]) if highs else float(df["high"].tail(20).max())
    recent_low = float(df.iloc[lows[-1]]["low"]) if lows else float(df["low"].tail(20).min())
    prev_close = float(df.iloc[-2]["close"]) if len(df) > 1 else price

    # BOS / CHoCH
    bos_up = price > recent_high and prev_close <= recent_high
    bos_down = price < recent_low and prev_close >= recent_low
    choch_up = bos_up and _trend_state(df.iloc[:-1]) == "BEARISH"
    choch_down = bos_down and _trend_state(df.iloc[:-1]) == "BULLISH"

    # Equal highs/lows + sweeps
    eq_high = abs(float(row["high"]) - recent_high) / max(price, 1e-9) <= 0.001
    eq_low = abs(float(row["low"]) - recent_low) / max(price, 1e-9) <= 0.001
    sweep_high = float(row["high"]) > recent_high and price < recent_high
    sweep_low = float(row["low"]) < recent_low and price > recent_low

    # Premium / discount from local range midpoint
    local_high = float(df["high"].tail(40).max())
    local_low = float(df["low"].tail(40).min())
    midpoint = (local_high + local_low) / 2
    in_discount = price < midpoint
    in_premium = price > midpoint

    # Entry timing: reclaim after sweep + avoid extended move from EMA50
    reclaim_after_sweep_long = sweep_low and price > recent_low
    reclaim_after_sweep_short = sweep_high and price < recent_high
    extended_move = abs(price - float(row.get("ema_50", price))) / max(price, 1e-9) > 0.02

    direction = "NONE"
    score = 20

    if h1_trend == "BULLISH" and h4_trend == "BULLISH":
        if (bos_up or choch_up or reclaim_after_sweep_long) and in_discount and not extended_move:
            direction = "LONG"
            score = 74
            reasons.append("HTF bullish + structure confirmation")
            tags.extend(["discount_long", "trend_continuation"])
            if bos_up:
                tags.append("bos_long")
            if choch_up:
                tags.append("choch_long")
            if reclaim_after_sweep_long:
                tags.append("sweep_reversal")

    if direction == "NONE" and h1_trend == "BEARISH" and h4_trend == "BEARISH":
        if (bos_down or choch_down or reclaim_after_sweep_short) and in_premium and not extended_move:
            direction = "SHORT"
            score = 72
            reasons.append("HTF bearish + structure confirmation")
            tags.extend(["premium_short", "trend_continuation"])
            if reclaim_after_sweep_short:
                tags.append("sweep_reversal")

    if direction == "NONE":
        reasons.append("Structure conditions not met")

    if eq_high or eq_low:
        reasons.append("equal_liquidity_zone")
    if sweep_high or sweep_low:
        reasons.append("liquidity_sweep_detected")

    sl, tp = calculate_sl_tp(price, direction, sl_pct, tp_pct)
    return Signal(
        symbol=symbol,
        timestamp=row["timestamp"].to_pydatetime(),
        direction=direction,
        confidence_score=score,
        reason=reasons,
        tags=tags,
        entry_price=price,
        suggested_stop_loss=sl,
        suggested_take_profit=tp,
    )
