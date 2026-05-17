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
        if h >= df.iloc[i - lookback : i + lookback + 1]["high"].max(): highs.append(i)
        if l <= df.iloc[i - lookback : i + lookback + 1]["low"].min(): lows.append(i)
    return highs, lows


def _trend_state(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "NEUTRAL"
    highs, lows = _swing_points(df)
    if len(highs) < 2 or len(lows) < 2: return "NEUTRAL"
    h1, h2 = df.iloc[highs[-2]]["high"], df.iloc[highs[-1]]["high"]
    l1, l2 = df.iloc[lows[-2]]["low"], df.iloc[lows[-1]]["low"]
    if h2 > h1 and l2 > l1: return "BULLISH"
    if h2 < h1 and l2 < l1: return "BEARISH"
    return "NEUTRAL"


def evaluate_signal(symbol: str, df: pd.DataFrame, sl_pct: float, tp_pct: float, h1_close: float | None = None, h1_ema50: float | None = None, h1_df: pd.DataFrame | None = None, h4_df: pd.DataFrame | None = None) -> Signal:
    row = df.iloc[-1]; price=float(row["close"])
    reasons, tags, missing = [], [], []
    highs, lows = _swing_points(df)
    recent_high = float(df.iloc[highs[-1]]["high"]) if highs else float(df["high"].tail(20).max())
    recent_low = float(df.iloc[lows[-1]]["low"]) if lows else float(df["low"].tail(20).min())
    prev_close = float(df.iloc[-2]["close"]) if len(df) > 1 else price

    h1_trend = _trend_state(h1_df) if h1_df is not None else ("BULLISH" if (h1_close and h1_ema50 and h1_close > h1_ema50) else "BEARISH" if (h1_close and h1_ema50 and h1_close < h1_ema50) else "NEUTRAL")
    h4_trend = _trend_state(h4_df)

    bos_up = price > recent_high and prev_close <= recent_high
    bos_down = price < recent_low and prev_close >= recent_low
    choch_up = bos_up and _trend_state(df.iloc[:-1]) == "BEARISH"
    choch_down = bos_down and _trend_state(df.iloc[:-1]) == "BULLISH"
    sweep_high = float(row["high"]) > recent_high and price < recent_high
    sweep_low = float(row["low"]) < recent_low and price > recent_low
    reclaim_long = sweep_low and price > recent_low
    reclaim_short = sweep_high and price < recent_high

    local_high, local_low = float(df["high"].tail(40).max()), float(df["low"].tail(40).min())
    midpoint = (local_high + local_low) / 2
    in_discount, in_premium = price < midpoint, price > midpoint
    extended_move = abs(price - float(row.get("ema_50", price))) / max(price, 1e-9) > 0.02

    score = 0
    if h4_trend == "BULLISH": score += 20
    else: missing.append("H4 bullish")
    if h1_trend == "BULLISH": score += 20
    else: missing.append("H1 bullish")
    if in_discount: score += 15
    else: missing.append("in discount")
    if sweep_low or sweep_high: score += 15
    else: missing.append("liquidity sweep")
    if choch_up or choch_down: score += 15
    else: missing.append("CHoCH")
    if bos_up or bos_down: score += 10
    else: missing.append("BOS")
    near_kijun = abs(price - midpoint) / max(price, 1e-9) <= 0.01
    if near_kijun: score += 5
    else: missing.append("near Kijun")

    grade = "A+" if score >= 80 else "B+" if score >= 60 else "WATCH" if score >= 40 else "REJECT"

    direction = "NONE"
    if h1_trend == "BULLISH" and h4_trend == "BULLISH" and (bos_up or choch_up or reclaim_long) and in_discount and not extended_move:
        direction = "LONG"; reasons.append("HTF bullish + structure confirmation"); tags += ["discount_long", "trend_continuation"]
        if bos_up: tags.append("bos_long")
        if choch_up: tags.append("choch_long")
        if reclaim_long: tags.append("sweep_reversal")
    elif h1_trend == "BEARISH" and h4_trend == "BEARISH" and (bos_down or choch_down or reclaim_short) and in_premium and not extended_move:
        direction = "SHORT"; reasons.append("HTF bearish + structure confirmation"); tags += ["premium_short", "trend_continuation"]
        if reclaim_short: tags.append("sweep_reversal")
    else:
        reasons.append("Structure conditions not met")

    sl, tp = calculate_sl_tp(price, direction, sl_pct, tp_pct)
    return Signal(symbol=symbol, timestamp=row["timestamp"].to_pydatetime(), direction=direction, confidence_score=score, setup_score=score, setup_grade=grade, missing_conditions=missing, reason=reasons, tags=tags, entry_price=price, suggested_stop_loss=sl, suggested_take_profit=tp)
