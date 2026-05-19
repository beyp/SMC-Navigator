from dataclasses import dataclass

import pandas as pd


@dataclass
class SwingSignal:
    signal: str  # SWING_LONG | SWING_EXIT | HOLD
    score: int
    reasons: list[str]
    tags: list[str]
    bos_high: float | None = None
    bos_low: float | None = None
    pullback_30: float | None = None
    pullback_50: float | None = None
    pullback_618: float | None = None


def _is_compression(df: pd.DataFrame, lookback: int = 20, threshold: float = 0.04) -> bool:
    if len(df) < lookback:
        return False
    rng = float(df["high"].tail(lookback).max() - df["low"].tail(lookback).min())
    px = float(df["close"].iloc[-1])
    return (rng / max(px, 1e-9)) < threshold


def _h4_context(h4: pd.DataFrame) -> tuple[bool, bool, bool, bool]:
    local_high = float(h4["high"].tail(60).max())
    local_low = float(h4["low"].tail(60).min())
    midpoint = (local_high + local_low) / 2
    price = float(h4.iloc[-1]["close"])
    discount = price < midpoint
    premium = price > midpoint
    compression = _is_compression(h4, lookback=24, threshold=0.06)
    accumulation = compression and discount
    return discount, premium, compression, accumulation


def _h1_structure(h1: pd.DataFrame) -> dict:
    out = {"bos_bull": False, "bos_bear": False, "choch": False, "impulse": False, "bos_high": None, "bos_low": None}
    if len(h1) < 30:
        return out
    recent_high = float(h1["high"].tail(20).max())
    recent_low = float(h1["low"].tail(20).min())
    prev_close = float(h1.iloc[-2]["close"])
    close = float(h1.iloc[-1]["close"])
    rng = float(h1.iloc[-1]["high"] - h1.iloc[-1]["low"])
    avg_rng = float((h1["high"] - h1["low"]).tail(20).mean())

    bos_bull = close > recent_high and prev_close <= recent_high
    bos_bear = close < recent_low and prev_close >= recent_low
    choch = bos_bull and float(h1.iloc[-10]["close"]) < float(h1.iloc[-20]["close"])
    impulse = rng > avg_rng * 1.5

    out.update({"bos_bull": bos_bull, "bos_bear": bos_bear, "choch": choch, "impulse": impulse, "bos_high": recent_high, "bos_low": recent_low})
    return out


def _pullback_levels(low: float, high: float) -> tuple[float, float, float]:
    span = high - low
    return high - span * 0.30, high - span * 0.50, high - span * 0.618


def evaluate_swing_signal(weekly: pd.DataFrame, daily: pd.DataFrame, h4: pd.DataFrame, h1: pd.DataFrame | None = None, m15: pd.DataFrame | None = None, m5: pd.DataFrame | None = None) -> SwingSignal:
    if weekly.empty or daily.empty or h4.empty:
        return SwingSignal(signal="HOLD", score=0, reasons=["insufficient_data"], tags=[])

    h1 = h1 if h1 is not None and not h1.empty else daily
    m15 = m15 if m15 is not None and not m15.empty else h1
    m5 = m5 if m5 is not None and not m5.empty else m15

    score = 0
    reasons: list[str] = []
    tags: list[str] = []

    discount, premium, compression, accumulation = _h4_context(h4)
    structure = _h1_structure(h1)

    if discount:
        score += 25; reasons.append("h4_discount")
    if compression:
        score += 10; reasons.append("h4_compression")
        tags.append("compression_breakout")
    if accumulation:
        score += 10; reasons.append("h4_accumulation")

    if structure["bos_bull"]:
        score += 25; reasons.append("h1_bos_bullish")
        tags.append("bos_pullback")
    if structure["choch"]:
        score += 10; reasons.append("h1_choch")
    if structure["impulse"]:
        score += 10; reasons.append("h1_impulse_expansion")

    bos_low = float(structure["bos_low"]) if structure["bos_low"] is not None else float(h1["low"].tail(20).min())
    bos_high = float(structure["bos_high"]) if structure["bos_high"] is not None else float(h1["high"].tail(20).max())
    pb30, pb50, pb618 = _pullback_levels(bos_low, bos_high)

    px_m15 = float(m15.iloc[-1]["close"])
    in_pullback_zone = pb618 <= px_m15 <= pb30
    if in_pullback_zone:
        score += 15; reasons.append("pullback_zone_reached")

    # execution timing M15/M5
    m15_reclaim = float(m15.iloc[-1]["close"]) > float(m15.iloc[-2]["high"])
    m5_cont = float(m5.iloc[-1]["close"]) > float(m5.iloc[-2]["high"])
    if m15_reclaim:
        score += 10; reasons.append("m15_bullish_reclaim"); tags.append("reclaim_entry")
    if m5_cont:
        score += 10; reasons.append("m5_continuation_confirmation"); tags.append("continuation_reentry")

    # avoid breakout chasing/extension
    extension = abs(px_m15 - pb50) / max(px_m15, 1e-9) > 0.04
    if extension:
        reasons.append("avoid_extension")
    breakout_chase = px_m15 > bos_high * 1.01
    if breakout_chase:
        reasons.append("avoid_breakout_chasing")

    if discount and structure["bos_bull"] and in_pullback_zone and m15_reclaim and m5_cont and not extension and not breakout_chase:
        return SwingSignal("SWING_LONG", score, reasons, tags, bos_high=bos_high, bos_low=bos_low, pullback_30=pb30, pullback_50=pb50, pullback_618=pb618)

    if premium and structure["bos_bear"]:
        return SwingSignal("SWING_EXIT", score, reasons + ["h4_premium_with_bearish_shift"], tags + ["sweep_reversal"], bos_high=bos_high, bos_low=bos_low, pullback_30=pb30, pullback_50=pb50, pullback_618=pb618)

    return SwingSignal("HOLD", score, reasons, tags, bos_high=bos_high, bos_low=bos_low, pullback_30=pb30, pullback_50=pb50, pullback_618=pb618)
