from dataclasses import dataclass

import pandas as pd


@dataclass
class SwingSignal:
    signal: str  # SWING_LONG | SWING_EXIT | HOLD
    score: int
    reasons: list[str]


def evaluate_swing_signal(weekly: pd.DataFrame, daily: pd.DataFrame, h4: pd.DataFrame) -> SwingSignal:
    if weekly.empty or daily.empty or h4.empty:
        return SwingSignal(signal="HOLD", score=0, reasons=["insufficient_data"])

    w = weekly.copy(); d = daily.copy(); h = h4.copy()
    w["ema_20"] = w["close"].ewm(span=20, adjust=False).mean()
    d["ema_20"] = d["close"].ewm(span=20, adjust=False).mean()
    h["ema_20"] = h["close"].ewm(span=20, adjust=False).mean()

    score = 0
    reasons: list[str] = []

    if float(w.iloc[-1]["close"]) > float(w.iloc[-1]["ema_20"]):
        score += 30; reasons.append("weekly_context_bullish")
    if float(d.iloc[-1]["close"]) > float(d.iloc[-1]["ema_20"]):
        score += 25; reasons.append("daily_context_bullish")
    if float(h.iloc[-1]["close"]) > float(h.iloc[-1]["ema_20"]):
        score += 25; reasons.append("h4_execution_trend")

    pullback = float(h.iloc[-1]["low"]) <= float(h["low"].tail(10).quantile(0.35))
    if pullback:
        score += 10; reasons.append("h4_pullback")

    breakout = float(h.iloc[-1]["close"]) > float(h["high"].tail(8).max())
    if breakout:
        score += 10; reasons.append("h4_breakout")

    if score >= 65:
        return SwingSignal(signal="SWING_LONG", score=score, reasons=reasons)
    if score < 25:
        return SwingSignal(signal="SWING_EXIT", score=score, reasons=reasons)
    return SwingSignal(signal="HOLD", score=score, reasons=reasons)
