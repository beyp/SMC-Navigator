from dataclasses import dataclass

import pandas as pd

from smc_navigator.strategy.predictive_core import evaluate_predictive_probabilities


@dataclass
class SwingSignal:
    signal: str
    score: int
    reasons: list[str]
    tags: list[str]
    reversal_probability: float = 0.0
    continuation_probability: float = 0.0
    exhaustion_probability: float = 0.0
    bos_high: float | None = None
    bos_low: float | None = None
    pullback_30: float | None = None
    pullback_50: float | None = None
    pullback_618: float | None = None


def _h1_bos(h1: pd.DataFrame) -> tuple[bool, float, float]:
    high = float(h1["high"].tail(20).max())
    low = float(h1["low"].tail(20).min())
    close = float(h1.iloc[-1]["close"])
    prev = float(h1.iloc[-2]["close"])
    return close > high and prev <= high, high, low


def _pullback_levels(low: float, high: float) -> tuple[float, float, float]:
    span = high - low
    return high - span * 0.30, high - span * 0.50, high - span * 0.618


def evaluate_swing_signal(weekly: pd.DataFrame, daily: pd.DataFrame, h4: pd.DataFrame, h1: pd.DataFrame | None = None, m15: pd.DataFrame | None = None, m5: pd.DataFrame | None = None) -> SwingSignal:
    if weekly.empty or daily.empty or h4.empty:
        return SwingSignal("HOLD", 0, ["insufficient_data"], [])
    h1 = h1 if h1 is not None and not h1.empty else daily
    m15 = m15 if m15 is not None and not m15.empty else h1
    m5 = m5 if m5 is not None and not m5.empty else m15

    probs = evaluate_predictive_probabilities(h4, h1)
    h1_bos, bos_high, bos_low = _h1_bos(h1)
    pb30, pb50, pb618 = _pullback_levels(bos_low, bos_high)
    px = float(m15.iloc[-1]["close"])
    in_pullback = pb618 <= px <= pb30
    m15_reclaim = float(m15.iloc[-1]["close"]) > float(m15.iloc[-2]["high"])
    m5_reclaim = float(m5.iloc[-1]["close"]) > float(m5.iloc[-2]["high"])

    score = int(round((probs.reversal_probability * 0.4 + probs.continuation_probability * 0.6) * 100))
    reasons: list[str] = []
    if h1_bos: reasons.append("h1_bos_confirmed")
    if in_pullback: reasons.append("pullback_retracement_zone")
    if m15_reclaim: reasons.append("m15_reclaim_confirmation")
    if m5_reclaim: reasons.append("m5_reclaim_confirmation")

    if h1_bos and in_pullback and m15_reclaim and m5_reclaim and probs.continuation_probability > 0.52:
        return SwingSignal("SWING_LONG", score, reasons, probs.tags, probs.reversal_probability, probs.continuation_probability, probs.exhaustion_probability, bos_high, bos_low, pb30, pb50, pb618)

    if probs.exhaustion_probability > 0.7:
        return SwingSignal("SWING_EXIT", max(score - 15, 0), reasons + ["probabilistic_exhaustion"], probs.tags, probs.reversal_probability, probs.continuation_probability, probs.exhaustion_probability, bos_high, bos_low, pb30, pb50, pb618)

    return SwingSignal("HOLD", score, reasons, probs.tags, probs.reversal_probability, probs.continuation_probability, probs.exhaustion_probability, bos_high, bos_low, pb30, pb50, pb618)
