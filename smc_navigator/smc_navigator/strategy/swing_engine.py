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
    h4_context: str = "NEUTRAL"
    h1_structure: str = "NEUTRAL"
    bos_direction: str = "NONE"
    pullback_zone: str = "NOT_REACHED"
    m15_confirmation: str = "WAITING"
    trigger_reason: str = "HOLD"
    missing_conditions: list[str] | None = None


def _h1_bos(h1: pd.DataFrame) -> tuple[bool, float, float]:
    high = float(h1["high"].tail(20).max())
    low = float(h1["low"].tail(20).min())
    close = float(h1.iloc[-1]["close"])
    prev = float(h1.iloc[-2]["close"])
    return close > high and prev <= high, high, low


def _pullback_levels(low: float, high: float) -> tuple[float, float, float]:
    span = high - low
    return high - span * 0.30, high - span * 0.50, high - span * 0.618


def evaluate_swing_signal(weekly: pd.DataFrame, daily: pd.DataFrame, h4: pd.DataFrame, h1: pd.DataFrame | None = None, m15: pd.DataFrame | None = None, m5: pd.DataFrame | None = None, features: dict | None = None) -> SwingSignal:
    if weekly.empty or daily.empty or h4.empty:
        return SwingSignal("HOLD", 0, ["insufficient_data"], [])
    h1 = h1 if h1 is not None and not h1.empty else daily
    m15 = m15 if m15 is not None and not m15.empty else h1
    m5 = m5 if m5 is not None and not m5.empty else m15

    probs = evaluate_predictive_probabilities(h4, h1, features=features)
    trigger_cfg = (features or {}).get("trigger_engine", {}) if isinstance((features or {}).get("trigger_engine"), dict) else {}
    require_h1_bos = bool(trigger_cfg.get("require_h1_bos", True))
    require_pullback = bool(trigger_cfg.get("require_pullback_zone", True))
    require_m15 = bool(trigger_cfg.get("require_m15_reclaim", True))
    max_extension_pct = float(trigger_cfg.get("max_extension_pct", 2.5)) / 100.0
    min_rr_ratio = float(trigger_cfg.get("min_rr_ratio", 1.5))
    allow_short = bool(trigger_cfg.get("allow_short", False))
    features = features or {}
    h1_bos, bos_high, bos_low = _h1_bos(h1)
    if not features.get("bos_pullback_logic", True):
        h1_bos = True
    pb30, pb50, pb618 = _pullback_levels(bos_low, bos_high)
    px = float(m15.iloc[-1]["close"])
    in_pullback = pb618 <= px <= pb30
    m15_reclaim = True if not features.get("reclaim_confirmation", True) else float(m15.iloc[-1]["close"]) > float(m15.iloc[-2]["high"])
    m5_reclaim = True if not features.get("m5_execution", False) else float(m5.iloc[-1]["close"]) > float(m5.iloc[-2]["high"])

    score = int(round((probs.reversal_probability * 0.4 + probs.continuation_probability * 0.6) * 100))
    reasons: list[str] = []
    missing: list[str] = []
    if h1_bos: reasons.append("h1_bos_confirmed")
    if in_pullback: reasons.append("pullback_retracement_zone")
    if m15_reclaim: reasons.append("m15_reclaim_confirmation")
    if m5_reclaim: reasons.append("m5_reclaim_confirmation")
    if not h1_bos: missing.append("h1_bos")
    if not in_pullback: missing.append("pullback_zone")
    if not m15_reclaim: missing.append("m15_reclaim")

    h4_context = "HTF_LONG_ZONE" if probs.reversal_probability >= 0.55 else "HTF_SHORT_ZONE" if probs.continuation_probability < 0.45 else "NEUTRAL"
    h1_structure = "BULLISH_BOS" if h1_bos else "NO_BOS"
    bos_direction = "BULLISH" if h1_bos else "NONE"
    pullback_state = "REACHED" if in_pullback else "NOT_REACHED"
    m15_state = "RECLAIM_CONFIRMED" if m15_reclaim else "WAITING_RECLAIM"
    extension = abs(px - pb50) / max(px, 1e-9)
    rr = abs((bos_high - px) / max(px - bos_low, 1e-9))
    if extension > max_extension_pct:
        missing.append("overextended")
    if rr < min_rr_ratio:
        missing.append("rr_below_min")

    long_ok = (not require_h1_bos or h1_bos) and (not require_pullback or in_pullback) and (not require_m15 or m15_reclaim) and m5_reclaim and probs.continuation_probability > 0.52 and extension <= max_extension_pct and rr >= min_rr_ratio and h4_context == "HTF_LONG_ZONE"
    if long_ok:
        return SwingSignal("SWING_LONG_TRIGGER", score, reasons, probs.tags, probs.reversal_probability, probs.continuation_probability, probs.exhaustion_probability, bos_high, bos_low, pb30, pb50, pb618, h4_context, h1_structure, bos_direction, pullback_state, m15_state, "LONG_TRIGGER", missing_conditions=missing)

    short_ok = allow_short and h4_context == "HTF_SHORT_ZONE" and (not require_pullback or in_pullback) and probs.exhaustion_probability < 0.75
    if short_ok:
        return SwingSignal("SWING_SHORT_TRIGGER", score, reasons, probs.tags, probs.reversal_probability, probs.continuation_probability, probs.exhaustion_probability, bos_high, bos_low, pb30, pb50, pb618, h4_context, "BEARISH_BOS", "BEARISH", pullback_state, m15_state, "SHORT_TRIGGER", missing_conditions=missing)
    if probs.exhaustion_probability > 0.7:
        return SwingSignal("SWING_EXIT", max(score - 15, 0), reasons + ["probabilistic_exhaustion"], probs.tags, probs.reversal_probability, probs.continuation_probability, probs.exhaustion_probability, bos_high, bos_low, pb30, pb50, pb618, h4_context, h1_structure, bos_direction, pullback_state, m15_state, "EXIT", missing_conditions=missing)

    return SwingSignal("HOLD", score, reasons, probs.tags, probs.reversal_probability, probs.continuation_probability, probs.exhaustion_probability, bos_high, bos_low, pb30, pb50, pb618, h4_context, h1_structure, bos_direction, pullback_state, m15_state, "HOLD", missing_conditions=missing)
