from dataclasses import dataclass

import pandas as pd


PREDICTIVE_TAGS = {
    "htf_long_zone": "HTF_LONG_ZONE",
    "absorption": "ABSORPTION",
    "stop_hunt": "STOP_HUNT",
    "bos_reclaim": "BOS_RECLAIM",
    "compression_breakout": "COMPRESSION_BREAKOUT",
    "trend_exhaustion": "TREND_EXHAUSTION",
}


@dataclass
class PredictiveProbabilities:
    reversal_probability: float
    continuation_probability: float
    exhaustion_probability: float
    feature_scores: dict[str, float]
    tags: list[str]


def _norm(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _on(features: dict, name: str, default: bool = True) -> bool:
    return bool(features.get(name, default))


def evaluate_predictive_probabilities(htf: pd.DataFrame, ltf: pd.DataFrame | None = None, features: dict | None = None) -> PredictiveProbabilities:
    features = features or {}
    if htf.empty or not _on(features, "predictive_engine", True):
        return PredictiveProbabilities(0.0, 0.0, 0.0, {}, [])

    tags: list[str] = []
    feature_scores: dict[str, float] = {}
    local_high = float(htf["high"].tail(60).max()); local_low = float(htf["low"].tail(60).min())
    eq = (local_high + local_low) / 2; last = htf.iloc[-1]; close = float(last["close"]); rng = max(local_high - local_low, 1e-9)

    discount_score = _norm((eq - close) / rng + 0.5)
    proximity_to_eq = _norm(1.0 - abs(close - eq) / rng)
    compression = _norm(1.0 - ((float(htf["high"].tail(20).max()) - float(htf["low"].tail(20).min())) / max(close, 1e-9)) / 0.10) if _on(features, "compression_detection", True) else 0.0
    recent_range = (htf["high"] - htf["low"]).tail(20)
    absorption = _norm((recent_range.mean() - recent_range.tail(5).mean()) / max(recent_range.mean(), 1e-9) + 0.5)
    stop_hunt = _norm((float(last["high"]) - float(htf["high"].tail(10).max())) / max(rng, 1e-9) + 0.5) if _on(features, "sweep_detection", True) else 0.0
    closes = htf["close"]
    bos = float(closes.iloc[-1] > htf["high"].tail(20).max())
    choch = float(closes.iloc[-1] > closes.iloc[-5] and closes.iloc[-8] < closes.iloc[-12]) if len(closes) > 12 else 0.0
    reclaim = float(closes.iloc[-1] > eq and closes.iloc[-2] <= eq) if len(closes) > 2 else 0.0
    rejection_strength = _norm((float(last["close"]) - float(last["low"])) / max(float(last["high"] - last["low"]), 1e-9))
    trend_exhaustion = _norm((closes.diff().tail(12).abs().mean() - closes.diff().tail(3).abs().mean()) / max(closes.diff().tail(12).abs().mean(), 1e-9) + 0.5)
    expansion_after_compression = _norm((htf["high"].tail(3).max() - htf["low"].tail(3).min()) / max(htf["high"].tail(20).max() - htf["low"].tail(20).min(), 1e-9)) if _on(features, "compression_detection", True) else 0.0

    feature_scores.update({"htf_discount_premium": discount_score, "monthly_equilibrium_proximity": proximity_to_eq, "volatility_compression": compression, "liquidity_sweeps": stop_hunt, "stop_hunts": stop_hunt, "absorption": absorption, "bos_choch": _norm((bos + choch) / 2), "reclaim_after_sweep": reclaim, "rejection_strength": rejection_strength, "trend_exhaustion": trend_exhaustion, "expansion_after_compression": expansion_after_compression})
    reversal = _norm((discount_score + absorption + reclaim + rejection_strength + trend_exhaustion) / 5)
    continuation = _norm((feature_scores["bos_choch"] + expansion_after_compression + (1 - trend_exhaustion)) / 3)
    exhaustion = _norm((trend_exhaustion + compression + proximity_to_eq) / 3)

    if discount_score > 0.58: tags.append(PREDICTIVE_TAGS["htf_long_zone"])
    if absorption > 0.58: tags.append(PREDICTIVE_TAGS["absorption"])
    if stop_hunt > 0.55: tags.append(PREDICTIVE_TAGS["stop_hunt"])
    if reclaim > 0.5 and bos > 0.5: tags.append(PREDICTIVE_TAGS["bos_reclaim"])
    if compression > 0.6 and expansion_after_compression > 0.35: tags.append(PREDICTIVE_TAGS["compression_breakout"])
    if trend_exhaustion > 0.6: tags.append(PREDICTIVE_TAGS["trend_exhaustion"])
    return PredictiveProbabilities(reversal, continuation, exhaustion, feature_scores, tags)
