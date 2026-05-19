from dataclasses import dataclass

import pandas as pd

from smc_navigator.strategy.predictive_core import evaluate_predictive_probabilities


@dataclass
class InvestorSignal:
    signal: str
    score: int
    reasons: list[str]
    regime: str
    reversal_probability: float = 0.0
    continuation_probability: float = 0.0
    exhaustion_probability: float = 0.0
    tags: list[str] | None = None


def evaluate_investor_signal(monthly: pd.DataFrame, weekly: pd.DataFrame, daily: pd.DataFrame) -> InvestorSignal:
    if monthly.empty or weekly.empty or daily.empty:
        return InvestorSignal("HOLD", 0, ["insufficient_data"], "compression", tags=[])

    probs = evaluate_predictive_probabilities(monthly, weekly)
    reasons: list[str] = []
    score = int(round((probs.reversal_probability * 0.55 + probs.continuation_probability * 0.45) * 100))

    if probs.reversal_probability > 0.62:
        reasons.append("htf_reversal_probability_high")
    if probs.continuation_probability > 0.58:
        reasons.append("htf_continuation_probability_supportive")
    if probs.exhaustion_probability > 0.65:
        reasons.append("trend_exhaustion_risk")

    if probs.reversal_probability > 0.64 and probs.exhaustion_probability < 0.68:
        return InvestorSignal("INVEST_LONG", score, reasons, "bullish_probabilistic", probs.reversal_probability, probs.continuation_probability, probs.exhaustion_probability, probs.tags)

    if probs.exhaustion_probability > 0.70 and probs.continuation_probability < 0.45:
        return InvestorSignal("INVEST_EXIT", max(score - 20, 0), reasons + ["probabilistic_exhaustion_exit"], "defensive", probs.reversal_probability, probs.continuation_probability, probs.exhaustion_probability, probs.tags)

    return InvestorSignal("HOLD", score, reasons, "selective", probs.reversal_probability, probs.continuation_probability, probs.exhaustion_probability, probs.tags)
