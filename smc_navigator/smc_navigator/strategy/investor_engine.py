from dataclasses import dataclass

import pandas as pd


@dataclass
class InvestorSignal:
    signal: str  # INVEST_LONG | INVEST_EXIT | HOLD
    score: int
    reasons: list[str]
    regime: str


def _structure_state(df: pd.DataFrame) -> str:
    if len(df) < 6:
        return "neutral"
    highs = df["high"].tail(6).tolist()
    lows = df["low"].tail(6).tolist()
    if highs[-1] > highs[-3] and lows[-1] > lows[-3]:
        return "hh_hl"
    if highs[-1] < highs[-3] and lows[-1] < lows[-3]:
        return "lh_ll"
    return "mixed"


def _monthly_regime(monthly: pd.DataFrame) -> str:
    m = monthly.copy()
    m["ema_20"] = m["close"].ewm(span=20, adjust=False).mean()
    last = m.iloc[-1]
    width = (m["high"].tail(6).max() - m["low"].tail(6).min()) / max(float(last["close"]), 1e-9)
    if width < 0.12:
        return "compression"
    if float(last["close"]) > float(last["ema_20"]):
        return "accumulation" if width < 0.22 else "bullish"
    return "distribution" if width < 0.22 else "bearish"


def evaluate_investor_signal(monthly: pd.DataFrame, weekly: pd.DataFrame, daily: pd.DataFrame) -> InvestorSignal:
    if monthly.empty or weekly.empty or daily.empty:
        return InvestorSignal("HOLD", 0, ["insufficient_data"], "compression")

    regime = _monthly_regime(monthly)

    w = weekly.copy(); d = daily.copy()
    w["ema_20"] = w["close"].ewm(span=20, adjust=False).mean()
    d["ema_20"] = d["close"].ewm(span=20, adjust=False).mean()

    w_last = w.iloc[-1]
    d_last = d.iloc[-1]
    reasons: list[str] = []
    score = 0

    weekly_state = _structure_state(w)
    weekly_bos_bull = float(w_last["close"]) > float(w["high"].tail(5).max())
    weekly_bos_bear = float(w_last["close"]) < float(w["low"].tail(5).min())
    weekly_ema_reclaim = float(w_last["close"]) > float(w_last["ema_20"])
    weekly_ema_loss = float(w_last["close"]) < float(w_last["ema_20"])
    weekly_momentum_exp = w["close"].tail(3).std() > w["close"].tail(10).std()

    pullback_reclaim = float(d_last["close"]) > float(d_last["ema_20"]) and float(d_last["low"]) <= float(d["low"].tail(8).quantile(0.35))
    daily_vol_exp = d["close"].tail(4).std() > d["close"].tail(12).std()
    confirmation_close = float(d_last["close"]) > float(d["high"].tail(7).mean())
    trend_continuation = _structure_state(d) == "hh_hl"

    if regime in {"accumulation", "bullish"}:
        score += 20; reasons.append("monthly_not_bearish")
    if weekly_bos_bull or weekly_state == "hh_hl":
        score += 25; reasons.append("weekly_bullish_transition")
    if weekly_ema_reclaim:
        score += 15; reasons.append("weekly_ema20_reclaim")
    if weekly_momentum_exp:
        score += 10; reasons.append("weekly_momentum_expansion")
    if pullback_reclaim:
        score += 10; reasons.append("daily_reclaim_after_pullback")
    if daily_vol_exp:
        score += 10; reasons.append("daily_volatility_expansion")
    if confirmation_close and trend_continuation:
        score += 10; reasons.append("daily_confirmation_continuation")

    # entry logic
    if regime != "bearish" and (weekly_bos_bull or weekly_state == "hh_hl") and confirmation_close:
        return InvestorSignal("INVEST_LONG", score, reasons, "bullish_expansion" if regime in {"bullish", "accumulation"} else regime)

    # exit logic
    exit_reasons = []
    if regime in {"distribution", "bearish"}: exit_reasons.append("monthly_regime_deterioration")
    if weekly_ema_loss: exit_reasons.append("weekly_ema20_loss")
    if weekly_state == "lh_ll" or weekly_bos_bear: exit_reasons.append("weekly_bearish_structure_shift")
    if exit_reasons:
        return InvestorSignal("INVEST_EXIT", max(0, score - 30), reasons + exit_reasons, "bearish_expansion" if regime == "bearish" else "distribution")

    mapped = {
        "accumulation": "accumulation",
        "bullish": "bullish_expansion",
        "distribution": "distribution",
        "bearish": "bearish_expansion",
        "compression": "compression",
    }
    return InvestorSignal("HOLD", score, reasons, mapped.get(regime, "compression"))
