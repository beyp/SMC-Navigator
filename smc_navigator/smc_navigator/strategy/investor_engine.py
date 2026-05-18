from dataclasses import dataclass

import pandas as pd


@dataclass
class InvestorSignal:
    signal: str  # INVEST_LONG | INVEST_EXIT | HOLD
    score: int
    reasons: list[str]


def evaluate_investor_signal(monthly: pd.DataFrame, weekly: pd.DataFrame, daily: pd.DataFrame) -> InvestorSignal:
    if monthly.empty or weekly.empty or daily.empty:
        return InvestorSignal(signal="HOLD", score=0, reasons=["insufficient_data"])

    m = monthly.copy()
    w = weekly.copy()
    d = daily.copy()

    m["ema_20"] = m["close"].ewm(span=20, adjust=False).mean()
    m["kijun"] = (m["high"].rolling(26).max() + m["low"].rolling(26).min()) / 2
    w["ema_20"] = w["close"].ewm(span=20, adjust=False).mean()

    m_last = m.iloc[-1]
    w_last = w.iloc[-1]
    d_last = d.iloc[-1]

    reasons: list[str] = []
    score = 0

    monthly_reclaim = float(m_last["close"]) > float(m_last["ema_20"])
    weekly_bullish = float(w_last["close"]) > float(w_last["ema_20"])
    monthly_kijun_reclaim = pd.notna(m_last.get("kijun")) and float(m_last["close"]) > float(m_last["kijun"])

    local_high = float(m["high"].tail(12).max())
    local_low = float(m["low"].tail(12).min())
    midpoint = (local_high + local_low) / 2
    discount = float(m_last["close"]) < midpoint

    compression = (m["high"].tail(6).max() - m["low"].tail(6).min()) / max(float(m_last["close"]), 1e-9) < 0.15
    breakout = float(d_last["close"]) > float(w["high"].tail(4).max())

    if monthly_reclaim:
        score += 20; reasons.append("monthly_trend_reclaim")
    if weekly_bullish:
        score += 20; reasons.append("weekly_bullish_structure")
    if monthly_kijun_reclaim:
        score += 20; reasons.append("monthly_kijun_reclaim")
    if discount:
        score += 15; reasons.append("long_term_discount_zone")
    if compression and breakout:
        score += 15; reasons.append("compression_breakout")
    if m["close"].tail(4).std() < m["close"].tail(12).std():
        score += 10; reasons.append("volatility_contraction")

    if score >= 60:
        return InvestorSignal(signal="INVEST_LONG", score=score, reasons=reasons)
    if score <= 20 and not monthly_reclaim:
        return InvestorSignal(signal="INVEST_EXIT", score=score, reasons=reasons + ["macro_weakness"])
    return InvestorSignal(signal="HOLD", score=score, reasons=reasons)
