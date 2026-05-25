import pandas as pd

from smc_navigator.risk.position_sizing import calculate_position_size


def _recent_swing_levels(df: pd.DataFrame, lookback: int = 20) -> tuple[float | None, float | None]:
    if df is None or df.empty:
        return None, None
    return float(df["low"].tail(lookback).min()), float(df["high"].tail(lookback).max())


def compute_risk_plan(direction: str, entry: float, context_df: pd.DataFrame, capital: float, risk_cfg: dict) -> dict:
    use_structural = bool(risk_cfg.get("use_structural_sl", True))
    use_atr = bool(risk_cfg.get("use_atr_fallback", True))
    atr_mult = float(risk_cfg.get("atr_multiplier", 1.5))
    default_sl_pct = float(risk_cfg.get("default_stop_loss_pct", 3))
    rr_targets = risk_cfg.get("tp_rr_targets", [1.5, 2.0, 3.0])
    risk_pct = min(float(risk_cfg.get("risk_per_trade_pct", 1.0)), float(risk_cfg.get("max_risk_per_trade_pct", 1.5)))

    swing_low, swing_high = _recent_swing_levels(context_df)
    atr = float(context_df["atr_14"].iloc[-1]) if (context_df is not None and not context_df.empty and "atr_14" in context_df.columns and pd.notna(context_df["atr_14"].iloc[-1])) else None

    stop_method = "default_pct"
    if direction == "LONG":
        if use_structural and swing_low is not None and swing_low < entry:
            sl = swing_low
            stop_method = "structural_swing_low"
        elif use_atr and atr is not None and atr > 0:
            sl = entry - atr * atr_mult
            stop_method = "atr_fallback"
        else:
            sl = entry * (1 - default_sl_pct / 100)
    else:
        if use_structural and swing_high is not None and swing_high > entry:
            sl = swing_high
            stop_method = "structural_swing_high"
        elif use_atr and atr is not None and atr > 0:
            sl = entry + atr * atr_mult
            stop_method = "atr_fallback"
        else:
            sl = entry * (1 + default_sl_pct / 100)

    initial_risk = abs(entry - sl)
    tp1 = entry + initial_risk * float(rr_targets[0]) if direction == "LONG" else entry - initial_risk * float(rr_targets[0])
    tp2 = entry + initial_risk * float(rr_targets[1]) if direction == "LONG" else entry - initial_risk * float(rr_targets[1])
    tp3 = entry + initial_risk * float(rr_targets[2]) if direction == "LONG" else entry - initial_risk * float(rr_targets[2])

    liq_target = swing_high if direction == "LONG" else swing_low
    target_method = "rr_targets"
    take_profit = tp1
    if liq_target is not None:
        if (direction == "LONG" and liq_target > entry) or (direction == "SHORT" and liq_target < entry):
            take_profit = liq_target
            target_method = "liquidity_target"

    position_size, risk_amount = calculate_position_size(capital, risk_pct, entry, sl)
    return {
        "stop_loss": sl,
        "take_profit": take_profit,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "initial_risk": initial_risk,
        "rr_to_tp1": float(rr_targets[0]),
        "rr_to_tp2": float(rr_targets[1]),
        "rr_to_tp3": float(rr_targets[2]),
        "risk_amount": risk_amount,
        "position_size": position_size,
        "stop_method": stop_method,
        "target_method": target_method,
    }
