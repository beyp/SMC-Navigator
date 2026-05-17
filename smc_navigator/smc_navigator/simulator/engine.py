from uuid import uuid4

import pandas as pd

from smc_navigator.risk.position_sizing import calculate_position_size
from smc_navigator.simulator.journal import append_trade
from smc_navigator.simulator.trade import Trade
from smc_navigator.strategy.rules import evaluate_signal


def _apply_spread(price: float, direction: str, spread_pct: float, is_entry: bool) -> float:
    spread_factor = spread_pct / 100
    if direction == "LONG":
        return price * (1 + spread_factor) if is_entry else price * (1 - spread_factor)
    return price * (1 - spread_factor) if is_entry else price * (1 + spread_factor)


def _compute_fee(notional: float, fee_pct: float) -> float:
    return notional * (fee_pct / 100)


def evaluate_trade_outcome(trade: Trade, future_candles: pd.DataFrame, taker_fee_pct: float, spread_pct: float) -> Trade:
    entry_notional = trade.entry_price * trade.position_size
    entry_fee = _compute_fee(entry_notional, taker_fee_pct)

    for _, row in future_candles.iterrows():
        high, low, close = float(row["high"]), float(row["low"]), float(row["close"])
        if trade.direction == "LONG":
            if low <= trade.stop_loss:
                adjusted_exit = _apply_spread(trade.stop_loss, trade.direction, spread_pct, is_entry=False)
                exit_fee = _compute_fee(adjusted_exit * trade.position_size, taker_fee_pct)
                trade.close(adjusted_exit, "LOSS", entry_fee=entry_fee, exit_fee=exit_fee)
                return trade
            if high >= trade.take_profit:
                adjusted_exit = _apply_spread(trade.take_profit, trade.direction, spread_pct, is_entry=False)
                exit_fee = _compute_fee(adjusted_exit * trade.position_size, taker_fee_pct)
                trade.close(adjusted_exit, "WIN", entry_fee=entry_fee, exit_fee=exit_fee)
                return trade
        else:
            if high >= trade.stop_loss:
                adjusted_exit = _apply_spread(trade.stop_loss, trade.direction, spread_pct, is_entry=False)
                exit_fee = _compute_fee(adjusted_exit * trade.position_size, taker_fee_pct)
                trade.close(adjusted_exit, "LOSS", entry_fee=entry_fee, exit_fee=exit_fee)
                return trade
            if low <= trade.take_profit:
                adjusted_exit = _apply_spread(trade.take_profit, trade.direction, spread_pct, is_entry=False)
                exit_fee = _compute_fee(adjusted_exit * trade.position_size, taker_fee_pct)
                trade.close(adjusted_exit, "WIN", entry_fee=entry_fee, exit_fee=exit_fee)
                return trade

        last_close = close

    adjusted_exit = _apply_spread(last_close if not future_candles.empty else trade.entry_price, trade.direction, spread_pct, is_entry=False)
    exit_fee = _compute_fee(adjusted_exit * trade.position_size, taker_fee_pct)
    trade.close(adjusted_exit, "EXPIRED", entry_fee=entry_fee, exit_fee=exit_fee)
    return trade


def build_trade_from_signal(config: dict, signal, reason_suffix: str = "") -> Trade:
    spread_pct = float(config.get("spread_pct", 0.0))
    adjusted_entry_price = _apply_spread(signal.entry_price, signal.direction, spread_pct, is_entry=True)
    adjusted_sl = _apply_spread(signal.suggested_stop_loss, signal.direction, spread_pct, is_entry=False)
    adjusted_tp = _apply_spread(signal.suggested_take_profit, signal.direction, spread_pct, is_entry=False)

    size, risk_amount = calculate_position_size(
        capital=config["starting_capital"],
        risk_per_trade_pct=config["risk_per_trade_pct"],
        entry_price=adjusted_entry_price,
        stop_loss_price=adjusted_sl,
    )

    reason = "; ".join(signal.reason)
    if reason_suffix:
        reason = f"{reason}; {reason_suffix}"

    return Trade(
        trade_id=str(uuid4()),
        timestamp=signal.timestamp,
        exchange=config["exchange"],
        symbol=signal.symbol,
        timeframe=config["timeframe"],
        direction=signal.direction,
        entry_price=adjusted_entry_price,
        stop_loss=adjusted_sl,
        take_profit=adjusted_tp,
        position_size=size,
        risk_amount=risk_amount,
        confidence_score=signal.confidence_score,
        status="OPEN",
        exit_price=None,
        pnl=None,
        pnl_pct=None,
        entry_fee=0.0,
        exit_fee=0.0,
        total_fees=0.0,
        reason=reason,
    )


def run_backtest_for_symbol(config: dict, symbol: str, enriched_df: pd.DataFrame, journal_path: str, warmup: int = 60, max_holding_candles: int = 10) -> list[Trade]:
    trades: list[Trade] = []
    if len(enriched_df) <= warmup:
        return trades

    open_trade_until_index = -1
    taker_fee_pct = float(config.get("taker_fee_pct", 0.0))
    spread_pct = float(config.get("spread_pct", 0.0))

    for idx in range(warmup, len(enriched_df) - 1):
        if idx <= open_trade_until_index:
            continue

        history = enriched_df.iloc[: idx + 1]
        signal = evaluate_signal(symbol=symbol, df=history, sl_pct=config["default_stop_loss_pct"], tp_pct=config["default_take_profit_pct"])
        if signal.direction == "NONE":
            continue

        trade = build_trade_from_signal(config, signal, reason_suffix=f"signal_index={idx}")
        outcome_end = min(len(enriched_df), idx + 1 + max_holding_candles)
        future = enriched_df.iloc[idx + 1 : outcome_end]
        evaluate_trade_outcome(trade, future, taker_fee_pct=taker_fee_pct, spread_pct=spread_pct)
        trades.append(trade)
        append_trade(journal_path, trade)
        open_trade_until_index = outcome_end - 1

    return trades
