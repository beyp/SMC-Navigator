from uuid import uuid4

import pandas as pd

from smc_navigator.risk.position_sizing import calculate_position_size
from smc_navigator.simulator.journal import append_trade
from smc_navigator.simulator.trade import Trade
from smc_navigator.strategy.rules import evaluate_signal


def evaluate_trade_outcome(trade: Trade, future_candles: pd.DataFrame) -> Trade:
    for _, row in future_candles.iterrows():
        high, low = float(row["high"]), float(row["low"])
        if trade.direction == "LONG":
            if low <= trade.stop_loss:
                trade.close(trade.stop_loss, "LOSS")
                return trade
            if high >= trade.take_profit:
                trade.close(trade.take_profit, "WIN")
                return trade
        else:
            if high >= trade.stop_loss:
                trade.close(trade.stop_loss, "LOSS")
                return trade
            if low <= trade.take_profit:
                trade.close(trade.take_profit, "WIN")
                return trade

    trade.status = "EXPIRED"
    trade.exit_price = float(future_candles.iloc[-1]["close"]) if not future_candles.empty else trade.entry_price
    trade.close(trade.exit_price, "EXPIRED")
    return trade


def build_trade_from_signal(config: dict, signal, reason_suffix: str = "") -> Trade:
    size, risk_amount = calculate_position_size(
        capital=config["starting_capital"],
        risk_per_trade_pct=config["risk_per_trade_pct"],
        entry_price=signal.entry_price,
        stop_loss_price=signal.suggested_stop_loss,
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
        entry_price=signal.entry_price,
        stop_loss=signal.suggested_stop_loss,
        take_profit=signal.suggested_take_profit,
        position_size=size,
        risk_amount=risk_amount,
        confidence_score=signal.confidence_score,
        status="OPEN",
        exit_price=None,
        pnl=None,
        pnl_pct=None,
        reason=reason,
    )


def run_backtest_for_symbol(
    config: dict,
    symbol: str,
    enriched_df: pd.DataFrame,
    journal_path: str,
    warmup: int = 60,
    max_holding_candles: int = 10,
) -> list[Trade]:
    trades: list[Trade] = []
    if len(enriched_df) <= warmup:
        return trades

    open_trade_until_index = -1

    for idx in range(warmup, len(enriched_df) - 1):
        if idx <= open_trade_until_index:
            continue

        history = enriched_df.iloc[: idx + 1]
        signal = evaluate_signal(
            symbol=symbol,
            df=history,
            sl_pct=config["default_stop_loss_pct"],
            tp_pct=config["default_take_profit_pct"],
        )

        if signal.direction == "NONE":
            continue

        trade = build_trade_from_signal(config, signal, reason_suffix=f"signal_index={idx}")
        outcome_end = min(len(enriched_df), idx + 1 + max_holding_candles)
        future = enriched_df.iloc[idx + 1 : outcome_end]
        evaluate_trade_outcome(trade, future)
        trades.append(trade)
        append_trade(journal_path, trade)

        open_trade_until_index = outcome_end - 1

    return trades
