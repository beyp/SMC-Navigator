from uuid import uuid4

import pandas as pd

from smc_navigator.risk.position_sizing import calculate_position_size
from smc_navigator.simulator.journal import append_trade
from smc_navigator.simulator.trade import Trade
from smc_navigator.strategy.signal import Signal



def simulate_signal(config: dict, signal: Signal, df: pd.DataFrame, journal_path: str) -> Trade | None:
    if signal.direction == "NONE":
        return None

    size, risk_amount = calculate_position_size(
        capital=config["starting_capital"],
        risk_per_trade_pct=config["risk_per_trade_pct"],
        entry_price=signal.entry_price,
        stop_loss_price=signal.suggested_stop_loss,
    )

    trade = Trade(
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
        reason="; ".join(signal.reason),
    )

    future = df.iloc[-10:]
    for _, row in future.iterrows():
        high, low = float(row["high"]), float(row["low"])
        if trade.direction == "LONG":
            if low <= trade.stop_loss:
                trade.close(trade.stop_loss, "LOSS")
                break
            if high >= trade.take_profit:
                trade.close(trade.take_profit, "WIN")
                break
        else:
            if high >= trade.stop_loss:
                trade.close(trade.stop_loss, "LOSS")
                break
            if low <= trade.take_profit:
                trade.close(trade.take_profit, "WIN")
                break

    append_trade(journal_path, trade)
    return trade
