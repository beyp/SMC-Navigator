from uuid import uuid4

import pandas as pd

from smc_navigator.risk.position_sizing import calculate_position_size
from smc_navigator.simulator.journal import append_trade
from smc_navigator.simulator.trade import Trade
from smc_navigator.strategy.rules import evaluate_signal


def _apply_spread(price: float, direction: str, spread_pct: float, is_entry: bool) -> float:
    f = spread_pct / 100
    return price * (1 + f) if (direction == "LONG") == is_entry else price * (1 - f)


def _compute_fee(notional: float, fee_pct: float) -> float:
    return notional * (fee_pct / 100)


def _rr_ratio(entry: float, stop: float, take_profit: float) -> float:
    risk = abs(entry - stop)
    return abs(take_profit - entry) / risk if risk > 0 else 0.0


def _passes_filters(config: dict, signal, row: pd.Series, symbol_trade_count: int, cooldown_until_idx: int, idx: int) -> tuple[bool, list[str]]:
    reject_tags: list[str] = []
    if idx <= cooldown_until_idx: reject_tags.append("cooldown")
    if symbol_trade_count >= int(config.get("max_trades_per_symbol", 999999)): reject_tags.append("symbol_trade_cap")
    if signal.confidence_score < int(config.get("min_confidence_score", 0)): reject_tags.append("low_confidence")
    if signal.setup_score < int(config.get("minimum_setup_score", 0)): reject_tags.append("low_setup_score")
    if signal.direction == "LONG" and not bool(config.get("enable_long_trades", True)): reject_tags.append("long_disabled")
    if signal.direction == "SHORT" and not bool(config.get("enable_short_trades", True)): reject_tags.append("short_disabled")

    price = float(signal.entry_price)
    if pd.notna(row.get("vwap")) and abs(price - float(row["vwap"])) / max(price, 1e-9) > float(config.get("max_distance_from_vwap_pct", 100.0)) / 100:
        reject_tags.append("far_from_vwap")
    if pd.notna(row.get("ema_26")) and abs(price - float(row["ema_26"])) / max(price, 1e-9) > float(config.get("max_distance_from_ema26_pct", 100.0)) / 100:
        reject_tags.append("far_from_ema26")
    if pd.notna(row.get("ema_distance_pct")) and float(row["ema_distance_pct"]) < float(config.get("min_ema_distance_pct", 0.0)):
        reject_tags.append("weak_trend")
    if pd.notna(row.get("atr_pct")) and float(row["atr_pct"]) < float(config.get("min_atr_pct", 0.0)):
        reject_tags.append("low_volatility")
    if pd.notna(row.get("range_width_pct")) and float(row["range_width_pct"]) < float(config.get("min_range_width_pct", 0.0)):
        reject_tags.append("ranging_market")

    rr = _rr_ratio(signal.entry_price, signal.suggested_stop_loss, signal.suggested_take_profit)
    if rr < float(config.get("min_rr_ratio", 0.0)): reject_tags.append("below_min_rr")
    return len(reject_tags) == 0, reject_tags


def evaluate_trade_outcome(trade: Trade, future_candles: pd.DataFrame, taker_fee_pct: float, spread_pct: float) -> Trade:
    entry_fee = _compute_fee(trade.entry_price * trade.position_size, taker_fee_pct)
    last_close = trade.entry_price
    for i, (_, row) in enumerate(future_candles.iterrows(), start=1):
        high, low, close = float(row["high"]), float(row["low"]), float(row["close"])
        if trade.direction == "LONG":
            if low <= trade.stop_loss:
                ex = _apply_spread(trade.stop_loss, trade.direction, spread_pct, False); ef = _compute_fee(ex * trade.position_size, taker_fee_pct)
                trade.close(ex, "LOSS", i, entry_fee, ef); return trade
            if high >= trade.take_profit:
                ex = _apply_spread(trade.take_profit, trade.direction, spread_pct, False); ef = _compute_fee(ex * trade.position_size, taker_fee_pct)
                trade.close(ex, "WIN", i, entry_fee, ef); return trade
        else:
            if high >= trade.stop_loss:
                ex = _apply_spread(trade.stop_loss, trade.direction, spread_pct, False); ef = _compute_fee(ex * trade.position_size, taker_fee_pct)
                trade.close(ex, "LOSS", i, entry_fee, ef); return trade
            if low <= trade.take_profit:
                ex = _apply_spread(trade.take_profit, trade.direction, spread_pct, False); ef = _compute_fee(ex * trade.position_size, taker_fee_pct)
                trade.close(ex, "WIN", i, entry_fee, ef); return trade
        last_close = close
    ex = _apply_spread(last_close, trade.direction, spread_pct, False); ef = _compute_fee(ex * trade.position_size, taker_fee_pct)
    trade.close(ex, "EXPIRED", len(future_candles), entry_fee, ef); return trade


def build_trade_from_signal(config: dict, signal, reason_suffix: str = "") -> Trade:
    sp = float(config.get("spread_pct", 0.0))
    entry = _apply_spread(signal.entry_price, signal.direction, sp, True)
    sl = _apply_spread(signal.suggested_stop_loss, signal.direction, sp, False)
    tp = _apply_spread(signal.suggested_take_profit, signal.direction, sp, False)
    size, risk_amount = calculate_position_size(config.get("starting_capital", 100.0), config.get("risk_per_trade_pct", 1.0), entry, sl)
    rr = _rr_ratio(entry, sl, tp)
    reason = "; ".join(signal.reason) + (f"; {reason_suffix}" if reason_suffix else "")
    return Trade(str(uuid4()), signal.timestamp, config["exchange"], signal.symbol, config["timeframe"], signal.direction, entry, sl, tp, size, risk_amount, signal.confidence_score, "OPEN", None, None, None, 0.0, 0.0, 0.0, 0.0, 0, rr, reason, "|".join(signal.tags))


def run_backtest_for_symbol(config: dict, symbol: str, enriched_df: pd.DataFrame, journal_path: str, h1_df: pd.DataFrame | None = None, h4_df: pd.DataFrame | None = None, warmup: int = 60, max_holding_candles: int = 10, rejected_setups: list[dict] | None = None, watch_setups: list[dict] | None = None) -> list[Trade]:
    trades: list[Trade] = []
    if len(enriched_df) <= warmup: return trades
    taker_fee_pct, spread_pct = float(config.get("taker_fee_pct", 0.0)), float(config.get("spread_pct", 0.0))
    cooldown = int(config.get("cooldown_candles_after_trade", 0)); cooldown_until_idx=-1; symbol_trade_count=0

    prep_t0 = pd.Timestamp.utcnow()
    max_iters = int(config.get("max_backtest_iterations_per_symbol", 500))
    fast_backtest = bool(config.get("fast_backtest", False))
    start_idx = max(warmup, len(enriched_df) - 1 - max_iters)
    enriched_df = enriched_df.reset_index(drop=True)
    enriched_df["_bos_up"] = (enriched_df["close"] > enriched_df["high"].rolling(20, min_periods=1).max().shift(1)) & (enriched_df["close"].shift(1) <= enriched_df["high"].rolling(20, min_periods=1).max().shift(1))
    enriched_df["_bos_down"] = (enriched_df["close"] < enriched_df["low"].rolling(20, min_periods=1).min().shift(1)) & (enriched_df["close"].shift(1) >= enriched_df["low"].rolling(20, min_periods=1).min().shift(1))
    print(f"Backtest {symbol} preprocessing_seconds={(pd.Timestamp.utcnow()-prep_t0).total_seconds():.2f}")
    iter_t0 = pd.Timestamp.utcnow()
    for idx in range(start_idx, len(enriched_df) - 1):
        if idx % 100 == 0:
            print(f"Backtest {symbol} idx={idx}/{len(enriched_df)-1} trades={len(trades)}")
        history = enriched_df.iloc[max(0, idx - 120): idx + 1] if fast_backtest else enriched_df.iloc[: idx + 1]
        h1_close = h1_ema50 = None
        h1_hist = h4_hist = None
        if h1_df is not None and not h1_df.empty:
            cutoff = history.iloc[-1]["timestamp"]
            h1_cut = h1_df["timestamp"].searchsorted(cutoff, side="right")
            h1_hist = h1_df.iloc[:h1_cut]
            if not h1_hist.empty:
                h1_row = h1_hist.iloc[-1]
                h1_close, h1_ema50 = float(h1_row["close"]), float(h1_row.get("ema_50", h1_row["close"]))
        if h4_df is not None and not h4_df.empty:
            cutoff = history.iloc[-1]["timestamp"]
            h4_cut = h4_df["timestamp"].searchsorted(cutoff, side="right")
            h4_hist = h4_df.iloc[:h4_cut]

        signal = evaluate_signal(symbol, history, config.get("default_stop_loss_pct", 1.0), config.get("default_take_profit_pct", 2.0), h1_close=h1_close, h1_ema50=h1_ema50, h1_df=h1_hist, h4_df=h4_hist)
        if 40 <= signal.setup_score < int(config.get("minimum_setup_score", 0)) and watch_setups is not None:
            watch_setups.append({"symbol": symbol, "timestamp": str(signal.timestamp), "direction": signal.direction, "setup_score": signal.setup_score, "grade": signal.setup_grade, "missing_conditions": signal.missing_conditions})
        if signal.direction == "NONE":
            if rejected_setups is not None:
                rejected_setups.append({"symbol": symbol, "timestamp": str(signal.timestamp), "failed_conditions": signal.missing_conditions + ["no_direction_trigger"], "setup_score": signal.setup_score})
            continue
        passed, reject_tags = _passes_filters(config, signal, history.iloc[-1], symbol_trade_count, cooldown_until_idx, idx)
        if not passed:
            if rejected_setups is not None:
                rejected_setups.append({"symbol": symbol, "timestamp": str(signal.timestamp), "failed_conditions": reject_tags + signal.missing_conditions, "setup_score": signal.setup_score})
            continue

        trade = build_trade_from_signal(config, signal, reason_suffix=f"signal_index={idx}")
        outcome_end = min(len(enriched_df), idx + 1 + max_holding_candles)
        future = enriched_df.iloc[idx + 1 : outcome_end]
        evaluate_trade_outcome(trade, future, taker_fee_pct, spread_pct)
        append_trade(journal_path, trade)
        trades.append(trade)
        symbol_trade_count += 1
        cooldown_until_idx = outcome_end - 1 + cooldown
    print(f"Backtest {symbol} iteration_seconds={(pd.Timestamp.utcnow()-iter_t0).total_seconds():.2f}")
    return trades
