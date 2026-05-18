from pathlib import Path

import pandas as pd

from smc_navigator.core.config_loader import load_config
from smc_navigator.core.logger import get_logger
from smc_navigator.exchanges.binance import BinanceExchange
from smc_navigator.exchanges.kraken import KrakenExchange
from smc_navigator.market_data.candles import fetch_candles_df
from smc_navigator.market_data.indicators import add_indicators
from smc_navigator.reporting.charts import plot_equity_curve, plot_symbol_chart
from smc_navigator.reporting.stats import compute_trade_stats, save_backtest_summary
from smc_navigator.simulator.engine import run_backtest_for_symbol
from smc_navigator.strategy.investor_engine import evaluate_investor_signal
from smc_navigator.strategy.swing_engine import evaluate_swing_signal


def _build_exchange(name: str):
    if name.lower() == "kraken": return KrakenExchange()
    if name.lower() == "binance": return BinanceExchange()
    raise ValueError(f"Unsupported exchange: {name}")


def _long_term_metrics(trades: list, capital: float) -> dict:
    if not trades:
        return {"cagr_approx": 0.0, "max_drawdown": 0.0, "time_in_market": 0.0, "monthly_returns": {}, "avg_holding_duration": 0.0}
    pnl_series = [float(t.pnl or 0.0) for t in trades]
    total_return = sum(pnl_series) / max(capital, 1e-9)
    months = max(1, len({pd.Timestamp(t.timestamp).strftime('%Y-%m') for t in trades}))
    years = months / 12
    cagr = ((1 + total_return) ** (1 / years) - 1) if years > 0 and (1 + total_return) > 0 else -1.0
    eq = 0.0; peak = 0.0; mdd = 0.0
    for p in pnl_series:
        eq += p; peak = max(peak, eq); mdd = max(mdd, peak - eq)
    monthly = {}
    for t in trades:
        m = pd.Timestamp(t.timestamp).strftime('%Y-%m'); monthly[m] = monthly.get(m, 0.0) + float(t.pnl or 0.0)
    avg_hold = sum(t.holding_candles for t in trades) / len(trades)
    return {"cagr_approx": cagr, "max_drawdown": mdd, "time_in_market": 1.0, "monthly_returns": monthly, "avg_holding_duration": avg_hold}


def run(config_path: str = "config.yaml") -> None:
    logger = get_logger(); config = load_config(config_path)
    if not config.get("simulation_mode", True) or config.get("allow_real_orders", False):
        raise RuntimeError("Safety check failed: only simulation_mode=true and allow_real_orders=false are supported")

    ex = _build_exchange(config["exchange"])
    reports_dir = Path("reports"); charts_dir = reports_dir / "charts"; charts_dir.mkdir(parents=True, exist_ok=True)
    journal_path = Path("data/trade_journal.csv")

    investor_capital = float(config.get("investor_capital", config.get("starting_capital", 100) * 0.5))
    swing_capital = float(config.get("swing_capital", config.get("starting_capital", 100) * 0.5))

    investor_signals, swing_signals = [], []
    all_trades = []

    for symbol in config["symbols"]:
        m15 = fetch_candles_df(ex, symbol, config["timeframe"], limit=config["historical_limit_per_symbol"], since=config.get("backtest_since"), until=config.get("backtest_until"), max_fetch_batches=config.get("max_fetch_batches", 1))
        h4 = fetch_candles_df(ex, symbol, "4h", limit=config["historical_limit_per_symbol"], since=config.get("backtest_since"), until=config.get("backtest_until"), max_fetch_batches=config.get("max_fetch_batches", 1))
        d1 = fetch_candles_df(ex, symbol, "1d", limit=config["historical_limit_per_symbol"], since=config.get("backtest_since"), until=config.get("backtest_until"), max_fetch_batches=config.get("max_fetch_batches", 1))
        w1 = fetch_candles_df(ex, symbol, "1w", limit=config["historical_limit_per_symbol"], since=config.get("backtest_since"), until=config.get("backtest_until"), max_fetch_batches=config.get("max_fetch_batches", 1))
        m1 = fetch_candles_df(ex, symbol, "1M", limit=config["historical_limit_per_symbol"], since=config.get("backtest_since"), until=config.get("backtest_until"), max_fetch_batches=config.get("max_fetch_batches", 1))
        if m15.empty:
            continue

        inv_sig = evaluate_investor_signal(m1, w1, d1)
        sw_sig = evaluate_swing_signal(w1, d1, h4)
        investor_signals.append((symbol, inv_sig.signal, inv_sig.score, inv_sig.reasons))
        swing_signals.append((symbol, sw_sig.signal, sw_sig.score, sw_sig.reasons))

        m15i = add_indicators(m15); h4i = add_indicators(h4)
        trades = run_backtest_for_symbol(config={**config, "starting_capital": swing_capital}, symbol=symbol, enriched_df=m15i, h1_df=add_indicators(fetch_candles_df(ex, symbol, '1h', limit=config['historical_limit_per_symbol'], since=config.get('backtest_since'), until=config.get('backtest_until'), max_fetch_batches=config.get('max_fetch_batches',1))), h4_df=h4i, journal_path=str(journal_path))
        all_trades.extend(trades)
        plot_symbol_chart(df=m15i, symbol=symbol, output_path=charts_dir / f"{symbol.replace('/', '_')}.png", trade=trades[-1] if trades else None, confidence_score=sw_sig.score)

    stats = compute_trade_stats(all_trades)
    plot_equity_curve(all_trades, reports_dir / "equity_curve.png")
    save_backtest_summary(stats, reports_dir)

    investor_metrics = _long_term_metrics([], investor_capital)
    swing_metrics = _long_term_metrics(all_trades, swing_capital)

    logger.info("Investor Engine signals: %s", investor_signals)
    logger.info("Swing Engine signals: %s", swing_signals)
    logger.info("Investor metrics: %s", investor_metrics)
    logger.info("Swing metrics: %s", swing_metrics)
