from collections import Counter
from pathlib import Path

from smc_navigator.core.config_loader import load_config
from smc_navigator.core.logger import get_logger
from smc_navigator.exchanges.binance import BinanceExchange
from smc_navigator.exchanges.kraken import KrakenExchange
from smc_navigator.market_data.candles import fetch_candles_df
from smc_navigator.market_data.indicators import add_indicators
from smc_navigator.reporting.charts import plot_equity_curve, plot_symbol_chart
from smc_navigator.reporting.stats import compute_trade_stats, save_backtest_summary
from smc_navigator.simulator.engine import run_backtest_for_symbol
from smc_navigator.strategy.rules import evaluate_signal


def _build_exchange(name: str):
    if name.lower() == "kraken": return KrakenExchange()
    if name.lower() == "binance": return BinanceExchange()
    raise ValueError(f"Unsupported exchange: {name}")


def run(config_path: str = "config.yaml") -> None:
    logger = get_logger(); config = load_config(config_path)
    if not config.get("simulation_mode", True) or config.get("allow_real_orders", False):
        raise RuntimeError("Safety check failed: only simulation_mode=true and allow_real_orders=false are supported")

    ex = _build_exchange(config["exchange"])
    reports_dir = Path("reports"); charts_dir = reports_dir / "charts"; charts_dir.mkdir(parents=True, exist_ok=True)
    journal_path = Path("data/trade_journal.csv")
    all_trades, rejected, watch = [], [], []

    for symbol in config["symbols"]:
        m15 = fetch_candles_df(ex, symbol, config["timeframe"], limit=config["historical_limit_per_symbol"], since=config.get("backtest_since"), until=config.get("backtest_until"), max_fetch_batches=config.get("max_fetch_batches", 1))
        h1 = fetch_candles_df(ex, symbol, "1h", limit=config["historical_limit_per_symbol"], since=config.get("backtest_since"), until=config.get("backtest_until"), max_fetch_batches=config.get("max_fetch_batches", 1))
        h4 = fetch_candles_df(ex, symbol, "4h", limit=config["historical_limit_per_symbol"], since=config.get("backtest_since"), until=config.get("backtest_until"), max_fetch_batches=config.get("max_fetch_batches", 1))
        if m15.empty: continue
        m15i, h1i, h4i = add_indicators(m15), add_indicators(h1), add_indicators(h4)
        h1_row = h1i.iloc[-1] if not h1i.empty else None
        signal = evaluate_signal(symbol, m15i, config["default_stop_loss_pct"], config["default_take_profit_pct"], h1_close=float(h1_row["close"]) if h1_row is not None else None, h1_ema50=float(h1_row["ema_50"]) if h1_row is not None else None, h1_df=h1i, h4_df=h4i)
        trades = run_backtest_for_symbol(config=config, symbol=symbol, enriched_df=m15i, h1_df=h1i, h4_df=h4i, journal_path=str(journal_path), rejected_setups=rejected, watch_setups=watch)
        all_trades.extend(trades)
        plot_symbol_chart(df=m15i, symbol=symbol, output_path=charts_dir / f"{symbol.replace('/', '_')}.png", trade=trades[-1] if trades else None, confidence_score=signal.confidence_score)

    stats = compute_trade_stats(all_trades)
    plot_equity_curve(all_trades, reports_dir / "equity_curve.png")
    save_backtest_summary(stats, reports_dir)

    reject_counts = Counter(reason for r in rejected for reason in r.get("failed_conditions", []))
    score_distribution = Counter((r.get("setup_score", 0) // 10) * 10 for r in rejected)
    watch_stats = {"count": len(watch), "avg_score": round(sum(w["setup_score"] for w in watch) / len(watch), 2) if watch else 0.0}

    logger.info("Rejected setups: %s", len(rejected))
    logger.info("Most common rejection reasons: %s", dict(reject_counts.most_common(10)))
    logger.info("Setup score distribution: %s", dict(sorted(score_distribution.items())))
    logger.info("WATCH setups: %s", watch_stats)
    for w in watch[:20]:
        logger.info("WATCH %s %s score=%s missing=%s", w['symbol'], w['timestamp'], w['setup_score'], w['missing_conditions'])
    logger.info("Summary: trades=%s month_perf=%s symbol_perf=%s regime_perf=%s", stats.total_trades, stats.performance_by_month, stats.pnl_by_symbol, stats.performance_by_regime)
