from pathlib import Path

from smc_navigator.core.config_loader import load_config
from smc_navigator.core.logger import get_logger
from smc_navigator.exchanges.binance import BinanceExchange
from smc_navigator.exchanges.kraken import KrakenExchange
from smc_navigator.market_data.candles import fetch_candles_df
from smc_navigator.market_data.indicators import add_indicators
from smc_navigator.reporting.charts import plot_equity_curve, plot_latest_symbol_candles
from smc_navigator.reporting.stats import compute_trade_stats
from smc_navigator.simulator.engine import run_backtest_for_symbol
from smc_navigator.strategy.rules import evaluate_signal


def _build_exchange(name: str):
    if name.lower() == "kraken":
        return KrakenExchange()
    if name.lower() == "binance":
        return BinanceExchange()
    raise ValueError(f"Unsupported exchange: {name}")


def run(config_path: str = "config.yaml") -> None:
    logger = get_logger()
    config = load_config(config_path)
    if not config.get("simulation_mode", True) or config.get("allow_real_orders", False):
        raise RuntimeError("Safety check failed: only simulation_mode=true and allow_real_orders=false are supported")

    exchange = _build_exchange(config["exchange"])
    journal_path = Path("data/trade_journal.csv")
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    all_trades = []
    latest_symbol_df = None
    latest_symbol_name = ""
    latest_signal = None

    for symbol in config["symbols"]:
        candles = fetch_candles_df(exchange, symbol, config["timeframe"])
        enriched = add_indicators(candles)

        signal = evaluate_signal(symbol, enriched, config["default_stop_loss_pct"], config["default_take_profit_pct"])
        signal_name = "NO_TRADE" if signal.direction == "NONE" else f"{signal.direction}_CANDIDATE"

        trades = run_backtest_for_symbol(
            config=config,
            symbol=symbol,
            enriched_df=enriched,
            journal_path=str(journal_path),
        )
        all_trades.extend(trades)

        latest_symbol_df = enriched
        latest_symbol_name = symbol
        latest_signal = signal

        logger.info(
            "\nSymbol: %s\nSignal: %s\nConfidence: %s\nEntry: %.4f\nSL: %.4f\nTP: %.4f\nReason: %s\nBacktest trades: %s\n",
            symbol,
            signal_name,
            signal.confidence_score,
            signal.entry_price,
            signal.suggested_stop_loss,
            signal.suggested_take_profit,
            "; ".join(signal.reason),
            len(trades),
        )

    stats = compute_trade_stats(all_trades)
    plot_equity_curve(all_trades, reports_dir / "equity_curve.png")

    if latest_symbol_df is not None and latest_signal is not None:
        plot_latest_symbol_candles(
            df=latest_symbol_df,
            symbol=latest_symbol_name,
            output_path=reports_dir / "latest_symbol_candles.png",
            entry=latest_signal.entry_price,
            stop_loss=latest_signal.suggested_stop_loss,
            take_profit=latest_signal.suggested_take_profit,
        )

    logger.info(
        "\n=== Trade Statistics ===\nTotal trades: %s\nWins: %s\nLosses: %s\nWinrate: %.2f%%\nTotal PnL: %.4f\nAverage PnL: %.4f\nMax drawdown: %.4f\nReports: %s\n",
        stats.total_trades,
        stats.wins,
        stats.losses,
        stats.winrate,
        stats.total_pnl,
        stats.average_pnl,
        stats.max_drawdown,
        reports_dir,
    )
