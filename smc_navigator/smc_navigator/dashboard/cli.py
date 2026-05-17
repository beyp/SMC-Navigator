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
    charts_dir = reports_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    all_trades = []
    for symbol in config["symbols"]:
        candles = fetch_candles_df(exchange, symbol, config["timeframe"])
        enriched = add_indicators(candles)
        signal = evaluate_signal(symbol, enriched, config["default_stop_loss_pct"], config["default_take_profit_pct"])
        trades = run_backtest_for_symbol(config=config, symbol=symbol, enriched_df=enriched, journal_path=str(journal_path))
        all_trades.extend(trades)
        latest_trade = trades[-1] if trades else None
        safe_symbol = symbol.replace("/", "_")
        plot_symbol_chart(df=enriched, symbol=symbol, output_path=charts_dir / f"{safe_symbol}.png", trade=latest_trade, confidence_score=signal.confidence_score)

    stats_after = compute_trade_stats(all_trades)
    plot_equity_curve(all_trades, reports_dir / "equity_curve.png")
    save_backtest_summary(stats_after, reports_dir)

    # comparison before fees vs after fees
    gross_total = sum(t.gross_pnl for t in all_trades)
    net_total = stats_after.net_pnl_after_fees
    fees_impact = gross_total - net_total

    logger.info(
        "\n=== Backtest Summary ===\n"
        "Total trades: %s\nWins: %s\nLosses: %s\nWinrate: %.2f%%\n"
        "Gross profit: %.4f\nGross loss: %.4f\nNet PnL after fees: %.4f\n"
        "Total fees paid: %.4f\nAverage PnL/trade: %.4f\nProfit factor: %.4f\nExpectancy: %.4f\n"
        "Max drawdown: %.4f\nAverage holding candles: %.2f\n"
        "PnL by symbol: %s\nPnL by direction: %s\n"
        "\n=== Fees Impact Comparison ===\n"
        "Results before fees (gross): %.4f\nResults after fees (net): %.4f\nFees impact: %.4f\n"
        "Reports: %s\nCharts: %s\n",
        stats_after.total_trades,
        stats_after.wins,
        stats_after.losses,
        stats_after.winrate,
        stats_after.gross_profit,
        stats_after.gross_loss,
        stats_after.net_pnl_after_fees,
        stats_after.total_fees_paid,
        stats_after.average_pnl_per_trade,
        stats_after.profit_factor,
        stats_after.expectancy,
        stats_after.max_drawdown,
        stats_after.average_holding_candles,
        stats_after.pnl_by_symbol,
        stats_after.pnl_by_direction,
        gross_total,
        net_total,
        fees_impact,
        reports_dir,
        charts_dir,
    )
