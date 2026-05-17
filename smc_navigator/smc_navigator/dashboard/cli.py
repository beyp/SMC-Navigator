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
    logger = get_logger(); config = load_config(config_path)
    if not config.get("simulation_mode", True) or config.get("allow_real_orders", False):
        raise RuntimeError("Safety check failed: only simulation_mode=true and allow_real_orders=false are supported")

    exchange = _build_exchange(config["exchange"])
    journal_path = Path("data/trade_journal.csv"); reports_dir = Path("reports"); charts_dir = reports_dir / "charts"; charts_dir.mkdir(parents=True, exist_ok=True)
    all_trades = []
    for symbol in config["symbols"]:
        m15 = add_indicators(fetch_candles_df(exchange, symbol, config["timeframe"]))
        h1 = add_indicators(fetch_candles_df(exchange, symbol, "1h"))
        h1_row = h1.iloc[-1] if not h1.empty else None
        signal = evaluate_signal(symbol, m15, config["default_stop_loss_pct"], config["default_take_profit_pct"], h1_close=float(h1_row["close"]) if h1_row is not None else None, h1_ema50=float(h1_row["ema_50"]) if h1_row is not None else None)
        trades = run_backtest_for_symbol(config=config, symbol=symbol, enriched_df=m15, h1_df=h1, journal_path=str(journal_path))
        all_trades.extend(trades)
        plot_symbol_chart(df=m15, symbol=symbol, output_path=charts_dir / f"{symbol.replace('/', '_')}.png", trade=trades[-1] if trades else None, confidence_score=signal.confidence_score)

    stats = compute_trade_stats(all_trades)
    plot_equity_curve(all_trades, reports_dir / "equity_curve.png"); save_backtest_summary(stats, reports_dir)
    gross_total = sum(t.gross_pnl for t in all_trades); net_total = stats.net_pnl_after_fees
    logger.info("Backtest: trades=%s winrate=%.2f%% net=%.4f gross=%.4f fees=%.4f pf=%.4f dd=%.4f\nPnL by symbol=%s\nPnL by dir=%s\nPnL by tag=%s", stats.total_trades, stats.winrate, stats.net_pnl_after_fees, gross_total, stats.total_fees_paid, stats.profit_factor, stats.max_drawdown, stats.pnl_by_symbol, stats.pnl_by_direction, stats.pnl_by_tag)
