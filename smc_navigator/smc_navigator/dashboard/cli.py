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
from smc_navigator.strategy.rules import evaluate_signal


def _build_exchange(name: str):
    if name.lower() == "kraken":
        return KrakenExchange()
    if name.lower() == "binance":
        return BinanceExchange()
    raise ValueError(f"Unsupported exchange: {name}")


def _print_trade_table(logger, trades) -> None:
    if not trades:
        logger.info("No trades generated.")
        return

    rows = []
    for t in trades:
        rows.append(
            {
                "symbol": t.symbol,
                "direction": t.direction,
                "entry": round(t.entry_price, 4),
                "exit": round(t.exit_price or 0.0, 4),
                "pnl": round(t.gross_pnl, 4),
                "pnl_after_fees": round(t.pnl or 0.0, 4),
                "rr_ratio": round(t.rr_ratio, 3),
                "confidence": t.confidence_score,
                "trade_tags": t.tags,
            }
        )

    df = pd.DataFrame(rows)
    logger.info("\n=== Detailed Trades ===\n%s", df.to_string(index=False))


def _save_trades_report(reports_dir: Path, trades) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in trades:
        rows.append(
            {
                "trade_id": t.trade_id,
                "timestamp": t.timestamp,
                "symbol": t.symbol,
                "direction": t.direction,
                "entry": t.entry_price,
                "exit": t.exit_price,
                "pnl": t.gross_pnl,
                "pnl_after_fees": t.pnl,
                "rr_ratio": t.rr_ratio,
                "confidence": t.confidence_score,
                "trade_tags": t.tags,
                "reason": t.reason,
                "status": t.status,
            }
        )
    pd.DataFrame(rows).to_csv(reports_dir / "trades_report.csv", index=False)


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
        m15 = add_indicators(fetch_candles_df(exchange, symbol, config["timeframe"]))
        h1 = add_indicators(fetch_candles_df(exchange, symbol, "1h"))
        h1_row = h1.iloc[-1] if not h1.empty else None
        signal = evaluate_signal(
            symbol,
            m15,
            config["default_stop_loss_pct"],
            config["default_take_profit_pct"],
            h1_close=float(h1_row["close"]) if h1_row is not None else None,
            h1_ema50=float(h1_row["ema_50"]) if h1_row is not None else None,
        )
        trades = run_backtest_for_symbol(config=config, symbol=symbol, enriched_df=m15, h1_df=h1, journal_path=str(journal_path))
        all_trades.extend(trades)

        logger.info(
            "\nSymbol: %s\nSignal: %s\nConfidence: %s\nEntry: %.4f\nSL: %.4f\nTP: %.4f\nReason: %s\nTags: %s\nBacktest trades: %s\n",
            symbol,
            "NO_TRADE" if signal.direction == "NONE" else f"{signal.direction}_CANDIDATE",
            signal.confidence_score,
            signal.entry_price,
            signal.suggested_stop_loss,
            signal.suggested_take_profit,
            "; ".join(signal.reason),
            ", ".join(signal.tags) if signal.tags else "none",
            len(trades),
        )

        plot_symbol_chart(df=m15, symbol=symbol, output_path=charts_dir / f"{symbol.replace('/', '_')}.png", trade=trades[-1] if trades else None, confidence_score=signal.confidence_score)

    stats = compute_trade_stats(all_trades)
    plot_equity_curve(all_trades, reports_dir / "equity_curve.png")
    save_backtest_summary(stats, reports_dir)
    _save_trades_report(reports_dir, all_trades)
    _print_trade_table(logger, all_trades)

    if all_trades:
        best_trade = max(all_trades, key=lambda t: float(t.pnl or 0.0))
        worst_trade = min(all_trades, key=lambda t: float(t.pnl or 0.0))
        logger.info("\nBest trade: %s %s pnl_after_fees=%.4f tags=%s", best_trade.symbol, best_trade.direction, float(best_trade.pnl or 0.0), best_trade.tags)
        logger.info("Worst trade: %s %s pnl_after_fees=%.4f tags=%s", worst_trade.symbol, worst_trade.direction, float(worst_trade.pnl or 0.0), worst_trade.tags)

        losing_reasons = {}
        for t in all_trades:
            if (t.pnl or 0.0) < 0:
                key = (t.reason.split(";")[0] or "unknown").strip().lower()
                losing_reasons[key] = losing_reasons.get(key, 0) + 1
        top_losing = sorted(losing_reasons.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info("Top losing reason categories: %s", top_losing)

    gross_total = sum(t.gross_pnl for t in all_trades)
    logger.info(
        "\n=== Global Backtest Summary ===\n"
        "Total trades: %s | Wins: %s | Losses: %s | Winrate: %.2f%%\n"
        "Net after fees: %.4f | Gross before fees: %.4f | Fees: %.4f | Profit factor: %.4f | Max DD: %.4f\n"
        "PnL by symbol: %s\nPnL by direction: %s\nPnL by tag: %s\n"
        "Detailed report: %s\n",
        stats.total_trades,
        stats.wins,
        stats.losses,
        stats.winrate,
        stats.net_pnl_after_fees,
        gross_total,
        stats.total_fees_paid,
        stats.profit_factor,
        stats.max_drawdown,
        stats.pnl_by_symbol,
        stats.pnl_by_direction,
        stats.pnl_by_tag,
        reports_dir / "trades_report.csv",
    )
