from pathlib import Path

from smc_navigator.core.config_loader import load_config
from smc_navigator.core.logger import get_logger
from smc_navigator.exchanges.binance import BinanceExchange
from smc_navigator.exchanges.kraken import KrakenExchange
from smc_navigator.market_data.candles import fetch_candles_df
from smc_navigator.market_data.indicators import add_indicators
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

    for symbol in config["symbols"]:
        candles = fetch_candles_df(exchange, symbol, config["timeframe"])
        enriched = add_indicators(candles)

        latest_signal = evaluate_signal(symbol, enriched, config["default_stop_loss_pct"], config["default_take_profit_pct"])
        signal_name = "NO_TRADE" if latest_signal.direction == "NONE" else f"{latest_signal.direction}_CANDIDATE"

        trades = run_backtest_for_symbol(
            config=config,
            symbol=symbol,
            enriched_df=enriched,
            journal_path=str(journal_path),
        )

        logger.info(
            "\nSymbol: %s\nSignal: %s\nConfidence: %s\nEntry: %.4f\nSL: %.4f\nTP: %.4f\nReason: %s\nBacktest trades: %s\n",
            symbol,
            signal_name,
            latest_signal.confidence_score,
            latest_signal.entry_price,
            latest_signal.suggested_stop_loss,
            latest_signal.suggested_take_profit,
            "; ".join(latest_signal.reason),
            len(trades),
        )
