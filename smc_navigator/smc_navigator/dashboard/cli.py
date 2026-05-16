from pathlib import Path

from smc_navigator.core.config_loader import load_config
from smc_navigator.core.logger import get_logger
from smc_navigator.exchanges.binance import BinanceExchange
from smc_navigator.exchanges.kraken import KrakenExchange
from smc_navigator.market_data.candles import fetch_candles_df
from smc_navigator.market_data.indicators import add_indicators
from smc_navigator.simulator.engine import simulate_signal
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
        signal = evaluate_signal(symbol, enriched, config["default_stop_loss_pct"], config["default_take_profit_pct"])
        trade = simulate_signal(config, signal, enriched, str(journal_path))

        signal_name = "NO_TRADE" if signal.direction == "NONE" else f"{signal.direction}_CANDIDATE"
        logger.info(
            "\nSymbol: %s\nSignal: %s\nConfidence: %s\nEntry: %.4f\nSL: %.4f\nTP: %.4f\nReason: %s\nTrade status: %s\n",
            symbol,
            signal_name,
            signal.confidence_score,
            signal.entry_price,
            signal.suggested_stop_loss,
            signal.suggested_take_profit,
            "; ".join(signal.reason),
            trade.status if trade else "N/A",
        )
