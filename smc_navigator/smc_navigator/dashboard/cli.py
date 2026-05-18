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
from smc_navigator.strategy.investor_engine import evaluate_investor_signal
from smc_navigator.strategy.swing_engine import evaluate_swing_signal


def _build_exchange(name: str):
    if name.lower() == "kraken":
        return KrakenExchange()
    if name.lower() == "binance":
        return BinanceExchange()
    raise ValueError(f"Unsupported exchange: {name}")


def _engine_config(base: dict, capital: float, sl: float, tp: float, max_days: int) -> dict:
    fees = base["fees"]
    return {
        "exchange": base["exchange"],
        "timeframe": "",
        "starting_capital": capital,
        "risk_per_trade_pct": 1.0,
        "default_stop_loss_pct": sl,
        "default_take_profit_pct": tp,
        "simulation_mode": base["simulation_mode"],
        "allow_real_orders": base["allow_real_orders"],
        "maker_fee_pct": fees["maker_fee_pct"],
        "taker_fee_pct": fees["taker_fee_pct"],
        "spread_pct": fees["spread_pct"],
        "min_confidence_score": 0,
        "minimum_setup_score": 0,
        "max_trades_per_symbol": 999,
        "cooldown_candles_after_trade": 0,
        "max_distance_from_vwap_pct": 100,
        "max_distance_from_ema26_pct": 100,
        "min_ema_distance_pct": 0,
        "min_atr_pct": 0,
        "min_range_width_pct": 0,
        "min_rr_ratio": 0,
        "max_holding_candles": max_days,
    }


def run(config_path: str = "config.yaml") -> None:
    logger = get_logger()
    cfg = load_config(config_path)
    if not cfg.get("simulation_mode", True) or cfg.get("allow_real_orders", False):
        raise RuntimeError("Safety check failed: only simulation_mode=true and allow_real_orders=false are supported")

    ex = _build_exchange(cfg["exchange"])
    hist = cfg["historical_fetch"]
    reports_dir = Path("reports")
    charts_dir = reports_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    journal_path = Path("data/trade_journal.csv")

    logger.info("Investor capital=%s Swing capital=%s", cfg["investor"]["capital"], cfg["swing"]["capital"])
    logger.info("Investor TFs=%s Swing TFs=%s", cfg["investor"]["timeframes"], cfg["swing"]["timeframes"])

    all_swing_trades = []
    all_investor_trades = []

    for symbol in cfg["symbols"]:
        # Swing data
        h4 = fetch_candles_df(ex, symbol, cfg["swing"]["timeframes"]["execution"], limit=hist["historical_limit_per_symbol"], since=hist.get("backtest_since"), until=hist.get("backtest_until"), max_fetch_batches=hist.get("max_fetch_batches", 1))
        d1 = fetch_candles_df(ex, symbol, cfg["swing"]["timeframes"]["confirmation"], limit=hist["historical_limit_per_symbol"], since=hist.get("backtest_since"), until=hist.get("backtest_until"), max_fetch_batches=hist.get("max_fetch_batches", 1))
        w1 = fetch_candles_df(ex, symbol, cfg["swing"]["timeframes"]["context"], limit=hist["historical_limit_per_symbol"], since=hist.get("backtest_since"), until=hist.get("backtest_until"), max_fetch_batches=hist.get("max_fetch_batches", 1))

        # Investor data
        m1 = fetch_candles_df(ex, symbol, cfg["investor"]["timeframes"]["macro"], limit=hist["historical_limit_per_symbol"], since=hist.get("backtest_since"), until=hist.get("backtest_until"), max_fetch_batches=hist.get("max_fetch_batches", 1))
        iw = fetch_candles_df(ex, symbol, cfg["investor"]["timeframes"]["confirmation"], limit=hist["historical_limit_per_symbol"], since=hist.get("backtest_since"), until=hist.get("backtest_until"), max_fetch_batches=hist.get("max_fetch_batches", 1))
        idy = fetch_candles_df(ex, symbol, cfg["investor"]["timeframes"]["timing"], limit=hist["historical_limit_per_symbol"], since=hist.get("backtest_since"), until=hist.get("backtest_until"), max_fetch_batches=hist.get("max_fetch_batches", 1))

        if not h4.empty:
            logger.info("%s swing range: %s -> %s (%s candles)", symbol, h4.iloc[0]["timestamp"], h4.iloc[-1]["timestamp"], len(h4))
        if not idy.empty:
            logger.info("%s investor range: %s -> %s (%s candles)", symbol, idy.iloc[0]["timestamp"], idy.iloc[-1]["timestamp"], len(idy))

        swing_sig = evaluate_swing_signal(w1, d1, h4)
        investor_sig = evaluate_investor_signal(m1, iw, idy)
        logger.info("%s swing=%s score=%s | investor=%s score=%s", symbol, swing_sig.signal, swing_sig.score, investor_sig.signal, investor_sig.score)

        if cfg["swing"]["enabled"] and not h4.empty:
            h4i = add_indicators(h4)
            h1i = add_indicators(fetch_candles_df(ex, symbol, "1h", limit=hist["historical_limit_per_symbol"], since=hist.get("backtest_since"), until=hist.get("backtest_until"), max_fetch_batches=hist.get("max_fetch_batches", 1)))
            swing_cfg = _engine_config(cfg, cfg["swing"]["capital"], cfg["swing"]["default_stop_loss_pct"], cfg["swing"]["take_profit_targets_pct"][0], cfg["swing"]["max_position_days"] * 6)
            swing_trades = run_backtest_for_symbol(swing_cfg, symbol, h4i, str(journal_path), h1_df=h1i, h4_df=h4i)
            all_swing_trades.extend(swing_trades)
            plot_symbol_chart(h4i, f"{symbol}_SWING", charts_dir / f"{symbol.replace('/', '_')}_swing.png", trade=swing_trades[-1] if swing_trades else None, confidence_score=swing_sig.score)

        if cfg["investor"]["enabled"] and not idy.empty:
            idyi = add_indicators(idy)
            iweekly = add_indicators(iw)
            investor_cfg = _engine_config(cfg, cfg["investor"]["capital"], cfg["investor"]["default_stop_loss_pct"], cfg["investor"]["take_profit_targets_pct"][0], cfg["investor"]["max_position_months"] * 30)
            investor_trades = run_backtest_for_symbol(investor_cfg, symbol, idyi, str(journal_path), h1_df=iweekly, h4_df=add_indicators(m1))
            all_investor_trades.extend(investor_trades)
            plot_symbol_chart(idyi, f"{symbol}_INVESTOR", charts_dir / f"{symbol.replace('/', '_')}_investor.png", trade=investor_trades[-1] if investor_trades else None, confidence_score=investor_sig.score)

    all_trades = all_swing_trades + all_investor_trades
    stats = compute_trade_stats(all_trades)
    plot_equity_curve(all_trades, reports_dir / "equity_curve.png")
    save_backtest_summary(stats, reports_dir)
    logger.info("Summary total=%s swing=%s investor=%s", len(all_trades), len(all_swing_trades), len(all_investor_trades))
