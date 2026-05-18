from dataclasses import asdict
from pathlib import Path
import json

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
from smc_navigator.simulator.trade import Trade
from smc_navigator.strategy.investor_engine import evaluate_investor_signal
from smc_navigator.strategy.swing_engine import evaluate_swing_signal


def _build_exchange(name: str):
    if name.lower() == "kraken":
        return KrakenExchange()
    if name.lower() == "binance":
        return BinanceExchange()
    raise ValueError(f"Unsupported exchange: {name}")


def _fetch(ex, cfg, symbol, timeframe):
    hist = cfg["historical_fetch"]
    return fetch_candles_df(
        ex,
        symbol,
        timeframe,
        limit=hist["historical_limit_per_symbol"],
        since=hist.get("backtest_since"),
        until=hist.get("backtest_until"),
        max_fetch_batches=hist.get("max_fetch_batches", 1),
        refresh_market_data=cfg.get("refresh_market_data", False),
    )


def _save_trades_csv(path: Path, trades: list[Trade]) -> None:
    rows = [asdict(t) for t in trades]
    pd.DataFrame(rows).to_csv(path, index=False)


def _save_summary_files(stats, reports_dir: Path, prefix: str) -> None:
    payload = asdict(stats)
    (reports_dir / f"{prefix}_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame({"metric": list(payload.keys()), "value": [json.dumps(v) if isinstance(v, dict) else v for v in payload.values()]}).to_csv(
        reports_dir / f"{prefix}_summary.csv", index=False
    )


def _simulate_investor_trades(symbol: str, daily: pd.DataFrame, weekly: pd.DataFrame, monthly: pd.DataFrame, capital: float) -> list[Trade]:
    trades: list[Trade] = []
    open_trade: Trade | None = None
    if daily.empty:
        return trades

    for i in range(60, len(daily)):
        d_hist = daily.iloc[: i + 1]
        cutoff = d_hist.iloc[-1]["timestamp"]
        w_hist = weekly[weekly["timestamp"] <= cutoff]
        m_hist = monthly[monthly["timestamp"] <= cutoff]
        sig = evaluate_investor_signal(m_hist, w_hist, d_hist)
        price = float(d_hist.iloc[-1]["close"])

        if sig.signal == "INVEST_LONG" and open_trade is None:
            position_size = (capital * 0.5) / max(price, 1e-9)
            open_trade = Trade(
                trade_id=f"invest-{symbol}-{i}",
                timestamp=d_hist.iloc[-1]["timestamp"].to_pydatetime(),
                exchange="sim",
                symbol=symbol,
                timeframe="1d",
                direction="LONG",
                entry_price=price,
                stop_loss=price * 0.8,
                take_profit=price * 1.5,
                position_size=position_size,
                risk_amount=capital * 0.02,
                confidence_score=sig.score,
                status="OPEN",
                exit_price=None,
                pnl=None,
                pnl_pct=None,
                gross_pnl=0.0,
                entry_fee=0.0,
                exit_fee=0.0,
                total_fees=0.0,
                holding_candles=0,
                rr_ratio=2.5,
                reason=";".join(sig.reasons),
                tags="investor_engine",
            )
        elif sig.signal == "INVEST_EXIT" and open_trade is not None:
            hold = i - 60
            open_trade.close(price, "CLOSED_MANUAL", holding_candles=max(1, hold), entry_fee=0.0, exit_fee=0.0)
            trades.append(open_trade)
            open_trade = None

    if open_trade is not None:
        last_price = float(daily.iloc[-1]["close"])
        open_trade.close(last_price, "EXPIRED", holding_candles=max(1, len(daily) - 60), entry_fee=0.0, exit_fee=0.0)
        trades.append(open_trade)
    return trades


def run(config_path: str = "config.yaml") -> None:
    logger = get_logger()
    cfg = load_config(config_path)
    if not cfg.get("simulation_mode", True) or cfg.get("allow_real_orders", False):
        raise RuntimeError("Safety check failed: only simulation_mode=true and allow_real_orders=false are supported")

    ex = _build_exchange(cfg["exchange"])
    reports_dir = Path("reports")
    charts_dir = reports_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    journal_path = Path("data/trade_journal.csv")

    investor_trades_all: list[Trade] = []
    swing_trades_all: list[Trade] = []

    for symbol in cfg["symbols"]:
        h4, _ = _fetch(ex, cfg, symbol, cfg["swing"]["timeframes"]["execution"])
        d1, _ = _fetch(ex, cfg, symbol, cfg["swing"]["timeframes"]["confirmation"])
        w1, _ = _fetch(ex, cfg, symbol, cfg["swing"]["timeframes"]["context"])
        m1, _ = _fetch(ex, cfg, symbol, cfg["investor"]["timeframes"]["macro"])
        iw, _ = _fetch(ex, cfg, symbol, cfg["investor"]["timeframes"]["confirmation"])
        idy, _ = _fetch(ex, cfg, symbol, cfg["investor"]["timeframes"]["timing"])

        if cfg["swing"]["enabled"] and not h4.empty:
            h4i = add_indicators(h4)
            h1, _ = _fetch(ex, cfg, symbol, "1h")
            h1i = add_indicators(h1) if not h1.empty else h1
            swing_sig = evaluate_swing_signal(w1, d1, h4)
            swing_cfg = {
                "exchange": cfg["exchange"],
                "timeframe": cfg["swing"]["timeframes"]["execution"],
                "starting_capital": cfg["swing"]["capital"],
                "risk_per_trade_pct": 1.0,
                "default_stop_loss_pct": cfg["swing"]["default_stop_loss_pct"],
                "default_take_profit_pct": cfg["swing"]["take_profit_targets_pct"][0],
                "maker_fee_pct": cfg["fees"]["maker_fee_pct"],
                "taker_fee_pct": cfg["fees"]["taker_fee_pct"],
                "spread_pct": cfg["fees"]["spread_pct"],
            }
            swing_trades = run_backtest_for_symbol(swing_cfg, symbol, h4i, str(journal_path), h1_df=h1i if not h1i.empty else None, h4_df=h4i)
            swing_trades_all.extend(swing_trades)
            plot_symbol_chart(h4i, f"{symbol}_SWING", charts_dir / f"{symbol.replace('/', '_')}_swing.png", trade=swing_trades[-1] if swing_trades else None, confidence_score=swing_sig.score)

        if cfg["investor"]["enabled"] and not idy.empty:
            investor_sig = evaluate_investor_signal(m1, iw, idy)
            investor_trades = _simulate_investor_trades(symbol, idy, iw, m1, float(cfg["investor"]["capital"]))
            investor_trades_all.extend(investor_trades)
            plot_symbol_chart(add_indicators(idy), f"{symbol}_INVESTOR", charts_dir / f"{symbol.replace('/', '_')}_investor.png", trade=investor_trades[-1] if investor_trades else None, confidence_score=investor_sig.score)

    # Separate reporting
    swing_stats = compute_trade_stats(swing_trades_all)
    investor_stats = compute_trade_stats(investor_trades_all)

    plot_equity_curve(swing_trades_all, reports_dir / "equity_curve_swing.png")
    plot_equity_curve(investor_trades_all, reports_dir / "equity_curve_investor.png")

    _save_trades_csv(reports_dir / "swing_trades_report.csv", swing_trades_all)
    _save_trades_csv(reports_dir / "investor_trades_report.csv", investor_trades_all)

    _save_summary_files(swing_stats, reports_dir, "swing")
    _save_summary_files(investor_stats, reports_dir, "investor")

    # keep legacy aggregate summary too
    save_backtest_summary(compute_trade_stats(swing_trades_all + investor_trades_all), reports_dir)

    logger.info("Investor trades=%s Swing trades=%s", len(investor_trades_all), len(swing_trades_all))
