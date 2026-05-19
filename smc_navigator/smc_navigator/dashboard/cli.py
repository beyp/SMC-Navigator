from dataclasses import asdict
from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt

from smc_navigator.core.config_loader import load_config
from smc_navigator.core.logger import get_logger
from smc_navigator.core.config_validation import ensure_timeframes
from smc_navigator.exchanges.binance import BinanceExchange
from smc_navigator.exchanges.kraken import KrakenExchange
from smc_navigator.market_data.candles import fetch_candles_df
from smc_navigator.market_data.indicators import add_indicators
from smc_navigator.reporting.charts import plot_equity_curve, plot_symbol_chart, plot_yearly_equity_curve, plot_rolling_drawdown, plot_regime_performance
from smc_navigator.reporting.stats import compute_trade_stats
from smc_navigator.simulator.engine import run_backtest_for_symbol
from smc_navigator.simulator.trade import Trade
from smc_navigator.strategy.investor_engine import evaluate_investor_signal
from smc_navigator.strategy.swing_engine import evaluate_swing_signal


def _build_exchange(name: str):
    if name.lower() == "kraken": return KrakenExchange()
    if name.lower() == "binance": return BinanceExchange()
    raise ValueError(f"Unsupported exchange: {name}")


def _fetch(ex, shared, symbol, timeframe):
    hist = shared["historical_fetch"]
    return fetch_candles_df(
        ex, symbol, timeframe,
        limit=hist["historical_limit_per_symbol"],
        since=hist.get("backtest_since"),
        until=hist.get("backtest_until"),
        max_fetch_batches=hist.get("max_fetch_batches", 1),
        refresh_market_data=shared.get("refresh_market_data", False),
    )


def _save_trades_csv(path: Path, trades: list[Trade]) -> None:
    pd.DataFrame([asdict(t) for t in trades]).to_csv(path, index=False)


def _save_summary(path_json: Path, path_csv: Path, stats) -> None:
    payload = asdict(stats)
    path_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame({"metric": list(payload.keys()), "value": [json.dumps(v) if isinstance(v, dict) else v for v in payload.values()]}).to_csv(path_csv, index=False)


def _simulate_investor_trades(symbol: str, daily: pd.DataFrame, weekly: pd.DataFrame, monthly: pd.DataFrame, capital: float, exchange: str) -> list[Trade]:
    trades=[]; open_trade=None
    for i in range(60, len(daily)):
        d_hist=daily.iloc[:i+1]; cutoff=d_hist.iloc[-1]["timestamp"]
        sig=evaluate_investor_signal(monthly[monthly["timestamp"]<=cutoff], weekly[weekly["timestamp"]<=cutoff], d_hist)
        price=float(d_hist.iloc[-1]["close"])
        if sig.signal=="INVEST_LONG" and open_trade is None:
            size=(capital*0.5)/max(price,1e-9)
            open_trade=Trade(f"inv-{exchange}-{symbol}-{i}", d_hist.iloc[-1]["timestamp"].to_pydatetime(), exchange, symbol, "1d", "LONG", price, price*0.8, price*1.5, size, capital*0.02, sig.score, "OPEN", None, None, None, 0.0, 0.0, 0.0, 0.0, 0, 2.5, ";".join(sig.reasons), "investor_engine")
        elif sig.signal=="INVEST_EXIT" and open_trade is not None:
            open_trade.close(price, "CLOSED_MANUAL", max(1,i-60), 0.0, 0.0); trades.append(open_trade); open_trade=None
    if open_trade is not None:
        open_trade.close(float(daily.iloc[-1]["close"]), "EXPIRED", max(1,len(daily)-60), 0.0, 0.0); trades.append(open_trade)
    return trades


def _plot_investor_regime_chart(path: Path, daily: pd.DataFrame, trades: list[Trade], regime_label: str) -> None:
    if daily.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily["timestamp"], daily["close"], label="Daily Close", color="black")
    color = {
        "accumulation": "#b3e5fc",
        "bullish_expansion": "#c8e6c9",
        "distribution": "#ffe0b2",
        "bearish_expansion": "#ffcdd2",
    }.get(regime_label, "#eeeeee")
    ax.axhspan(daily["close"].min(), daily["close"].max(), color=color, alpha=0.2, label=f"Regime: {regime_label}")
    for tr in trades:
        ax.scatter(tr.timestamp, tr.entry_price, marker="^", color="green")
        if tr.exit_price is not None:
            ax.scatter(tr.timestamp, tr.exit_price, marker="v", color="red")
    ax.legend(); ax.set_title("Investor Regime + Entries/Exits")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def _buy_hold_return(daily: pd.DataFrame, capital: float) -> float:
    if daily.empty:
        return 0.0
    first = float(daily.iloc[0]["close"]); last = float(daily.iloc[-1]["close"])
    if first <= 0:
        return 0.0
    return capital * ((last / first) - 1)


def run(config_path: str = "config.yaml") -> None:
    logger = get_logger()
    cfg = ensure_timeframes(load_config(config_path))
    shared = cfg["shared"]
    if not shared.get("simulation_mode", True) or shared.get("allow_real_orders", False):
        raise RuntimeError("Safety check failed: only simulation_mode=true and allow_real_orders=false are supported")

    inv_ex = _build_exchange(cfg["investor"]["exchange"])
    sw_ex = _build_exchange(cfg["swing"]["exchange"])
    reports=Path("reports"); charts=reports/"charts"; charts.mkdir(parents=True, exist_ok=True)
    journal_path=Path("data/trade_journal.csv")

    investor_trades=[]; swing_trades=[]
    buy_hold_results = {}

    logger.info("Investor strategy exchange=%s capital=%s fees(maker/taker)=%.3f/%.3f", cfg['investor']['exchange'], cfg['investor']['capital'], cfg['investor']['maker_fee_pct'], cfg['investor']['taker_fee_pct'])
    logger.info("Swing strategy exchange=%s capital=%s fees(maker/taker)=%.3f/%.3f", cfg['swing']['exchange'], cfg['swing']['capital'], cfg['swing']['maker_fee_pct'], cfg['swing']['taker_fee_pct'])

    for symbol in cfg["investor"]["symbols"]:
        # larger historical windows for investor cycle testing
        shared_hist = shared["historical_fetch"]
        original_limit = shared_hist["historical_limit_per_symbol"]
        shared_hist["historical_limit_per_symbol"] = max(original_limit, 240)
        m1,_=_fetch(inv_ex,shared,symbol,cfg['investor']['timeframes']['macro'])
        shared_hist["historical_limit_per_symbol"] = max(original_limit, 300)
        w1,_=_fetch(inv_ex,shared,symbol,cfg['investor']['timeframes']['confirmation'])
        shared_hist["historical_limit_per_symbol"] = max(original_limit, 1100)
        d1,_=_fetch(inv_ex,shared,symbol,cfg['investor']['timeframes']['timing'])
        shared_hist["historical_limit_per_symbol"] = original_limit
        if d1.empty: continue
        investor_trades.extend(_simulate_investor_trades(symbol,d1,w1,m1,float(cfg['investor']['capital']),cfg['investor']['exchange']))
        plot_symbol_chart(add_indicators(d1), f"{symbol}_INVESTOR", charts/f"{symbol.replace('/','_')}_investor.png", trade=investor_trades[-1] if investor_trades else None, confidence_score=60)

    for symbol in cfg["swing"]["symbols"]:
        h4,_=_fetch(sw_ex,shared,symbol,cfg['swing']['timeframes']['execution'])
        d1,_=_fetch(sw_ex,shared,symbol,cfg['swing']['timeframes']['confirmation'])
        w1,_=_fetch(sw_ex,shared,symbol,cfg['swing']['timeframes']['context'])
        h1,_=_fetch(sw_ex,shared,symbol,"1h")
        m15,_=_fetch(sw_ex,shared,symbol,"15m")
        m5,_=_fetch(sw_ex,shared,symbol,"5m")
        if h4.empty: continue
        h4i=add_indicators(h4)
        h1i=add_indicators(h1) if not h1.empty else h1
        m15i=add_indicators(m15) if not m15.empty else m15
        m5i=add_indicators(m5) if not m5.empty else m5
        swing_sig=evaluate_swing_signal(w1,d1,h4,h1=h1i if not h1i.empty else None,m15=m15i if not m15i.empty else None,m5=m5i if not m5i.empty else None)
        swing_cfg={"exchange":cfg['swing']['exchange'],"timeframe":cfg['swing']['timeframes']['execution'],"starting_capital":cfg['swing']['capital'],"risk_per_trade_pct":1.0,"default_stop_loss_pct":cfg['swing']['default_stop_loss_pct'],"default_take_profit_pct":cfg['swing']['take_profit_targets_pct'][0],"maker_fee_pct":cfg['swing']['maker_fee_pct'],"taker_fee_pct":cfg['swing']['taker_fee_pct'],"spread_pct":cfg['swing']['spread_pct']}
        logger.info("Swing signal %s %s score=%s tags=%s pullback=[%.4f, %.4f, %.4f]", symbol, swing_sig.signal, swing_sig.score, swing_sig.tags, swing_sig.pullback_30 or 0.0, swing_sig.pullback_50 or 0.0, swing_sig.pullback_618 or 0.0)
        t=run_backtest_for_symbol(swing_cfg,symbol,h4i,str(journal_path),h1_df=h1i if not h1i.empty else None,h4_df=h4i)
        swing_trades.extend(t)
        plot_symbol_chart(h4i, f"{symbol}_SWING", charts/f"{symbol.replace('/','_')}_swing.png", trade=t[-1] if t else None, confidence_score=swing_sig.score)

    investor_stats=compute_trade_stats(investor_trades)
    swing_stats=compute_trade_stats(swing_trades)
    combined_stats=compute_trade_stats(investor_trades+swing_trades)

    plot_equity_curve(investor_trades, reports/"equity_curve_investor.png")
    plot_equity_curve(swing_trades, reports/"equity_curve_swing.png")
    plot_equity_curve(investor_trades+swing_trades, reports/"equity_curve_combined.png")

    _save_trades_csv(reports/"investor_trades_report.csv", investor_trades)
    _save_trades_csv(reports/"swing_trades_report.csv", swing_trades)
    _save_summary(reports/"investor_summary.json", reports/"investor_summary.csv", investor_stats)
    _save_summary(reports/"swing_summary.json", reports/"swing_summary.csv", swing_stats)
    _save_summary(reports/"combined_summary.json", reports/"combined_summary.csv", combined_stats)

    plot_yearly_equity_curve(investor_trades, reports/"yearly_equity_curve_investor.png")
    plot_yearly_equity_curve(swing_trades, reports/"yearly_equity_curve_swing.png")
    plot_rolling_drawdown(investor_trades, reports/"rolling_drawdown_investor.png")
    plot_rolling_drawdown(swing_trades, reports/"rolling_drawdown_swing.png")
    plot_regime_performance(investor_stats.performance_by_regime, reports/"regime_performance_investor.png")

    benchmark = {
        "buy_and_hold": _buy_hold_return(d1, float(cfg["investor"]["capital"])) if "d1" in locals() and not d1.empty else 0.0,
        "investor_engine": investor_stats.net_pnl_after_fees,
        "swing_engine": swing_stats.net_pnl_after_fees,
    }
    (reports/"benchmark_comparison.json").write_text(json.dumps(benchmark, indent=2), encoding="utf-8")
    (reports/"buy_and_hold_comparison.json").write_text(json.dumps(buy_hold_results, indent=2), encoding="utf-8")
