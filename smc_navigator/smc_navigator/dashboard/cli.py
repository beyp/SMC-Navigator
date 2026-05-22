from dataclasses import asdict
from pathlib import Path
import json
import time

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
        refresh_market_data=shared["historical_fetch"].get("refresh_market_data", False),
        request_delay_seconds=hist.get("request_delay_seconds", 0),
        max_retries=hist.get("max_retries", 0),
        retry_backoff_seconds=hist.get("retry_backoff_seconds", 1),
    )


def _save_trades_csv(path: Path, trades: list[Trade]) -> None:
    pd.DataFrame([asdict(t) for t in trades]).to_csv(path, index=False)


def _save_summary(path_json: Path, path_csv: Path, stats, enabled_features: dict[str, bool] | None = None) -> None:
    payload = asdict(stats)
    payload["enabled_features"] = enabled_features or {}
    path_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame({"metric": list(payload.keys()), "value": [json.dumps(v) if isinstance(v, dict) else v for v in payload.values()]}).to_csv(path_csv, index=False)


def _simulate_investor_trades(symbol: str, daily: pd.DataFrame, weekly: pd.DataFrame, monthly: pd.DataFrame, capital: float, exchange: str, features: dict[str, bool] | None = None) -> list[Trade]:
    trades=[]; open_trade=None
    for i in range(60, len(daily)):
        d_hist=daily.iloc[:i+1]; cutoff=d_hist.iloc[-1]["timestamp"]
        sig=evaluate_investor_signal(monthly[monthly["timestamp"]<=cutoff], weekly[weekly["timestamp"]<=cutoff], d_hist, features=features)
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


def _deadline_exceeded(start_ts: float, max_runtime_minutes: float) -> bool:
    return (time.time() - start_ts) > max_runtime_minutes * 60

def _select_symbols(symbols: list[str], debug_symbol: str | None) -> list[str]:
    if debug_symbol:
        return [s for s in symbols if s == debug_symbol]
    return symbols

def _slice_recent(df: pd.DataFrame, n: int | None) -> pd.DataFrame:
    if df.empty or not n or n <= 0:
        return df
    return df.tail(int(n)).reset_index(drop=True)

def run(config_path: str = "config.yaml") -> None:
    logger = get_logger()
    cfg = ensure_timeframes(load_config(config_path))
    shared = cfg["shared"]
    raw_features = cfg.get("features", {})
    features = {k: bool(v.get("enabled", False)) for k, v in raw_features.items() if isinstance(v, dict)}
    run_mode = str(shared.get("run_mode", "backtest")).lower()
    debug_symbol = shared.get("debug_symbol")
    max_runtime_minutes = float(shared.get("max_runtime_minutes", 60))
    fast_backtest = bool(shared.get("fast_backtest", False))
    start_ts = time.time()
    fetch_seconds = 0.0
    indicator_seconds = 0.0
    signal_seconds = 0.0
    bos_seconds = 0.0
    reclaim_seconds = 0.0
    chart_seconds = 0.0
    reporting_seconds = 0.0
    if run_mode == "live":
        logger.info("entering LIVE mode")
        shared["historical_fetch"]["historical_limit_per_symbol"] = min(int(shared["historical_fetch"].get("historical_limit_per_symbol", 300)), 300)
        shared["historical_fetch"]["max_fetch_batches"] = min(int(shared["historical_fetch"].get("max_fetch_batches", 2)), 2)
        features["m5_execution"] = False
    else:
        logger.info("entering BACKTEST mode")
    if not shared.get("simulation_mode", True) or shared.get("allow_real_orders", False):
        raise RuntimeError("Safety check failed: only simulation_mode=true and allow_real_orders=false are supported")


    inv_tf = cfg["investor"].get("timeframes", {})
    sw_tf = cfg["swing"].get("timeframes", {})
    if inv_tf.get("macro") is None:
        raise ValueError("Missing config: investor.timeframes.macro")
    if inv_tf.get("confirmation") is None:
        raise ValueError("Missing config: investor.timeframes.confirmation")
    if inv_tf.get("timing") is None:
        raise ValueError("Missing config: investor.timeframes.timing")
    if sw_tf.get("context") is None:
        raise ValueError("Missing config: swing.timeframes.context")
    if sw_tf.get("confirmation") is None:
        raise ValueError("Missing config: swing.timeframes.confirmation")
    if sw_tf.get("execution") is None:
        raise ValueError("Missing config: swing.timeframes.execution")
    inv_ex = _build_exchange(cfg["investor"]["exchange"])
    sw_ex = _build_exchange(cfg["swing"]["exchange"])
    logger.info("Investor exchange=%s", cfg["investor"]["exchange"])
    for name, enabled in features.items():
        logger.info("%s: %s", name.replace("_", " ").title(), "ENABLED" if enabled else "DISABLED")
    logger.info("Investor timeframes: macro=%s confirmation=%s timing=%s", inv_tf["macro"], inv_tf["confirmation"], inv_tf["timing"])
    logger.info("Swing exchange=%s", cfg["swing"]["exchange"])
    logger.info("Swing timeframes: context=%s confirmation=%s execution=%s", sw_tf["context"], sw_tf["confirmation"], sw_tf["execution"])
    reports=Path("reports"); charts=reports/"charts"; charts.mkdir(parents=True, exist_ok=True)
    detailed_visuals = bool(cfg.get("charts", {}).get("detailed_visuals", False))
    enable_charts = bool(shared.get("enable_charts", False))
    if run_mode == "backtest" and fast_backtest:
        enable_charts = False
    journal_path=Path("data/trade_journal.csv")

    investor_trades=[]; swing_trades=[]
    buy_hold_results = {}

    logger.info("Investor strategy exchange=%s capital=%s fees(maker/taker)=%.3f/%.3f", cfg['investor']['exchange'], cfg['investor']['capital'], cfg['investor']['maker_fee_pct'], cfg['investor']['taker_fee_pct'])
    logger.info("Swing strategy exchange=%s capital=%s fees(maker/taker)=%.3f/%.3f", cfg['swing']['exchange'], cfg['swing']['capital'], cfg['swing']['maker_fee_pct'], cfg['swing']['taker_fee_pct'])

    for i_symbol, symbol in enumerate(_select_symbols(cfg["investor"]["symbols"], debug_symbol), start=1):
        total_symbols = len(_select_symbols(cfg["investor"]["symbols"], debug_symbol))
        if _deadline_exceeded(start_ts, max_runtime_minutes):
            logger.warning("Max runtime reached, stopping gracefully before investor symbol %s", symbol); break
        logger.info("Progress investor symbol %s/%s (%s)", i_symbol, total_symbols, symbol)
        # larger historical windows for investor cycle testing
        shared_hist = shared["historical_fetch"]
        original_limit = shared_hist["historical_limit_per_symbol"]
        shared_hist["historical_limit_per_symbol"] = max(original_limit, 240)
        m1,m1_src=_fetch(inv_ex,shared,symbol,cfg['investor']['timeframes']['macro']); logger.info("timeframe 1/3 macro candles=%s source=%s", len(m1), m1_src)
        shared_hist["historical_limit_per_symbol"] = max(original_limit, 300)
        w1,w1_src=_fetch(inv_ex,shared,symbol,cfg['investor']['timeframes']['confirmation']); logger.info("timeframe 2/3 confirmation candles=%s source=%s", len(w1), w1_src)
        shared_hist["historical_limit_per_symbol"] = max(original_limit, 1100)
        d1,d1_src=_fetch(inv_ex,shared,symbol,cfg['investor']['timeframes']['timing']); logger.info("timeframe 3/3 timing candles=%s source=%s", len(d1), d1_src)
        shared_hist["historical_limit_per_symbol"] = original_limit
        logger.info("%s investor data source: m1=%s w1=%s d1=%s", symbol, m1_src, w1_src, d1_src)
        if d1.empty: continue
        investor_trades.extend(_simulate_investor_trades(symbol,d1,w1,m1,float(cfg['investor']['capital']),cfg['investor']['exchange'],features))
        if enable_charts:
            _tc=time.time()
            plot_symbol_chart(add_indicators(d1), f"{symbol}_INVESTOR", charts/f"{symbol.replace('/','_')}_investor.png", trade=investor_trades[-1] if investor_trades else None, confidence_score=60, overlays={"regime": "investor_htf_zone", "score_breakdown": "rev/cont/exh", "hold_reasons": ["no_reclaim", "weak_bos", "no_confirmation"]}, detailed_visuals=detailed_visuals)
            chart_seconds += (time.time()-_tc)

    for i_symbol, symbol in enumerate(_select_symbols(cfg["swing"]["symbols"], debug_symbol), start=1):
        total_symbols = len(_select_symbols(cfg["swing"]["symbols"], debug_symbol))
        if _deadline_exceeded(start_ts, max_runtime_minutes):
            logger.warning("Max runtime reached, stopping gracefully before swing symbol %s", symbol); break
        logger.info("Progress swing symbol %s/%s (%s)", i_symbol, total_symbols, symbol)
        _t0=time.time(); h4,h4_src=_fetch(sw_ex,shared,symbol,cfg['swing']['timeframes']['execution']); fetch_seconds += (time.time()-_t0); logger.info("timeframe 1/5 execution candles=%s source=%s", len(h4), h4_src)
        _t0=time.time(); d1,d1_src2=_fetch(sw_ex,shared,symbol,cfg['swing']['timeframes']['confirmation']); fetch_seconds += (time.time()-_t0); logger.info("timeframe 2/5 confirmation candles=%s source=%s", len(d1), d1_src2)
        _t0=time.time(); w1,w1_src2=_fetch(sw_ex,shared,symbol,cfg['swing']['timeframes']['context']); fetch_seconds += (time.time()-_t0); logger.info("timeframe 3/5 context candles=%s source=%s", len(w1), w1_src2)
        _t0=time.time(); h1,h1_src=_fetch(sw_ex,shared,symbol,"1h"); fetch_seconds += (time.time()-_t0); logger.info("timeframe 4/5 h1 candles=%s source=%s", len(h1), h1_src)
        _t0=time.time(); m15,m15_src=_fetch(sw_ex,shared,symbol,"15m"); fetch_seconds += (time.time()-_t0); logger.info("timeframe 5/5 m15 candles=%s source=%s", len(m15), m15_src)
        m5,m5_src = (pd.DataFrame(), "skipped")
        if cfg["swing"].get("use_m5_confirmation", False) and features.get("m5_execution", False):
            _t0=time.time(); m5,m5_src=_fetch(sw_ex,shared,symbol,"5m"); fetch_seconds += (time.time()-_t0)
        logger.info("%s swing data source: h4=%s d1=%s w1=%s h1=%s m15=%s m5=%s", symbol, h4_src, d1_src2, w1_src2, h1_src, m15_src, m5_src)
        if run_mode == "live":
            h4 = _slice_recent(h4, 500)
            h1 = _slice_recent(h1, 1000)
            m15 = _slice_recent(m15, 1500)
            logger.info("Applied live limits: h4=%s h1=%s m15=%s", len(h4), len(h1), len(m15))
        if h4.empty: continue
        _t0=time.time()
        h4i=add_indicators(h4)
        h1i=add_indicators(h1) if not h1.empty else h1
        m15i=add_indicators(m15) if not m15.empty else m15
        m5i=add_indicators(m5) if not m5.empty else m5
        indicator_seconds += (time.time()-_t0)
        _tb=time.time(); _=float(h1["close"].iloc[-1] > h1["high"].tail(20).max()) if not h1.empty else 0.0; bos_seconds += (time.time()-_tb)
        _tr=time.time(); _=float(m15["close"].iloc[-1] > m15["high"].iloc[-2]) if len(m15) > 1 else 0.0; reclaim_seconds += (time.time()-_tr)
        _t0=time.time(); swing_sig=evaluate_swing_signal(w1,d1,h4,h1=h1i if not h1i.empty else None,m15=m15i if not m15i.empty else None,m5=m5i if not m5i.empty else None,features=features); signal_seconds += (time.time()-_t0)
        swing_cfg={"exchange":cfg['swing']['exchange'],"timeframe":cfg['swing']['timeframes']['execution'],"starting_capital":cfg['swing']['capital'],"risk_per_trade_pct":1.0,"default_stop_loss_pct":cfg['swing']['default_stop_loss_pct'],"default_take_profit_pct":cfg['swing']['take_profit_targets_pct'][0],"maker_fee_pct":cfg['swing']['maker_fee_pct'],"taker_fee_pct":cfg['swing']['taker_fee_pct'],"spread_pct":cfg['swing']['spread_pct']}
        swing_cfg["max_backtest_iterations_per_symbol"] = int(cfg["swing"].get("max_backtest_iterations_per_symbol", 500))
        logger.info("Swing signal %s %s score=%s tags=%s pullback=[%.4f, %.4f, %.4f]", symbol, swing_sig.signal, swing_sig.score, swing_sig.tags, swing_sig.pullback_30 or 0.0, swing_sig.pullback_50 or 0.0, swing_sig.pullback_618 or 0.0)
        if run_mode == "backtest":
            backtest_max_candles = int(cfg["swing"].get("backtest_max_candles", 500))
            h4_backtest = h4i.tail(backtest_max_candles).reset_index(drop=True)
            h1_backtest = h1i.tail(1500).reset_index(drop=True) if fast_backtest and not h1i.empty else (h1i if not h1i.empty else None)
            logger.info("BACKTEST mode: running historical backtest regardless of current signal")
            logger.info("Starting swing historical backtest for %s", symbol)
            _bt0 = time.time()
            swing_cfg["fast_backtest"] = fast_backtest
            t=run_backtest_for_symbol(swing_cfg,symbol,h4_backtest,str(journal_path),h1_df=h1_backtest,h4_df=h4_backtest)
            logger.info("Finished swing historical backtest for %s in %.2fs trades=%s", symbol, time.time()-_bt0, len(t))
        else:
            if swing_sig.signal == "HOLD" and bool(cfg["swing"].get("skip_backtest_on_hold", True)):
                logger.info("LIVE mode: skipping historical backtest if HOLD")
            t=[]
        swing_trades.extend(t)
        if enable_charts:
            _tc=time.time()
            plot_symbol_chart(h4i, f"{symbol}_SWING", charts/f"{symbol.replace('/','_')}_swing.png", trade=t[-1] if t else None, confidence_score=swing_sig.score, overlays={"regime": "swing_htf_zone", "score_breakdown": f"rev={swing_sig.reversal_probability:.2f}|cont={swing_sig.continuation_probability:.2f}|exh={swing_sig.exhaustion_probability:.2f}", "hold_reasons": swing_sig.reasons if swing_sig.signal=="HOLD" else []}, detailed_visuals=detailed_visuals)
            chart_seconds += (time.time()-_tc)

    investor_stats=compute_trade_stats(investor_trades)
    swing_stats=compute_trade_stats(swing_trades)
    combined_stats=compute_trade_stats(investor_trades+swing_trades)

    _t0=time.time()
    if enable_charts:
        plot_equity_curve(investor_trades, reports/"equity_curve_investor.png")
        plot_equity_curve(swing_trades, reports/"equity_curve_swing.png")
        plot_equity_curve(investor_trades+swing_trades, reports/"equity_curve_combined.png")

    _save_trades_csv(reports/"investor_trades_report.csv", investor_trades)
    _save_trades_csv(reports/"swing_trades_report.csv", swing_trades)
    _save_summary(reports/"investor_summary.json", reports/"investor_summary.csv", investor_stats, features)
    _save_summary(reports/"swing_summary.json", reports/"swing_summary.csv", swing_stats, features)
    _save_summary(reports/"combined_summary.json", reports/"combined_summary.csv", combined_stats, features)

    if enable_charts:
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
    reporting_seconds += (time.time()-_t0)
    logger.info("Timing summary: fetch=%.2fs indicators=%.2fs feature=%.2fs bos=%.2fs reclaim=%.2fs signal=%.2fs chart=%.2fs reporting=%.2fs", fetch_seconds, indicator_seconds, indicator_seconds, bos_seconds, reclaim_seconds, signal_seconds, chart_seconds, reporting_seconds)
