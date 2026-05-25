from pathlib import Path
import logging

import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd

from smc_navigator.simulator.trade import Trade

LOGGER = logging.getLogger(__name__)


def plot_equity_curve(trades: list[Trade], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    equity = []
    total = 0.0
    for trade in trades:
        total += float(trade.pnl or 0.0)
        equity.append(total)
    plt.figure(figsize=(10, 4)); plt.plot(range(1, len(equity) + 1), equity, marker="o"); plt.title("Equity Curve"); plt.xlabel("Trade #"); plt.ylabel("Cumulative PnL"); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(output_path); plt.close()


def _trade_marker_series(index: pd.Index, trade: Trade | None) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    long_marker = pd.Series(float("nan"), index=index); short_marker = pd.Series(float("nan"), index=index); sl_marker = pd.Series(float("nan"), index=index); tp_marker = pd.Series(float("nan"), index=index)
    if trade is None: return long_marker, short_marker, sl_marker, tp_marker
    ts = pd.Timestamp(trade.timestamp)
    if ts in index:
        if trade.direction == "LONG": long_marker.loc[ts] = trade.entry_price
        elif trade.direction == "SHORT": short_marker.loc[ts] = trade.entry_price
    sl_marker[:] = trade.stop_loss; tp_marker[:] = trade.take_profit
    return long_marker, short_marker, sl_marker, tp_marker


def _clean_chart_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    required = ["open", "high", "low", "close"]
    chart_df = df.copy().tail(120)
    if chart_df.empty: return chart_df
    if "timestamp" not in chart_df.columns:
        if isinstance(chart_df.index, pd.DatetimeIndex) or chart_df.index.name == "timestamp": chart_df = chart_df.reset_index().rename(columns={chart_df.columns[0]: "timestamp"})
        else: LOGGER.warning("Skipping chart generation for %s: missing timestamp column.", symbol); return pd.DataFrame()
    missing_required = [c for c in required if c not in chart_df.columns]
    if missing_required: LOGGER.warning("Skipping chart generation for %s: missing required OHLC columns %s.", symbol, missing_required); return pd.DataFrame()
    chart_df["timestamp"] = pd.to_datetime(chart_df["timestamp"], utc=True, errors="coerce")
    for col in ["open", "high", "low", "close", "volume", "ema_9", "ema_26", "ema_50", "vwap", "support", "resistance", "reversal_probability", "continuation_probability"]:
        if col in chart_df.columns: chart_df[col] = pd.to_numeric(chart_df[col], errors="coerce")
    chart_df = chart_df.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp").set_index("timestamp")
    return chart_df


def _is_valid_series(chart_df: pd.DataFrame, column: str) -> bool:
    if column not in chart_df.columns: return False
    return not pd.to_numeric(chart_df[column], errors="coerce").dropna().empty


def plot_symbol_chart(df: pd.DataFrame, symbol: str, output_path: str | Path, trade: Trade | None = None, confidence_score: int | None = None, overlays: dict | None = None, detailed_visuals: bool = False) -> None:
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    chart_df = _clean_chart_df(df, symbol)
    if chart_df.empty: LOGGER.warning("Skipping chart generation for %s: dataframe empty after cleaning.", symbol); return
    overlays = overlays or {}
    addplots = []
    for column, kwargs in [("ema_9", {"color": "dodgerblue", "width": 1}), ("ema_26", {"color": "orange", "width": 1}), ("ema_50", {"color": "purple", "width": 1}), ("vwap", {"color": "black", "width": 1}), ("support", {"color": "green", "linestyle": "--", "width": 0.8}), ("resistance", {"color": "red", "linestyle": "--", "width": 0.8})]:
        if _is_valid_series(chart_df, column): addplots.append(mpf.make_addplot(pd.to_numeric(chart_df[column], errors="coerce"), **kwargs))
    if detailed_visuals and _is_valid_series(chart_df, "reversal_probability"):
        addplots.append(mpf.make_addplot(chart_df["reversal_probability"], panel=1, color="red", ylabel="rev_prob"))
    if detailed_visuals and _is_valid_series(chart_df, "continuation_probability"):
        addplots.append(mpf.make_addplot(chart_df["continuation_probability"], panel=1, color="green", ylabel="cont_prob"))
    long_marker, short_marker, sl_marker, tp_marker = _trade_marker_series(chart_df.index, trade)
    if not long_marker.dropna().empty: addplots.append(mpf.make_addplot(long_marker, type="scatter", marker="^", markersize=100, color="lime"))
    if not short_marker.dropna().empty: addplots.append(mpf.make_addplot(short_marker, type="scatter", marker="v", markersize=100, color="crimson"))
    if not sl_marker.dropna().empty: addplots.append(mpf.make_addplot(pd.to_numeric(sl_marker, errors="coerce"), color="red", linestyle=":"))
    if not tp_marker.dropna().empty: addplots.append(mpf.make_addplot(pd.to_numeric(tp_marker, errors="coerce"), color="green", linestyle=":"))
    title = f"{symbol} | Predictive Visual"
    if confidence_score is not None: title += f" | Score:{confidence_score}"
    if overlays.get("regime"): title += f" | Regime:{overlays['regime']}"
    if overlays.get("hold_reasons"): title += " | HOLD because: " + ",".join(overlays["hold_reasons"][:3])
    if overlays.get("score_breakdown"): title += f" | breakdown:{overlays['score_breakdown']}"
    mpf.plot(chart_df, type="candle", style="yahoo", volume=True, addplot=addplots if addplots else None, title=title, figsize=(14, 8), savefig=dict(fname=str(output_path), dpi=120, bbox_inches="tight"))


def plot_yearly_equity_curve(trades: list[Trade], output_path: str | Path) -> None:
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    if not trades: return
    df = pd.DataFrame({"timestamp": [t.timestamp for t in trades], "pnl": [float(t.pnl or 0.0) for t in trades]}); df["year"] = pd.to_datetime(df["timestamp"]).dt.year.astype(str); grouped = df.groupby("year")["pnl"].sum().cumsum(); plt.figure(figsize=(10,4)); grouped.plot(marker='o'); plt.title("Yearly Equity Curve"); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(output_path); plt.close()


def plot_rolling_drawdown(trades: list[Trade], output_path: str | Path) -> None:
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    if not trades: return
    eq=[]; total=0.0
    for t in trades: total += float(t.pnl or 0.0); eq.append(total)
    s = pd.Series(eq); rolling_max = s.cummax(); dd = rolling_max - s; plt.figure(figsize=(10,4)); dd.plot(color='red'); plt.title('Rolling Drawdown'); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(output_path); plt.close()


def plot_regime_performance(perf_by_regime: dict[str, float], output_path: str | Path) -> None:
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    if not perf_by_regime: return
    plt.figure(figsize=(8,4)); plt.bar(list(perf_by_regime.keys()), list(perf_by_regime.values())); plt.title('Regime Performance'); plt.tight_layout(); plt.savefig(output_path); plt.close()


def plot_reversal_probability_heatmap(df: pd.DataFrame, output_path: str | Path, probability_col: str = "reversal_probability") -> None:
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty or probability_col not in df.columns: return
    hdf = df.copy().tail(200); hdf["timestamp"] = pd.to_datetime(hdf["timestamp"], errors="coerce"); hdf = hdf.dropna(subset=["timestamp", probability_col])
    if hdf.empty: return
    plt.figure(figsize=(12, 2.8)); plt.scatter(hdf["timestamp"], [1] * len(hdf), c=hdf[probability_col], cmap="RdYlGn", vmin=0, vmax=1, s=30); plt.colorbar(label="Reversal Probability"); plt.yticks([]); plt.title("Reversal Probability Heatmap"); plt.tight_layout(); plt.savefig(output_path); plt.close()
