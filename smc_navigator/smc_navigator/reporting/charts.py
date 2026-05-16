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

    plt.figure(figsize=(10, 4))
    plt.plot(range(1, len(equity) + 1), equity, marker="o")
    plt.title("Equity Curve")
    plt.xlabel("Trade #")
    plt.ylabel("Cumulative PnL")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _trade_marker_series(index: pd.Index, trade: Trade | None) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    long_marker = pd.Series(float("nan"), index=index)
    short_marker = pd.Series(float("nan"), index=index)
    sl_marker = pd.Series(float("nan"), index=index)
    tp_marker = pd.Series(float("nan"), index=index)

    if trade is None:
        return long_marker, short_marker, sl_marker, tp_marker

    ts = pd.Timestamp(trade.timestamp)
    if ts in index:
        if trade.direction == "LONG":
            long_marker.loc[ts] = trade.entry_price
        elif trade.direction == "SHORT":
            short_marker.loc[ts] = trade.entry_price

    sl_marker[:] = trade.stop_loss
    tp_marker[:] = trade.take_profit

    return long_marker, short_marker, sl_marker, tp_marker


def _is_valid_series(chart_df: pd.DataFrame, column: str) -> bool:
    if column not in chart_df.columns:
        LOGGER.warning("Skipping plot column '%s': column not found.", column)
        return False
    series = chart_df[column]
    if series.dropna().empty:
        LOGGER.warning("Skipping plot column '%s': series has no non-NaN values.", column)
        return False
    return True


def plot_symbol_chart(
    df: pd.DataFrame,
    symbol: str,
    output_path: str | Path,
    trade: Trade | None = None,
    confidence_score: int | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chart_df = df.copy().tail(120)
    if chart_df.empty:
        LOGGER.warning("Skipping chart generation for %s: chart dataframe is empty.", symbol)
        return

    chart_df = chart_df.set_index("timestamp")
    chart_df.index = pd.DatetimeIndex(chart_df.index)

    addplots = []
    indicator_specs = [
        ("ema_9", {"color": "dodgerblue", "width": 1}),
        ("ema_26", {"color": "orange", "width": 1}),
        ("ema_50", {"color": "purple", "width": 1}),
        ("vwap", {"color": "black", "width": 1}),
        ("support", {"color": "green", "linestyle": "--", "width": 0.8}),
        ("resistance", {"color": "red", "linestyle": "--", "width": 0.8}),
    ]

    for column, kwargs in indicator_specs:
        if _is_valid_series(chart_df, column):
            addplots.append(mpf.make_addplot(chart_df[column], **kwargs))

    long_marker, short_marker, sl_marker, tp_marker = _trade_marker_series(chart_df.index, trade)
    if not long_marker.dropna().empty:
        addplots.append(mpf.make_addplot(long_marker, type="scatter", marker="^", markersize=100, color="lime"))
    if not short_marker.dropna().empty:
        addplots.append(mpf.make_addplot(short_marker, type="scatter", marker="v", markersize=100, color="crimson"))
    if not sl_marker.dropna().empty:
        addplots.append(mpf.make_addplot(sl_marker, color="red", linestyle=":"))
    if not tp_marker.dropna().empty:
        addplots.append(mpf.make_addplot(tp_marker, color="green", linestyle=":"))

    title = f"{symbol} | EMA9/26/50 + VWAP + S/R"
    if confidence_score is not None:
        title += f" | Confidence: {confidence_score}"

    mpf.plot(
        chart_df,
        type="candle",
        style="yahoo",
        volume=True,
        addplot=addplots if addplots else None,
        title=title,
        figsize=(14, 8),
        savefig=dict(fname=str(output_path), dpi=120, bbox_inches="tight"),
    )
