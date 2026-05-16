from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from smc_navigator.simulator.trade import Trade


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


def plot_latest_symbol_candles(
    df: pd.DataFrame,
    symbol: str,
    output_path: str | Path,
    entry: float,
    stop_loss: float,
    take_profit: float,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    recent = df.tail(60).reset_index(drop=True)
    x = range(len(recent))

    plt.figure(figsize=(12, 5))
    ax = plt.gca()

    for i, row in recent.iterrows():
        color = "green" if row["close"] >= row["open"] else "red"
        ax.plot([i, i], [row["low"], row["high"]], color=color, linewidth=1)
        body_low = min(row["open"], row["close"])
        body_height = max(abs(row["close"] - row["open"]), 0.0001)
        ax.add_patch(plt.Rectangle((i - 0.3, body_low), 0.6, body_height, color=color, alpha=0.7))

    ax.axhline(entry, color="blue", linestyle="--", label=f"Entry {entry:.4f}")
    ax.axhline(stop_loss, color="red", linestyle="--", label=f"SL {stop_loss:.4f}")
    ax.axhline(take_profit, color="green", linestyle="--", label=f"TP {take_profit:.4f}")
    ax.set_title(f"{symbol} Candlestick (latest)")
    ax.set_xlabel("Candle Index")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
