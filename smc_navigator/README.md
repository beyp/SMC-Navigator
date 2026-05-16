# SMC Navigator

SMC Navigator is a **simulation-first crypto trading decision tool**.

> ⚠️ Version 1 is strictly simulation-only. It fetches public OHLCV market data, evaluates strategy rules, computes risk/SL/TP, and logs simulated trades. It **does not place real orders**.

## Project goals
- Modular architecture with strict separation of concerns
- No private API keys
- No live order execution
- Human validation before any future execution layer

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration
Edit `config.yaml`:
- exchange (default: `kraken`)
- symbols
- timeframe
- starting capital
- risk settings
- safety flags (`simulation_mode: true`, `allow_real_orders: false`)

## Run
```bash
python main.py
```

The CLI will:
1. Load config
2. Fetch candles from public exchange endpoints
3. Compute indicators (EMA, RSI, VWAP, support/resistance placeholders)
4. Generate strategy signals
5. Simulate trades
6. Append results to `data/trade_journal.csv`

## Testing
```bash
pytest -q
```

## Safety constraints
- `allow_real_orders` must remain `false`
- no private keys required
- no order placement functions are implemented in version 1

## Next evolutions
- Paper trading dashboard
- Historical backtesting module
- Telegram/Discord alert integrations
- Human validation mode before execution
- Future execution engine with strict safeguards and explicit opt-in


## Reporting
- Saves charts to `reports/`:
  - `equity_curve.png`
  - `latest_symbol_candles.png`
- Prints trade statistics in CLI:
  - total trades
  - wins
  - losses
  - winrate
  - total pnl
  - average pnl
  - max drawdown
