from smc_navigator.dashboard.cli import _build_exchange
from smc_navigator.exchanges.binance import BinanceExchange
from smc_navigator.exchanges.kraken import KrakenExchange


def test_build_exchange_investor_kraken_swing_binance() -> None:
    investor_exchange = _build_exchange("kraken")
    swing_exchange = _build_exchange("binance")
    assert isinstance(investor_exchange, KrakenExchange)
    assert isinstance(swing_exchange, BinanceExchange)


def test_build_exchange_investor_binance_swing_kraken() -> None:
    investor_exchange = _build_exchange("binance")
    swing_exchange = _build_exchange("kraken")
    assert isinstance(investor_exchange, BinanceExchange)
    assert isinstance(swing_exchange, KrakenExchange)
