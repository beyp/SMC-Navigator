import pytest

from smc_navigator.core.config_validation import ensure_timeframes


def _base_cfg() -> dict:
    return {
        "investor": {
            "exchange": "kraken",
            "symbols": ["BTC/EUR"],
            "maker_fee_pct": 0.16,
            "taker_fee_pct": 0.26,
            "timeframes": {"macro": "1M", "confirmation": "1w", "timing": "1d"},
        },
        "swing": {
            "exchange": "binance",
            "symbols": ["BTC/USDT"],
            "maker_fee_pct": 0.1,
            "taker_fee_pct": 0.1,
            "timeframes": {"context": "1w", "confirmation": "1d", "execution": "4h"},
        },
    }


def test_mixed_exchange_configuration_passes() -> None:
    cfg = _base_cfg()
    out = ensure_timeframes(cfg)
    assert out["investor"]["exchange"] == "kraken"
    assert out["swing"]["exchange"] == "binance"


def test_inverse_mixed_exchange_configuration_passes() -> None:
    cfg = _base_cfg()
    cfg["investor"]["exchange"] = "binance"
    cfg["swing"]["exchange"] = "kraken"
    out = ensure_timeframes(cfg)
    assert out["investor"]["exchange"] == "binance"
    assert out["swing"]["exchange"] == "kraken"


def test_missing_investor_macro_raises() -> None:
    cfg = _base_cfg()
    cfg["investor"]["timeframes"]["macro"] = None
    with pytest.raises(ValueError, match="Missing config: investor.timeframes.macro"):
        ensure_timeframes(cfg)
