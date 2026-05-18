import pytest

from smc_navigator.core.config_validation import ensure_timeframes


def test_missing_investor_timing_raises() -> None:
    cfg = {
        "investor": {"timeframes": {"macro": "1M", "confirmation": "1w", "timing": None}},
        "swing": {"timeframes": {"context": "1w", "confirmation": "1d", "execution": "4h"}},
    }
    with pytest.raises(ValueError, match="investor.timeframes.timing"):
        ensure_timeframes(cfg)


def test_missing_swing_execution_raises() -> None:
    cfg = {
        "investor": {"timeframes": {"macro": "1M", "confirmation": "1w", "timing": "1d"}},
        "swing": {"timeframes": {"context": "1w", "confirmation": "1d", "execution": None}},
    }
    with pytest.raises(ValueError, match="swing.timeframes.execution"):
        ensure_timeframes(cfg)
