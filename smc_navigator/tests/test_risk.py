from smc_navigator.risk.position_sizing import calculate_position_size
from smc_navigator.risk.sl_tp import calculate_sl_tp


def test_position_sizing_normal_case() -> None:
    size, risk = calculate_position_size(100, 1, 100, 99)
    assert risk == 1
    assert size == 1


def test_position_sizing_invalid_sl() -> None:
    size, risk = calculate_position_size(100, 1, 100, 100)
    assert size == 0
    assert risk == 1


def test_sl_tp_long() -> None:
    sl, tp = calculate_sl_tp(100, "LONG", 1, 2)
    assert sl == 99
    assert tp == 102
