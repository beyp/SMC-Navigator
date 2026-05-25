def calculate_sl_tp(entry_price: float, direction: str, sl_pct: float, tp_pct: float) -> tuple[float, float]:
    if direction == "LONG":
        sl = entry_price * (1 - sl_pct / 100)
        tp = entry_price * (1 + tp_pct / 100)
    elif direction == "SHORT":
        sl = entry_price * (1 + sl_pct / 100)
        tp = entry_price * (1 - tp_pct / 100)
    else:
        sl = entry_price
        tp = entry_price
    return sl, tp
