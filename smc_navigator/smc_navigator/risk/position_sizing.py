def calculate_position_size(capital: float, risk_per_trade_pct: float, entry_price: float, stop_loss_price: float) -> tuple[float, float]:
    if capital <= 0 or risk_per_trade_pct <= 0 or entry_price <= 0:
        return 0.0, 0.0

    risk_amount = capital * (risk_per_trade_pct / 100)
    risk_per_unit = abs(entry_price - stop_loss_price)

    if risk_per_unit <= 0:
        return 0.0, risk_amount

    position_size = risk_amount / risk_per_unit
    return max(position_size, 0.0), risk_amount
