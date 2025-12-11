def calculate_costs(trade_value, commission_pct=0.001, slippage_pct=0.0005):
    commission = abs(trade_value) * commission_pct
    slippage = abs(trade_value) * slippage_pct
    return commission + slippage


# TODO: add more realistic slippage model based on volume
