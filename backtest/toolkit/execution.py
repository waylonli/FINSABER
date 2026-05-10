import math


def calculate_commission(quantity, price, commission_per_share=0.0049, min_commission=0.99, max_commission_rate=0.01):
    quantity = abs(int(quantity))
    if quantity <= 0 or price <= 0:
        return 0.0
    commission = quantity * commission_per_share
    transaction_amount = quantity * price
    return min(max(commission, min_commission), transaction_amount * max_commission_rate)


def adjusted_bar_price(bar, field):
    if not isinstance(bar, dict):
        return bar
    if field in bar:
        return bar[field]
    if field.startswith("adjusted_") and "adjusted_close" in bar and "close" in bar:
        raw_field = field.removeprefix("adjusted_")
        if raw_field in bar and bar["close"] != 0:
            return bar[raw_field] * (bar["adjusted_close"] / bar["close"])
    if field.endswith("close"):
        return bar.get("adjusted_close", bar.get("close"))
    return bar.get("close", bar.get("adjusted_close"))


def prior_volume_stats(data_loader, ticker, date, lookback_days):
    dates = [d for d in data_loader.get_date_range() if d < date]
    dates = dates[-lookback_days:]
    volumes = []
    for prior_date in dates:
        bar = data_loader.get_data_by_date(prior_date).get("price", {}).get(ticker)
        if isinstance(bar, dict) and bar.get("volume") is not None:
            volumes.append(bar["volume"])
    if not volumes:
        return {"average_volume": None, "observations": 0}
    return {"average_volume": sum(volumes) / len(volumes), "observations": len(volumes)}


def average_prior_volume(data_loader, ticker, date, lookback_days):
    return prior_volume_stats(data_loader, ticker, date, lookback_days)["average_volume"]


def apply_liquidity_cap(quantity, average_volume, cap_pct, require_volume=False):
    quantity = abs(int(quantity))
    if cap_pct <= 0:
        return quantity
    if average_volume is None:
        return 0 if require_volume else quantity
    cap = int(math.floor(average_volume * cap_pct))
    return max(0, min(quantity, cap))


def apply_slippage(price, side, quantity, average_volume=None, slippage_perc=0.0, slippage_impact=0.0):
    if price <= 0:
        return price, 0.0
    participation = 0.0
    if average_volume and average_volume > 0:
        participation = abs(quantity) / average_volume
    slippage_rate = slippage_perc + slippage_impact * (participation ** 2)
    signed_rate = slippage_rate if side == "buy" else -slippage_rate
    fill_price = price * (1 + signed_rate)
    slippage_cost = abs(fill_price - price) * abs(quantity)
    return fill_price, slippage_cost
