"""Closing-order validation in base quantity and quote value, never margin units."""
from decimal import Decimal
from math import isfinite


def plan_exit(
    exchange, pair, total, requested, price, *, min_notional_exempt=False, min_remaining_value=0.0
):
    """Return (quantity, fallback_reason); None quantity means even full close is invalid.

    Binance linear futures reduce-only exits may omit min-notional validation.
    Quantity/contract filters still apply, as does Freqtrade's local remainder floor.
    """
    for value in (total, price):
        if not isfinite(float(value)) or value <= 0:
            raise ValueError('退出数量和价格必须是有限正数')
    if not isfinite(float(requested)) or requested < 0:
        raise ValueError('请求退出数量无效')
    if not isfinite(float(min_remaining_value)) or min_remaining_value < 0:
        raise ValueError('剩余仓位门槛无效')
    market = exchange.markets.get(pair)
    if not market:
        raise ValueError('无法读取平仓交易规则')

    def dec(value):
        return Decimal(str(value))

    total, price = dec(total), dec(price)
    size = dec(market.get('contractSize') or 1) if market.get('contract') else Decimal(1)
    if size <= 0 or not size.is_finite():
        raise ValueError('合约面值无效')
    limits = market.get('limits', {})
    amount_limits, cost_limits = limits.get('amount', {}), limits.get('cost', {})
    filters = market.get('info', {}).get('filters', [])
    steps = [
        dec(rule.get('stepSize', 0))
        for rule in filters
        if rule.get('filterType') in ('LOT_SIZE', 'MARKET_LOT_SIZE')
        and dec(rule.get('stepSize', 0)) > 0
    ]
    tolerance = Decimal('1e-12') * max(total, Decimal(1))
    if steps:
        tolerance = min(tolerance, min(steps) * size * Decimal('1e-9'))
    contract_tolerance = tolerance / size

    def invalid(quantity, *, check_max=True):
        if quantity <= 0:
            return 'quantity_zero'
        contracts = quantity / size
        minimum = amount_limits.get('min')
        maximum = amount_limits.get('max')
        if minimum is not None and contracts + contract_tolerance < dec(minimum):
            return 'quantity_minimum'
        if check_max and maximum is not None and contracts > dec(maximum) + contract_tolerance:
            return 'quantity_maximum'
        minimum_costs = [dec(cost_limits['min'])] if cost_limits.get('min') is not None else []
        for rule in filters:
            kind = rule.get('filterType')
            if kind in ('LOT_SIZE', 'MARKET_LOT_SIZE'):
                if contracts + contract_tolerance < dec(rule.get('minQty', 0)):
                    return 'quantity_minimum'
                if (
                    check_max
                    and dec(rule.get('maxQty', 0)) > 0
                    and contracts > dec(rule['maxQty']) + contract_tolerance
                ):
                    return 'quantity_maximum'
                step = dec(rule.get('stepSize', 0))
                if step > 0:
                    remainder = contracts % step
                    if min(remainder, step - remainder) > contract_tolerance:
                        return 'quantity_step'
            elif kind == 'MIN_NOTIONAL':
                minimum_costs.append(dec(rule.get('notional', rule.get('minNotional', 0))))
        if (
            not min_notional_exempt
            and minimum_costs
            and (quantity + tolerance) * price < max(minimum_costs)
        ):
            return 'notional_minimum'
        return None

    quantity = dec(exchange.amount_to_contract_precision(pair, float(min(dec(requested), total))))
    if abs(quantity - total) <= tolerance:
        quantity = total
    reason = invalid(quantity)
    remaining = total - quantity
    if quantity > total + tolerance:
        raise ValueError('退出数量超过真实剩余仓位')
    if not reason and remaining > tolerance:
        reason = invalid(remaining, check_max=False)
        if reason:
            reason = 'remainder_' + reason
        elif (remaining + tolerance) * price < dec(min_remaining_value):
            reason = 'remainder_framework_minimum'
    if not reason:
        return float(quantity), None
    full = dec(exchange.amount_to_contract_precision(pair, float(total)))
    full_error = invalid(full)
    if full_error or abs(full - total) > tolerance:
        return None, 'full_' + (full_error or 'quantity_step')
    return float(total), reason
