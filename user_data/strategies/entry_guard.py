"""Fresh exchange-position validation for explicit risk-increasing commands only."""
from decimal import Decimal, InvalidOperation
from math import isfinite, ulp


def _number(value, *, positive=False):
    if isinstance(value, bool):
        raise ValueError('仓位数据包含无效数值')
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        raise ValueError('仓位数据包含无效数值') from None
    if not number.is_finite() or number < 0 or (positive and number == 0):
        raise ValueError('仓位数据包含无效数值')
    return number


def ensure_position_matches(exchange, pair, trade, *, dry_run=False):
    if dry_run:
        return
    try:
        positions = exchange.fetch_positions(pair)
    except Exception:
        raise ValueError('无法核对交易所仓位，未提交订单；请检查连接后重试') from None
    if not isinstance(positions, list):
        raise ValueError('交易所仓位响应无效，未提交订单')
    active = []
    for position in positions:
        if not isinstance(position, dict) or not isinstance(position.get('symbol'), str):
            raise ValueError('交易所仓位数据不完整，未提交订单')
        if position['symbol'] != pair:
            continue
        contracts = _number(position.get('contracts'))
        if contracts == 0:
            continue
        if (
            position.get('side') not in ('long', 'short')
            or position.get('hedged') is not False
            or (position.get('info') or {}).get('positionSide') in ('LONG', 'SHORT')
        ):
            raise ValueError('交易所仓位方向/持仓模式异常，未提交订单')
        active.append((position, contracts))
    if len(active) > 1:
        raise ValueError('交易所存在多个持仓方向/重复记录，未提交订单')
    if trade is None:
        if active:
            raise ValueError(f'{pair} 存在未接管仓位，请先核对交易所，不会自动并仓')
        return
    expected = _number(trade.amount, positive=True)
    if not trade.is_open:
        raise ValueError('本地持仓记录异常，未提交订单')
    if not active:
        raise ValueError('交易所为空仓但本地仍有数量，请先核对 /status、/order 与交易所')
    position, contracts = active[0]
    if (position['side'] == 'short') != trade.is_short:
        raise ValueError('交易所与本地持仓方向不一致，未提交订单')
    market = exchange.markets.get(pair, {})
    if not market.get('contract'):
        raise ValueError('无法确定合约数量单位，未提交订单')
    size = _number(market.get('contractSize'), positive=True)
    if (
        position.get('contractSize') is not None
        and _number(position['contractSize'], positive=True) != size
    ):
        raise ValueError('交易所合约面值与本地市场数据不一致，未提交订单')
    actual = contracts * size
    if not isfinite(float(actual)) or not isfinite(float(expected)):
        raise ValueError('仓位数量超出有效范围，未提交订单')
    steps = [
        _number(rule.get('stepSize', 0))
        for rule in market.get('info', {}).get('filters', [])
        if rule.get('filterType') in ('LOT_SIZE', 'MARKET_LOT_SIZE')
    ]
    steps = [step for step in steps if step > 0]
    rounding_noise = Decimal(str(4 * max(ulp(float(actual)), ulp(float(expected)))))
    tolerance = min(rounding_noise, min(steps, default=Decimal(1)) * size * Decimal('.001'))
    if abs(actual - expected) > tolerance:
        raise ValueError('交易所与本地持仓数量不一致，未提交订单；请核对 /status、/order 与交易所')
