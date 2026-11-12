"""Initial-position exit targets; amounts are USDT, percentages are underlying moves."""
from copy import deepcopy
from decimal import Decimal, InvalidOperation

UNIT_FIELDS = ('loss', 'sl_pct', 'profit', 'tp_pct')


class TargetError(ValueError):
    """Controlled target validation failure, safe to show to the user."""


def _number(value, *, zero=False):
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        raise TargetError('目标参数必须是有限正数') from None
    if isinstance(value, bool) or not result.is_finite() or (result < 0 if zero else result <= 0):
        raise TargetError('目标参数必须是有限正数')
    return result


def has_units(plan):
    return any(key in plan for key in UNIT_FIELDS)


def validate_units(plan):
    if 'loss' in plan and 'sl_pct' in plan or 'profit' in plan and 'tp_pct' in plan:
        raise TargetError('同一SL或TP只能使用一种口径')
    for key in ('loss', 'sl_pct'):
        if key in plan:
            _number(plan[key])
    for key in ('profit', 'tp_pct'):
        if key in plan:
            values = [_number(value) for value in plan[key]]
            if not 1 <= len(values) <= 3 or values != sorted(set(values)):
                raise TargetError('金额/幅度TP须为一至三个严格递增且不同的正数')


def convert_targets(plan, rate, quantity, fee_open, fee_close, round_price):
    """Resolve raw units into native prices, using full initial Q and conservative ticks."""
    validate_units(plan)
    result = deepcopy(plan)
    rate, quantity = _number(rate), _number(quantity)
    opened, closed = _number(fee_open, zero=True), _number(fee_close, zero=True)
    if opened >= 1 or closed >= 1:
        raise TargetError('手续费率无效，无法换算目标')
    short = plan['direction'] == 'short'
    cost = quantity * rate * (1 - opened if short else 1 + opened)

    def cash(value):
        return (
            (cost - value) / (quantity * (1 + closed))
            if short
            else (cost + value) / (quantity * (1 - closed))
        )

    def rounded(price):
        return float(_number(round_price(_number(price), short)))

    if 'loss' in plan:
        result['stop_loss'] = rounded(cash(-_number(plan['loss'])))
    elif 'sl_pct' in plan:
        fraction = _number(plan['sl_pct']) / 100
        result['stop_loss'] = rounded(rate * (1 + fraction if short else 1 - fraction))
    stop = _number(result.get('stop_loss'))
    if stop <= rate if short else stop >= rate:
        raise TargetError('止损预算不足以覆盖费用或价格精度，换算SL须位于亏损方向')
    if 'profit' in plan:
        result['exit_points'] = [rounded(cash(_number(value))) for value in plan['profit']]
    elif 'tp_pct' in plan:
        result['exit_points'] = [
            rounded(rate * (1 - _number(value) / 100 if short else 1 + _number(value) / 100))
            for value in plan['tp_pct']
        ]
    targets = [_number(value) for value in result.get('exit_points', [])]
    if targets != sorted(set(targets), reverse=short):
        raise TargetError('目标换算后按交易所精度重合或顺序无效，请调整TP')
    if any(value >= rate if short else value <= rate for value in targets):
        raise TargetError('换算TP须位于成交参考价的盈利方向')
    result['target_basis'] = dict(
        rate=float(rate),
        quantity=float(quantity),
        fee_open=float(opened),
        fee_close=float(closed),
        anchored=False,
        fee_source='入场及平仓费率估算，不含未来资金费和滑点',
    )
    return result


def clear_units(plan, *, stop=False, tp=False):
    for key in (('loss', 'sl_pct') if stop else ()) + (('profit', 'tp_pct') if tp else ()):
        plan.pop(key, None)
    if not has_units(plan):
        plan.pop('target_basis', None)
    return plan


def describe_targets(plan):
    parts = []
    if 'loss' in plan:
        parts.append(f"入场SL预算约 {plan['loss']:g} USDT")
    if 'sl_pct' in plan:
        parts.append(f"入场SL：币价逆向 {plan['sl_pct']:g}%")
    if 'profit' in plan:
        parts.append('TP：原始完整仓位净浮盈 ' + '/'.join(f'{v:g}' for v in plan['profit']) + ' USDT')
    if 'tp_pct' in plan:
        parts.append('TP：币价顺向 ' + '/'.join(f'{v:g}' for v in plan['tp_pct']) + '%')
    if parts:
        basis = plan.get('target_basis', {})
        parts.append('目标已按初始成交冻结' if basis.get('anchored') else '参考价预估，待初始成交校准')
    return '；'.join(parts)


def reconcile_basis(current, saved):
    """An anchored copy repairs a provisional peer; cleared manual units never revive."""
    result = deepcopy(current)
    if not has_units(result):
        return result
    basis, previous = result.get('target_basis', {}), saved.get('target_basis', {})
    if not previous.get('anchored'):
        return result
    active = [key for key in UNIT_FIELDS if key in result]
    if any(result[key] != saved.get(key) for key in active):
        raise TargetError('已冻结目标与当前金额/幅度规格冲突')
    if basis.get('request_ref') != previous.get('request_ref'):
        raise TargetError('目标成交基准的请求身份冲突')
    if basis.get('anchored'):
        fields = [] if not any(k in result for k in ('loss', 'sl_pct')) else ['stop_loss']
        fields += [] if not any(k in result for k in ('profit', 'tp_pct')) else ['exit_points']
        if basis != previous or any(result.get(key) != saved.get(key) for key in fields):
            raise TargetError('已冻结的成交目标记录不一致')
        return result
    for key, units in (('stop_loss', ('loss', 'sl_pct')), ('exit_points', ('profit', 'tp_pct'))):
        if any(unit in result for unit in units):
            result[key] = deepcopy(saved[key])
    result['target_basis'] = deepcopy(previous)
    return result
