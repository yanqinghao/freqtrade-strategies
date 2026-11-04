"""Read actual tracked stop receipts; never create or cancel orders here."""
import logging
from datetime import timezone
from decimal import Decimal, InvalidOperation
from math import isfinite, ulp

from freqtrade.exchange import ROUND_DOWN, ROUND_UP

logger = logging.getLogger(__name__)


def _number(value, *, positive=False):
    if isinstance(value, bool):
        raise ValueError('无效数值')
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise ValueError('无效数值') from None
    if (
        not result.is_finite()
        or not isfinite(float(result))
        or result < 0
        or (positive and result == 0)
    ):
        raise ValueError('无效数值')
    return result


def _flag(row, name):
    info = row.get('info', {})
    if not isinstance(info, dict):
        raise ValueError('止损回执字段不完整')
    values = []
    for source in (row, info):
        value = source.get(name)
        if value is None:
            continue
        if isinstance(value, bool):
            parsed = value
        elif isinstance(value, int) and value in (0, 1):
            parsed = bool(value)
        elif isinstance(value, str) and value.lower() in ('true', 'false'):
            parsed = value.lower() == 'true'
        else:
            raise ValueError('止损平仓标志无效')
        values.append(parsed)
    if len(set(values)) > 1:
        raise ValueError('止损平仓标志不一致')
    return values[0] if values else False


def _noise(exchange, pair, first, second, kind, field, scale=Decimal(1)):
    noise = Decimal(str(4 * max(ulp(float(first)), ulp(float(second)))))
    steps = [
        _number(rule.get(field, 0))
        for rule in exchange.markets.get(pair, {}).get('info', {}).get('filters', [])
        if rule.get('filterType') == kind
    ]
    steps = [step for step in steps if step > 0]
    return min(noise, min(steps) * scale / 1000) if steps else Decimal(0)


def _evaluate(exchange, trade, ident, row, target):
    if (
        not isinstance(row, dict)
        or str(row.get('id')) != str(ident)
        or row.get('symbol') != trade.pair
    ):
        raise ValueError('止损回执身份不一致')
    if row.get('status') != 'open' or row.get('status_stop') == 'triggered':
        raise ValueError('止损已触发或不再活跃，等待原生退出核实')
    if row.get('side') != trade.exit_side:
        raise ValueError('止损方向不匹配')
    info = row.get('info', {})
    if not isinstance(info, dict) or info.get('positionSide') not in (None, 'BOTH'):
        raise ValueError('止损持仓模式不匹配')
    if row.get('type') not in (
        'stop_market',
        'stop_loss',
        'stop',
        'stop_loss_limit',
        'stoploss',
        'market',
        'limit',
    ):
        raise ValueError('止损类型不匹配')
    if any(
        'TAKE_PROFIT' in str(info.get(key, '')).upper() for key in ('origType', 'orderType', 'type')
    ):
        raise ValueError('订单不是保护止损')
    close_all, reduce_only = _flag(row, 'closePosition'), _flag(row, 'reduceOnly')
    if not close_all and not reduce_only:
        raise ValueError('止损不是只减仓或整仓关闭')
    if _number(row.get('filled', 0)) > 0:
        raise ValueError('止损已开始成交，等待原生退出核实')
    trigger = row.get('stopPrice') if row.get('stopPrice') is not None else row.get('triggerPrice')
    actual_price, required = _number(trigger, positive=True), _number(trade.amount, positive=True)
    tolerance = _noise(exchange, trade.pair, actual_price, target, 'PRICE_FILTER', 'tickSize')
    if (
        (actual_price > target + tolerance)
        if trade.is_short
        else (actual_price < target - tolerance)
    ):
        raise ValueError('实际止损价尚未达到当前保护目标')
    if close_all:
        covered = required
    else:
        amount, filled = _number(row.get('amount')), _number(row.get('filled'))
        if filled > amount:
            raise ValueError('止损覆盖数据无效')
        covered = (
            _number(row.get('remaining')) if row.get('remaining') is not None else amount - filled
        )
        size = _number(exchange.markets[trade.pair].get('contractSize', 1), positive=True)
        consistency = _noise(
            exchange, trade.pair, covered, amount - filled, 'LOT_SIZE', 'stepSize', size
        )
        if abs(covered - (amount - filled)) > consistency:
            raise ValueError('止损总量、成交量与剩余量不一致')
        tolerance = _noise(exchange, trade.pair, covered, required, 'LOT_SIZE', 'stepSize', size)
        if covered + tolerance < required:
            raise ValueError('止损覆盖数量不足')
    return {
        'status': 'confirmed',
        'order_id': str(ident),
        'stop_price': float(actual_price),
        'covered_amount': float(covered),
        'reason': '交易所活跃止损已核实',
    }


def check_protection(strategy, trade, now, *, force=False):
    if not trade.is_open or not trade.amount:
        return False
    mode = strategy.config.get('runmode')
    simulated = strategy.config.get('dry_run', getattr(mode, 'value', mode) == 'dry_run')
    if simulated:
        state = strategy._execution(trade)
        state['protection'] = {'status': 'simulated', 'checked_at': now.timestamp()}
        strategy._save_execution(trade, state)
        return True
    exchange = strategy.dp._exchange
    state = strategy._execution(trade)
    old = state.get('protection') or {}
    stamp = now.timestamp()
    target = None
    try:
        plan = strategy.manual_open.get(trade.pair) or {}
        direction = 'short' if trade.is_short else 'long'
        if plan.get('trade_id') not in (None, trade.id) or plan.get('direction') != direction:
            plan = {}
        plan = plan or trade.get_custom_data('assistant_plan', default={}) or {}
        if not plan or plan.get('direction') != direction:
            raise ValueError('没有可核对的止损计划')
        target = _number(
            exchange.price_to_precision(
                trade.pair,
                strategy._stop_price(trade, plan),
                rounding_mode=ROUND_DOWN if trade.is_short else ROUND_UP,
            ),
            positive=True,
        )
        signature = [float(target), float(trade.amount)]
        interval = 30 if old.get('status') == 'confirmed' else 15
        if (
            not force
            and old.get('target_signature') == signature
            and stamp - old.get('checked_at', 0) < interval
        ):
            return old.get('status') == 'confirmed'
        result = {'status': 'unconfirmed', 'reason': '未找到活跃的已跟踪止损单'}
        for order in trade.open_sl_orders:
            try:
                raw = exchange.fetch_stoploss_order(order.order_id, trade.pair)
            except Exception as error:
                logger.warning('Stop verification unavailable (%s)', type(error).__name__)
                result = {'status': 'unconfirmed', 'reason': '交易所止损状态暂未查询确认'}
                continue
            try:
                result = _evaluate(exchange, trade, order.order_id, raw, target)
                break
            except ValueError as error:
                result = {'status': 'unconfirmed', 'reason': str(error)}
            except Exception as error:
                logger.warning('Stop verification unavailable (%s)', type(error).__name__)
                result = {'status': 'unconfirmed', 'reason': '交易所止损状态暂未查询确认'}
    except Exception as error:
        logger.warning('Stop target unavailable (%s)', type(error).__name__)
        signature = [None, float(trade.amount)]
        result = {'status': 'unconfirmed', 'reason': '当前保护目标或市场数据无效'}
    # _stop_price may tighten/persist state: never overwrite that or fill accounting.
    state = strategy._execution(trade)
    previous = state.get('protection') or {}
    since = previous.get('pending_since')
    if since is None:
        opened = getattr(trade, 'open_date_utc', None) or getattr(trade, 'open_date', now)
        since = opened.replace(tzinfo=timezone.utc).timestamp()
    if previous.get('status') == 'confirmed' and previous.get('target_signature') != signature:
        since = stamp
    if result['status'] != 'confirmed' and stamp - since < 15:
        result['status'] = 'pending'
    result.update(checked_at=stamp, pending_since=since, target_signature=signature)
    key = [
        result['status'],
        result['reason'],
        result.get('order_id'),
        result.get('stop_price'),
        result.get('covered_amount'),
        signature,
    ]
    old_key = previous.get('notified_key') or []
    notified = previous.get('notified_status') or (old_key[0] if old_key else None)
    notify = result['status'] != 'pending' and (
        result['status'] != notified
        or previous.get('status') == 'unconfirmed'
        and result['status'] == 'confirmed'
    )
    result['notified_status'] = notified
    result['notified_key'] = previous.get('notified_key')
    state['protection'] = result
    strategy._save_execution(trade, state)
    if notify:
        queued = strategy._exit_notice(
            f"【保护正常】{trade.pair}：止损已确认，触发价 {result['stop_price']:g}，"
            f"覆盖 {result['covered_amount']:g}；"
            f"核验时间 {now:%Y-%m-%d %H:%M:%S} UTC。"
            if result['status'] == 'confirmed'
            else f"【需要处理】{trade.pair}：止损尚未确认（{result['reason']}）。\n"
            '下一步：用 /order 与交易所核对止损单及覆盖数量；已暂停加仓，平仓照常。',
            key=f'protection:{trade.id}',
        )
        if queued:
            result['notified_status'] = result['status']
            result['notified_key'] = key
            state['protection'] = result
            strategy._save_execution(trade, state)
    return result['status'] == 'confirmed'
