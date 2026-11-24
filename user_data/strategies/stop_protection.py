"""Refresh isolated risk and verify live stop orders without creating or cancelling orders."""
import logging
from datetime import timezone
from decimal import Decimal, InvalidOperation
from math import isfinite, ulp

from freqtrade.exchange import ROUND_DOWN, ROUND_UP
from entry_guard import ensure_position_matches
from entry_risk import manual_stop_active
from notification_format import notice

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


def refresh_isolated_boundary(strategy, trade):
    """Refresh the native, buffered boundary from a matching live position."""
    exchange = strategy.dp._exchange
    position = ensure_position_matches(exchange, trade.pair, trade)
    if position.get('marginMode') != 'isolated':
        raise ValueError('交易所逐仓模式无法核实')
    raw = _number(position.get('liquidationPrice'), positive=True)
    buffer = _number(exchange.liquidation_buffer)
    if buffer > 1:
        raise ValueError('强平缓冲比例无效')
    offset = abs(raw - _number(trade.open_rate, positive=True)) * buffer
    boundary = raw - offset if trade.is_short else raw + offset
    price = exchange.price_to_precision(
        trade.pair, float(boundary), rounding_mode=ROUND_DOWN if trade.is_short else ROUND_UP
    )
    trade.set_liquidation_price(float(_number(price, positive=True)))
    return float(raw)


def listed_stops(exchange, trade):
    """Read Binance open conditional exits, including stops created by other clients."""
    try:
        rows = exchange._api.fetch_open_orders(trade.pair, params={'trigger': True})
    except Exception:
        raise ValueError('交易所条件单列表暂未核实') from None
    if not isinstance(rows, list):
        raise ValueError('交易所条件单列表无效')
    stops = {}
    for row in rows:
        if not isinstance(row, dict) or row.get('symbol') != trade.pair:
            raise ValueError('交易所条件单身份无法核实')
        if row.get('side') not in ('buy', 'sell') or row.get('status') != 'open':
            raise ValueError('交易所条件单状态无法核实')
        if row['side'] != trade.exit_side:
            continue
        info = row.get('info') or {}
        kind = str(info.get('orderType') or info.get('origType') or row.get('type')).upper()
        if kind.startswith('TAKE_PROFIT'):
            continue
        if not (_flag(row, 'reduceOnly') or _flag(row, 'closePosition')):
            continue
        ident = str(row.get('id') or '')
        if not ident or ident in stops:
            raise ValueError('交易所止损编号无法核实')
        stops[ident] = float(
            _number(row.get('stopPrice') or row.get('triggerPrice'), positive=True)
        )
    return stops


def unique_tracked_stop(exchange, trade, *, allow_missing=False):
    stops = listed_stops(exchange, trade)
    tracked = [str(order.order_id) for order in trade.open_sl_orders]
    if allow_missing and not tracked and not stops:
        return None
    if len(tracked) != 1 or set(tracked) != set(stops):
        details = '、'.join(
            f"{price:g}（{'已跟踪' if ident in tracked else '未跟踪'}）" for ident, price in stops.items()
        )
        raise ValueError('交易所止损未唯一匹配：' + (details or '未查到活跃止损'))
    return stops[tracked[0]]


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
    signature = [None, float(trade.amount)]
    reading_receipts = False
    boundary_info = {}
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
        if strategy.config.get('margin_mode') == 'isolated':
            raw_liq = refresh_isolated_boundary(strategy, trade)
            boundary = trade.liquidation_price
            boundary_info = dict(exchange_liquidation=raw_liq, buffered_liquidation=boundary)
            target = (min if trade.is_short else max)(target, _number(boundary, positive=True))
        reading_receipts = True
        listed = unique_tracked_stop(exchange, trade)
        ident = trade.open_sl_orders[0].order_id
        try:
            raw = exchange.fetch_stoploss_order(ident, trade.pair)
        except Exception:
            raise ValueError('交易所止损回执暂未核实') from None
        result = _evaluate(exchange, trade, ident, raw, target)
        if listed != result['stop_price']:
            raise ValueError('条件单列表与止损回执已变化，等待重新核验')
        stage = state.get('stage', 0)
        exact = manual_stop_active(state, stage) or (
            stage == 2 and state.get('tail_stop', plan.get('tail_stop_preset')) is not None
        )
        if exact and result['stop_price'] != signature[0]:
            raise ValueError('交易所触发价与手动止损设置尚未一致')
    except ValueError as error:
        reason = str(error) if reading_receipts else '当前保护目标或强平边界无法核实'
        result = {'status': 'unconfirmed', 'reason': reason}
    except Exception as error:
        logger.warning('Stop verification unavailable (%s)', type(error).__name__)
        result = {'status': 'unconfirmed', 'reason': '当前保护目标或交易所数据无法核实'}
    result.update(boundary_info)
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
        confirmed = result['status'] == 'confirmed'
        queued = strategy._exit_notice(
            notice(
                '止损保护',
                f'{trade.pair} #{trade.id}',
                summary=['止损已确认' if confirmed else '需要处理：止损尚未确认'],
                sections=[
                    (
                        '核验结果',
                        [
                            f"策略目标：{signature[0]:g}",
                            f"触发价：{result['stop_price']:g}",
                            f"覆盖数量：{result['covered_amount']:g}",
                        ]
                        if confirmed
                        else [result['reason'], '已暂停加仓，平仓照常'],
                    )
                ],
                action='用 /order 与交易所核对止损单及覆盖数量。',
                at=now,
            ),
            key=f'protection:{trade.id}',
        )
        if queued:
            result['notified_status'] = result['status']
            result['notified_key'] = key
            state['protection'] = result
            strategy._save_execution(trade, state)
    return result['status'] == 'confirmed'
