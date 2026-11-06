"""Settle tracked limit entries using real cancellation receipts and native accounting."""
import logging
import re
from math import isfinite
from freqtrade.exceptions import ExchangeError
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)
TERMINAL = {'closed', 'canceled', 'expired', 'rejected'}


def entry_ttl(order):
    tag = order.ft_order_tag or ''
    match = re.fullmatch(
        r'manual_(?:long|short)\|ttl=(\d+)(?:\|until=(\d+))?(?:\|req=[0-9a-f]{24})?', tag
    )
    if not match:
        if '|ttl=' in tag:
            raise ValueError('订单有效期元数据无效')
        return None
    seconds = int(match[1])
    if not 60 <= seconds <= 5 * 86400:
        raise ValueError('订单有效期元数据无效')
    if match[2]:
        return int(match[2]) - order.order_date_utc.timestamp()
    return seconds


def _notice(strategy, trade, order_id, code, message):
    state = strategy._execution(trade)
    events = state.setdefault('entry_events', {})
    if events.get(str(order_id)) == code:
        return
    if strategy._exit_notice(f'{trade.pair}：{message}', key=f'entry:{trade.id}:{order_id}'):
        events[str(order_id)] = code
        strategy._save_execution(trade, state)


def _reconcile_intent(strategy, trade, order_id, active):
    state = strategy._execution(trade)
    pending = state.setdefault('entry_reconcile', {})
    if active:
        pending[str(order_id)] = True
    else:
        pending.pop(str(order_id), None)
    strategy._save_execution(trade, state)


def _receipt(row, trade, order_id):
    if (
        not isinstance(row, dict)
        or str(row.get('id')) != str(order_id)
        or row.get('symbol') != trade.pair
        or row.get('side') != trade.entry_side
        or row.get('type') != 'limit'
    ):
        raise ValueError('订单回执身份不一致')
    status = row.get('status')
    if status == 'cancelled':
        row = {**row, 'status': 'canceled'}
        status = 'canceled'
    if status not in TERMINAL | {'open'}:
        raise ValueError('订单状态未确认')
    values = []
    for key in ('amount', 'filled'):
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not isfinite(value):
            raise ValueError('订单成交量未确认')
        values.append(float(value))
    amount, filled = values
    if amount <= 0 or filled < 0 or filled > amount:
        raise ValueError('订单成交量无效')
    return row


def install_entry_receipt_guard(strategy, bot):
    """Validate tracked entry receipts before native code can overwrite fill evidence.

    Keep upstream code intact. Only this exchange instance's two receipt boundaries
    are wrapped; native code already defers an order on ExchangeError.
    """
    exchange = bot.exchange
    if getattr(exchange, '_assistant_entry_receipts', False):
        return

    def checked(order_id, pair, row):
        for trade in Trade.get_trades([Trade.is_open.is_(True), Trade.pair == pair]):
            order = trade.select_order_by_order_id(order_id)
            if not order or order.ft_order_side != trade.entry_side or order.order_type != 'limit':
                continue
            if not (order.ft_order_tag or '').startswith(('manual_long', 'manual_short')):
                continue
            state = strategy._execution(trade)
            evidence = state.setdefault('entry_observed', {})
            floor = max(float(order.filled or 0), float(evidence.get(str(order_id), 0)))
            try:
                row = _receipt(row, trade, order_id)
                if row['filled'] < floor:
                    raise ValueError('累计成交量倒退')
            except ValueError:
                if floor > 0:
                    evidence[str(order_id)] = floor
                    state.setdefault('entry_reconcile', {})[str(order_id)] = True
                    strategy._save_execution(trade, state)
                raise ExchangeError('入场回执无效或累计成交量倒退，保留记录等待核实') from None
            if row['filled'] > float(evidence.get(str(order_id), 0)):
                evidence[str(order_id)] = float(row['filled'])
                strategy._save_execution(trade, state)
            break
        return row

    fetch = exchange.fetch_order
    cancel = exchange.cancel_order_with_result

    def fetch_checked(order_id, pair, *args, **kwargs):
        return checked(order_id, pair, fetch(order_id, pair, *args, **kwargs))

    def cancel_checked(order_id, pair, *args, **kwargs):
        return checked(order_id, pair, cancel(order_id, pair, *args, **kwargs))

    exchange.fetch_order = fetch_checked
    exchange.cancel_order_with_result = cancel_checked
    exchange._assistant_entry_receipts = True
    logger.info('Tracked manual entry receipt validation installed')


def process_entry_orders(strategy, bot, trade, now):
    install_entry_receipt_guard(strategy, bot)
    trade_id, pair = trade.id, trade.pair
    pending = strategy._execution(trade).get('entry_reconcile', {})
    for order in [o for o in trade.orders if o.ft_is_open or str(o.order_id) in pending]:
        order_id = order.order_id
        if (
            order.ft_order_side != trade.entry_side
            or order.order_type != 'limit'
            or not (order.ft_order_tag or '').startswith(('manual_long', 'manual_short'))
        ):
            continue
        try:
            ttl = entry_ttl(order)
            expired = ttl is not None and (now - order.order_date_utc).total_seconds() >= ttl
            partial = float(order.filled or 0) > 0
            if not expired and not partial and str(order_id) not in pending:
                continue
            with bot._exit_lock:
                current = _receipt(
                    bot.exchange.fetch_order(order.order_id, trade.pair), trade, order.order_id
                )
                if current['status'] == 'open':
                    _reconcile_intent(strategy, trade, order_id, True)
                    result = bot.exchange.cancel_order_with_result(
                        order.order_id, trade.pair, trade.amount
                    )
                    current = _receipt(result, trade, order.order_id)
                if current['status'] not in TERMINAL:
                    _notice(strategy, trade, order.order_id, 'pending', '撤单尚未确认，继续核对该订单；未提交替代订单。')
                    continue
                filled = float(current['filled'])
                reason = '部分成交，优先保护已成交仓位' if filled > 0 else '限价单有效期到期'
                if filled > 0:
                    _reconcile_intent(strategy, trade, order_id, True)
                    order.ft_cancel_reason = reason
                    # A positive terminal fill must never pass through native's tiny-fill
                    # pre-cancel minimum or zero-isclose cleanup branch.
                    bot.update_trade_state(trade, order.order_id, current)
                    _reconcile_intent(strategy, trade, order_id, False)
                    if not trade.is_open:
                        continue
                    text = (
                        f'已撤未成交余量，已成交 {filled:g}，交由原生持仓与止损流程管理。'
                        if current['status'] != 'closed'
                        else f'订单已结束并成交 {filled:g}，交由原生持仓与止损流程管理。'
                    )
                    _notice(strategy, trade, order.order_id, 'settled_' + current['status'], text)
                else:
                    _notice(strategy, trade, order.order_id, 'empty_terminal', '限价单有效期到期，已确认零成交结束。')
                    had_position = trade.has_open_position
                    bot.handle_cancel_enter(trade, current, order, reason)
                    if not had_position:
                        strategy.delete_manual_plan(trade.pair, trade_id=trade.id)
                        return
                    _reconcile_intent(strategy, trade, order_id, False)
        except Exception as error:
            logger.warning('Entry lifecycle check unavailable (%s)', type(error).__name__)
            Trade.rollback()
            current_trade = Trade.get_trades([Trade.id == trade_id]).first()
            message = '入场订单状态暂未确认，请通过 /order 与交易所核对；不会盲目重下。'
            if current_trade is not None:
                _notice(strategy, current_trade, order_id, 'unconfirmed', message)
            else:
                strategy._exit_notice(f'{pair}：{message}')
