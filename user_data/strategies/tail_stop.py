"""Explicit per-position stop replacements, called under the native exit lock."""

import logging
from datetime import datetime, timezone
from time import monotonic, sleep

from freqtrade.exchange import ROUND_DOWN, ROUND_UP
from freqtrade.exceptions import ExchangeError, OperationalException
from freqtrade.persistence import Trade
from entry_guard import ensure_position_matches
from entry_risk import cost_stop, number
from stop_protection import _evaluate, check_protection


logger = logging.getLogger(__name__)


def _crossed(short, stop, rate):
    return stop <= rate if short else stop >= rate


def _cancel_old_stop(bot, trade, order, rate, simulated):
    exchange, ident = bot.exchange, order.order_id
    row = exchange.fetch_stoploss_order(ident, trade.pair)
    trigger = number(row.get('stopPrice') or row.get('triggerPrice'), positive=True)
    if simulated:
        if row.get('status') != 'open' or row.get('status_stop') == 'triggered':
            raise ValueError('旧止损状态已变化，请刷新。')
    else:
        _evaluate(exchange, trade, ident, row, trigger)
    if _crossed(trade.is_short, trigger, number(rate)):
        raise ValueError('已触及旧止损，等待交易所处理。')
    # Never treat a cancel acknowledgement (or a synthesized fallback) as a terminal receipt.
    try:
        exchange.cancel_stoploss_order(ident, trade.pair)
    except (ExchangeError, OperationalException, ValueError) as error:
        # A lost or unparseable acknowledgement can follow a successful cancellation.
        logger.warning(
            'Stop cancel acknowledgement unavailable trade=%s order=%s (%s)',
            trade.id,
            ident,
            type(error).__name__,
        )
    deadline = monotonic() + 4
    for delay in (0, 0.15, 0.35, 0.75, 1.5):
        if delay:
            if monotonic() + delay > deadline:
                raise ValueError('撤单回执核对超时，新止损未设置。请刷新核对当前保护价。')
            sleep(delay)
        try:
            final = exchange.fetch_stoploss_order(ident, trade.pair)
        except ExchangeError as error:
            logger.warning(
                'Stop receipt unavailable trade=%s order=%s (%s)',
                trade.id,
                ident,
                type(error).__name__,
            )
            continue
        status = final.get('status')
        logger.info(
            'Stop cancel receipt trade=%s order=%s status=%s filled=%s',
            trade.id,
            ident,
            status,
            final.get('filled'),
        )
        if (
            str(final.get('id')) != str(ident)
            or final.get('symbol') != trade.pair
            or final.get('side') != trade.exit_side
            or final.get('status_stop') == 'triggered'
            or (final.get('filled') is not None and number(final['filled']) != 0)
        ):
            raise ValueError('旧止损撤单或成交待核实，等待原生止损流程处理。')
        if status in ('canceled', 'cancelled'):
            if final.get('filled') is not None:
                break
        elif status != 'open':
            raise ValueError('旧止损撤单或成交待核实，等待原生止损流程处理。')
    else:
        raise ValueError('交易所尚未确认旧止损已撤且零成交，新止损未设置。请刷新核对当前保护价。')
    logger.info('Stop cancel confirmed trade=%s order=%s', trade.id, ident)
    bot.update_trade_state(trade, ident, final, stoploss_order=True)
    Trade.commit()


def tail_price(exchange, trade, value):
    short = trade.is_short
    cost = cost_stop(trade)
    requested = cost if value == 'cost' else number(value, positive=True)
    if requested > cost if short else requested < cost:
        raise ValueError('尾仓止损最多可放宽到含手续费成本价。')
    price = float(
        exchange.price_to_precision(
            trade.pair, float(requested), rounding_mode=ROUND_DOWN if short else ROUND_UP
        )
    )
    rounded = number(price, positive=True)
    if rounded > cost if short else rounded < cost:
        raise ValueError('成本价的交易所精度无法核实。')
    return price


def store_plan(strategy, trade, plan):
    if strategy.entry_pause:
        strategy.entry_pause.revise_trade(trade.id, plan)
    strategy.set_manual_plan(trade.pair, plan)
    strategy._plan(trade)


def replace_stop(bot, trade, data, price, *, plan=None):
    """Replace a verified, untriggered stop under the caller's native exit lock."""
    strategy, exchange = bot.strategy, bot.exchange
    state = strategy._execution(trade)
    short = trade.is_short
    rate, old = data['rate'], data['stop']
    if old is None or _crossed(short, old, rate) or _crossed(short, price, rate):
        raise ValueError('已触及当前或请求的止损价，请核对持仓。')
    if trade.liquidation_price and _crossed(short, trade.liquidation_price, price):
        raise ValueError('止损已越过强平保护边界。')
    targets = strategy._targets(data['plan'], trade)
    final = targets[-1] if targets and state.get('stage', 0) >= len(targets) - 1 else None
    if final and _crossed(short, rate, final):
        raise ValueError('已触及最终止盈价，等待退出。')
    mode = strategy.config.get('runmode')
    simulated = strategy.config.get('dry_run', getattr(mode, 'value', mode) == 'dry_run')
    on_exchange = strategy.order_types.get('stoploss_on_exchange')
    logger.info('Stop edit trade=%s old=%s requested=%s', trade.id, old, price)
    if on_exchange:
        orders = list(trade.open_sl_orders)
        if len(orders) > 1 or (not orders and not simulated):
            raise ValueError('当前交易所止损单尚未唯一确认，请等待止损核验后刷新。')
        if (
            plan is None
            and orders
            and old == price
            and state.get('tail_stop') == price
            and check_protection(strategy, trade, datetime.now(timezone.utc), force=True)
            and not simulated
        ):
            return f'当前尾仓止损已经是 {price:g}，交易所已确认。'
        if orders:
            _cancel_old_stop(bot, trade, orders[0], rate, simulated)
            ensure_position_matches(exchange, trade.pair, trade, dry_run=simulated)
            rate = float(
                number(
                    exchange.get_rate(trade.pair, side='exit', is_short=short, refresh=True),
                    positive=True,
                )
            )
            if (
                not trade.is_open
                or trade.has_open_orders
                or _crossed(short, old, rate)
                or _crossed(short, price, rate)
                or (final and _crossed(short, rate, final))
            ):
                raise ValueError('撤单后行情或持仓已变化，交回原生保护与退出流程。')

    if plan is not None:
        store_plan(strategy, trade, plan)
    state = strategy._execution(trade)
    if plan is None:
        label = '尾仓止损'
        state.update(tail_stop=price, tail_overridden_sl=data['plan']['stop_loss'])
        state.pop('manual_stop_phase', None)
    else:
        label = '硬止损'
        state['manual_stop_phase'] = [state.get('stage', 0), bool(state.get('extension'))]
        state.pop('tail_overridden_sl', None)
    state['protected_stop'] = price
    state.pop('protection', None)
    trade.stop_loss = price
    trade.stop_loss_pct = -abs((price / rate - 1) * (trade.leverage or 1))
    strategy._save_execution(trade, state)
    Trade.commit()
    if not on_exchange:
        return f'策略{label}已设为 {price:g}。'
    try:
        bot.handle_stoploss_on_exchange(trade)
    except (ExchangeError, OperationalException) as error:
        logger.warning('Stop placement unavailable trade=%s (%s)', trade.id, type(error).__name__)
    Trade.commit()
    if not trade.is_open:
        return '持仓已结束，请核对成交。'
    confirmed = check_protection(strategy, trade, datetime.now(timezone.utc), force=True)
    if simulated:
        return f'模拟{label}目标已设为 {price:g}。'
    return f'{label} {price:g}：' + ('交易所已确认。' if confirmed else '交易所尚未确认，请核对 /order。')
