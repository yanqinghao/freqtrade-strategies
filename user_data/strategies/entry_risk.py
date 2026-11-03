"""Pure entry estimates and strict, assistant-only risk limits."""
import json
import logging
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from math import isfinite
from pathlib import Path

from freqtrade.persistence import Order, Trade

logger = logging.getLogger(__name__)
LIMITS = ('max_margin_usdt', 'max_leverage', 'max_total_notional_usdt', 'max_stop_loss_usdt')


class EntryRiskError(ValueError):
    """A local rejection, before invoking the exchange order submission."""


def number(value, *, positive=False):
    if isinstance(value, bool):
        raise EntryRiskError('风险数据含无效数值，本次未提交')
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        raise EntryRiskError('风险数据含无效数值，本次未提交') from None
    if (
        not result.is_finite()
        or not isfinite(float(result))
        or result < 0
        or (positive and not result)
    ):
        raise EntryRiskError('风险数据含无效数值，本次未提交')
    return result


def load_limits(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        row = json.loads(path.read_text())
        if not isinstance(row, dict) or set(row) != set(LIMITS):
            raise ValueError('schema')
        result = {key: float(number(row[key], positive=True)) for key in LIMITS}
        if result['max_leverage'] < 1:
            raise ValueError('leverage')
        return result
    except Exception:
        raise EntryRiskError('风险上限配置无效或不可读，已阻止新入场，请核对 assistant_risk.json') from None


def saved_plan(strategy, trade):
    direction = 'short' if trade.is_short else 'long'
    plan = strategy.manual_open.get(trade.pair) or {}
    if not isinstance(plan, dict):
        raise EntryRiskError('交易计划格式无效')
    if plan.get('trade_id') not in (None, trade.id) or plan.get('direction') != direction:
        plan = {}
    plan = plan or trade.get_custom_data('assistant_plan', default={}) or {}
    if not isinstance(plan, dict) or (
        plan
        and (plan.get('trade_id') not in (None, trade.id) or plan.get('direction') != direction)
    ):
        raise EntryRiskError('交易计划与持仓不匹配')
    return deepcopy(plan)


def plan_for_entry(strategy, request, trade):
    if trade and trade.is_short != (request['direction'] == 'short'):
        raise EntryRiskError('同一币种只允许单方向持仓，请先平掉原方向')
    if (
        trade
        and trade.amount
        and number(trade.leverage, positive=True) != number(request['leverage'], positive=True)
    ):
        raise EntryRiskError('加仓必须沿用现有持仓杠杆')
    plan = saved_plan(strategy, trade) if trade else {}
    plan.update(
        {k: v for k, v in request.items() if k not in ('pair', 'entry_price', 'ttl_seconds')}
    )
    plan.setdefault('exit_points', [])
    if not plan.get('stop_loss'):
        raise EntryRiskError('开仓必须提供 sl=<止损价>')
    number(plan['stop_loss'], positive=True)
    return plan


def effective_stop(strategy, trade, plan, fee_open=0, fee_close=0):
    stop = number(plan.get('stop_loss'), positive=True)
    if trade is None:
        return stop
    state = strategy._execution(trade)
    candidates = [stop]
    saved = trade.get_custom_data('assistant_plan', default={}) or {}
    stage = state.get('stage', 0)
    if saved and saved.get('exit_points') != plan.get('exit_points'):
        stage = 0
    if not isinstance(stage, int) or isinstance(stage, bool) or not 0 <= stage <= 3:
        raise EntryRiskError('TP执行阶段无效')
    if stage >= 1:
        fees = number(trade.fee_open or fee_open) + number(trade.fee_close or fee_close)
        candidates.append(
            number(trade.open_rate, positive=True) * (1 - fees if trade.is_short else 1 + fees)
        )
    targets = sorted(
        {number(p, positive=True) for p in plan.get('exit_points', [])}, reverse=trade.is_short
    )
    if stage >= 2 and targets:
        candidates.append(targets[0])
    for value in (state.get('protected_stop'), getattr(trade, 'stop_loss', None)):
        if value is not None and value != 0:
            candidates.append(number(value, positive=True))
    return min(candidates) if trade.is_short else max(candidates)


def observed_quantities(trade):
    state = trade.get_custom_data('assistant_execution', default={}) or {}
    observed = state.get('entry_observed', {})
    entries, exits, pending = Decimal(0), Decimal(0), Decimal(0)
    for order in trade.orders:
        if order.ft_order_side not in (trade.entry_side, trade.exit_side, 'stoploss'):
            continue
        filled = number(order.safe_filled)
        if order.ft_order_side == trade.entry_side:
            filled = max(filled, number(observed.get(str(order.order_id), 0)))
            amount = number(order.safe_amount, positive=True)
            if filled > amount:
                raise EntryRiskError('入场累计成交量与总量矛盾')
            entries += filled
            if order.ft_is_open:
                remaining = amount - filled
                price = number(order.safe_placement_price, positive=True)
                pending += remaining * price
        else:
            exits += filled
    return max(number(trade.amount), entries - exits, Decimal(0)), pending


def estimate_exposure(exchange, trades, *, fresh=True):
    positions, pending, pairs = Decimal(0), Decimal(0), {}
    for trade in trades:
        if not trade.is_open:
            continue
        quantity, reserved = observed_quantities(trade)
        price = Decimal(0)
        if quantity:
            try:
                price = number(
                    exchange.get_rate(trade.pair, side='entry', is_short=False, refresh=True)
                    if fresh
                    else trade.open_rate,
                    positive=True,
                )
            except Exception:
                raise EntryRiskError('无法核对助手持仓参考价，总敞口不可用') from None
        position = quantity * price
        positions += position
        pending += reserved
        pairs[trade.pair] = {
            'quantity': float(quantity),
            'position_notional': float(position),
            'pending_notional': float(reserved),
            'reference_price': float(price),
        }
    return {
        'position_notional': float(positions),
        'pending_notional': float(pending),
        'total_notional': float(positions + pending),
        'pairs': pairs,
    }


def unknown_requests(strategy, ignore_request=None):
    store = strategy.entry_requests
    if not store.path.exists():
        return []
    unknown = []
    for row in store.unresolved():
        if row['ref'] == ignore_request:
            continue
        orders = (
            Trade.session.query(Order)
            .filter(Order.ft_order_tag.endswith('|req=' + row['ref']))
            .all()
        )
        matching = [
            o
            for o in orders
            if o.ft_pair == row['request']['pair']
            and o.trade is not None
            and o.ft_order_side == ('sell' if row['request']['direction'] == 'short' else 'buy')
        ]
        if not matching:
            unknown.append(row)
    return unknown


def _fee(strategy, pair, side):
    exchange = strategy.dp._exchange
    try:
        value = number(
            exchange.get_fee(symbol=pair, order_type='market', side=side, taker_or_maker='taker')
        )
        if value >= 1:
            raise ValueError('rate')
        if (
            not value
            and exchange.markets[pair].get('taker') is None
            and strategy.config.get('fee') is None
        ):
            raise ValueError('unknown zero')
        return value
    except Exception:
        raise EntryRiskError('无法确认交易手续费参考，未把未知费用按零处理') from None


def assess_entry(strategy, request, trade, plan, *, rate=None, amount=None, ignore_request=None):
    try:
        exchange = strategy.dp._exchange
        pair = request['pair']
        market = exchange.markets[pair]
        if (
            market.get('linear') is not True
            or market.get('contract') is not True
            or not pair.endswith('/USDT:USDT')
        ):
            raise EntryRiskError('风险预览目前仅支持USDT线性合约')
        number(market.get('contractSize'), positive=True)
        margin, leverage = number(request['size'], positive=True), number(
            request['leverage'], positive=True
        )
        if leverage < 1:
            raise EntryRiskError('杠杆必须至少为1')
        if rate is None:
            rate = request.get('entry_price') or exchange.get_rate(
                pair, side='entry', is_short=request['direction'] == 'short', refresh=True
            )
        rate = number(rate, positive=True)
        raw_amount = (
            number(amount, positive=True) if amount is not None else margin * leverage / rate
        )
        quantity = number(
            exchange.amount_to_contract_precision(pair, float(raw_amount)), positive=True
        )
        notional = max(margin * leverage, quantity * rate)
        open_fee = _fee(strategy, pair, 'sell' if request['direction'] == 'short' else 'buy')
        close_fee = _fee(strategy, pair, 'buy' if request['direction'] == 'short' else 'sell')
        stop = effective_stop(strategy, trade, plan, open_fee, close_fee)
        distance = stop - rate if request['direction'] == 'short' else rate - stop
        if distance <= 0:
            raise EntryRiskError('有效保护止损已越过本次入场价，请核对计划')
        loss, fees = quantity * distance, notional * open_fee + quantity * stop * close_fee
        if trade and trade.amount:
            existing = number(trade.amount, positive=True)
            original = number(trade.open_rate, positive=True)
            old_distance = stop - original if trade.is_short else original - stop
            loss += existing * max(old_distance, Decimal(0))
            fees += existing * original * max(
                number(trade.fee_open or 0), open_fee
            ) + existing * stop * max(number(trade.fee_close or 0), close_fee)
        exposure = estimate_exposure(exchange, Trade.get_open_trades())
        total = number(exposure['total_notional']) + notional
        limits = load_limits(strategy.risk_limits_file)
        reasons = []
        if limits:
            if margin > number(limits['max_margin_usdt']):
                reasons.append(f"单笔保证金 {margin:g}U 超过上限 {limits['max_margin_usdt']:g}U")
            if leverage > number(limits['max_leverage']):
                reasons.append(f"杠杆 {leverage:g}倍超过上限 {limits['max_leverage']:g}倍")
            if total > number(limits['max_total_notional_usdt']):
                reasons.append(f"助手总名义敞口预计 {total:g}U 超过上限 {limits['max_total_notional_usdt']:g}U")
            if loss + fees > number(limits['max_stop_loss_usdt']):
                reasons.append(
                    f"该币种加仓后估算止损亏损 {loss + fees:g}U 超过上限 {limits['max_stop_loss_usdt']:g}U"
                )
            if unknown_requests(strategy, ignore_request):
                reasons.append('存在待核实入场请求，助手总敞口无法可靠确认')
        return {
            'pair': pair,
            'direction': request['direction'],
            'margin': float(margin),
            'leverage': float(leverage),
            'reference_price': float(rate),
            'amount': float(quantity),
            'notional': float(notional),
            'stop_price': float(stop),
            'stop_distance_pct': float(distance / rate * 100),
            'price_loss': float(loss),
            'fees': float(fees),
            'estimated_loss': float(loss + fees),
            'exposure': exposure,
            'projected_total_notional': float(total),
            'limits': limits,
            'reasons': reasons,
            'allowed': not reasons,
        }
    except EntryRiskError:
        raise
    except Exception as error:
        logger.warning('Entry risk estimate unavailable (%s)', type(error).__name__)
        raise EntryRiskError('风险数据暂无法核实，本次未提交，请核对行情及记录') from None
