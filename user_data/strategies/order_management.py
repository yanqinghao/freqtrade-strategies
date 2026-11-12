"""Trading-thread operations for the Telegram order panel."""
import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from math import isfinite, ulp

from freqtrade.enums import SignalDirection
from freqtrade.persistence import Trade
from candle_exits import RULE_FIELDS, validate_rules
from entry_guard import ensure_position_matches
from entry_lifecycle import (
    _receipt,
    TERMINAL,
    entry_ttl,
    install_entry_receipt_guard,
    _reconcile_intent,
)
from entry_risk import number, saved_plan, effective_stop, observed_quantities
from entry_pause import ENDED as PAUSE_ENDS
from entry_targets import clear_units


PAUSE_PHASES = {'paused', 'canceling', 'resuming', 'submitting', 'unknown'}
PAUSE_LABELS = {
    'paused': '暂停',
    'canceling': '撤单核对中',
    'resuming': '恢复提交中',
    'submitting': '提交核对中',
    'unknown': '回执待核实',
}


def pause_plans(owner):
    store = getattr(owner._rpc._freqtrade.strategy, 'entry_requests', None)
    if store is None:
        return []
    return sorted(
        (
            meta
            for meta in store.pauses()
            if meta['phase'] in PAUSE_PHASES
            and (meta['decision'] not in PAUSE_ENDS or meta['phase'] != 'paused')
        ),
        key=lambda m: (m['created'], m['ref']),
        reverse=True,
    )


def pause_trade(meta, trades):
    """Match only durable order/trade identities, never another position in the same coin."""
    for trade in trades:
        if trade.pair != meta['request']['pair'] or trade.is_short != (
            meta['request']['direction'] == 'short'
        ):
            continue
        # Zero-fill trades are deleted and SQLite may reuse their numeric id.
        # A real entry order id or request tag is the durable association proof.
        if any(
            (meta.get('order_id') is not None and str(order.order_id) == str(meta['order_id']))
            or (order.ft_order_tag or '').endswith('|req=' + meta['attempt_ref'])
            for order in trade.orders
            if order.ft_order_side == trade.entry_side
        ):
            return trade
    return None


def pause_snapshot(owner, ref):
    strategy = owner._rpc._freqtrade.strategy
    meta = strategy.entry_requests.get_pause(ref)
    if meta is None:
        raise ValueError('暂停计划不存在，请返回订单列表。')
    trade = pause_trade(meta, Trade.get_open_trades())
    row = strategy.entry_requests.get(meta['attempt_ref'])
    uncertain = meta['phase'] in ('unknown', 'resuming', 'submitting') or (
        row is not None and row['state'] in ('submitting', 'unknown')
    )
    state = dict(
        ref=ref,
        phase=meta['phase'],
        attempt_ref=meta['attempt_ref'],
        ended=meta['decision'] if meta['decision'] in PAUSE_ENDS else None,
        request=meta['request'],
        deadline=meta['deadline'],
        created=meta['created'],
        version=meta.get('version', 1),
        trade_id=trade.id if trade else None,
        amount=float(trade.amount) if trade else None,
        observed_amount=float(observed_quantities(trade)[0]) if trade else None,
        orders=[
            (str(o.order_id), o.status, o.ft_is_open, float(o.safe_filled)) for o in trade.orders
        ]
        if trade
        else [],
        receipt_state=row['state'] if row else None,
    )
    revision = hashlib.sha256(
        json.dumps(state, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    return dict(
        meta=deepcopy(meta),
        revision=revision,
        uncertain=uncertain,
        trade_id=trade.id if trade else None,
        can_cancel=meta['phase'] in PAUSE_PHASES
        and meta['decision'] not in PAUSE_ENDS
        and datetime.now(timezone.utc).timestamp() < meta['deadline'],
    )


def cancel_pause(owner, ref, revision):
    data = pause_snapshot(owner, ref)
    if data['revision'] != revision:
        raise ValueError('计划阶段、订单或成交已变化，请刷新详情后重试。')
    if not data['can_cancel']:
        raise ValueError('该计划已结束、已恢复挂单或已有成交，请刷新并在订单详情核对。')
    controller = owner._rpc._freqtrade.strategy.entry_pause
    if controller is None:
        raise ValueError('暂停计划控制尚未启动，请稍后重试。')
    controller.cancel(ref)
    if data['uncertain'] or data['meta']['phase'] == 'canceling' or data['trade_id'] is not None:
        return '已取消后续自动恢复；已有订单或提交回执仍需核对，已有成交继续按原计划保护。'
    return '已取消暂停计划，不再自动恢复挂单。'


def get_trade(ident):
    trade = Trade.get_trades([Trade.is_open.is_(True), Trade.id == int(ident)]).first()
    if not trade:
        raise ValueError('订单已结束或不存在，请返回订单列表。')
    return trade


def position_profit(trade, rate):
    """Native remaining-position PnL, never realized proceeds or unfilled entry notional."""
    result = dict(profit=None, profit_abs=None, pnl_status='盈亏待更新')
    try:
        amount = number(trade.amount)
        observed, _ = observed_quantities(trade)
        # Display-only tolerance for floating-point accounting noise.
        noise = number(4 * max(ulp(float(observed)), ulp(float(amount))))
        if observed - amount > noise:
            return {**result, 'pnl_status': '成交待记账'}
        if not amount:
            return {**result, 'pnl_status': '未成交'}
        rate = float(number(rate, positive=True))
        profit = trade.calculate_profit(rate)
        percent = profit.profit_ratio * 100
        if not all(isfinite(value) for value in (profit.profit_abs, percent)):
            return result
        return dict(profit=profit.profit_ratio * 100, profit_abs=profit.profit_abs, pnl_status='可用')
    except (ValueError, TypeError, ArithmeticError):
        return result


def snapshot(owner, ident):
    bot = owner._rpc._freqtrade
    trade = get_trade(ident)
    strategy = bot.strategy
    plan = saved_plan(strategy, trade)
    state = strategy._execution(trade)
    rows, errors = [], []
    applied = trade.get_custom_data('assistant_plan', default={}) or {}
    if applied and applied.get('exit_points', []) != plan.get('exit_points', []):
        errors.append('TP 计划尚待策略应用，请稍后刷新。')
    for order in trade.open_orders:
        row = dict(
            id=str(order.order_id),
            side=order.ft_order_side,
            status=order.status,
            price=float(order.safe_placement_price),
            amount=float(order.safe_amount),
            filled=float(order.safe_filled),
            type=order.order_type,
        )
        try:
            remote = bot.exchange.fetch_order(order.order_id, trade.pair)
            if order.ft_order_side == trade.entry_side and order.order_type == 'limit':
                remote = _receipt(remote, trade, order.order_id)
            elif (
                not isinstance(remote, dict)
                or str(remote.get('id')) != str(order.order_id)
                or remote.get('symbol') != trade.pair
                or remote.get('side') != order.ft_order_side
                or remote.get('status') not in TERMINAL | {'open'}
            ):
                raise ValueError('订单回执无效')
            filled = float(number(remote.get('filled')))
            amount = float(number(remote.get('amount'), positive=True))
            floor = max(row['filled'], float(state.get('entry_observed', {}).get(row['id'], 0)))
            if filled < floor or filled > amount:
                raise ValueError('成交量无效')
            row.update(filled=filled, amount=amount, status=remote['status'])
            if remote.get('price') is not None:
                row['price'] = float(number(remote['price'], positive=True))
            if remote['status'] != 'open' or filled != float(order.safe_filled):
                errors.append('交易所订单已有变化，等待原生流程记账后刷新。')
        except Exception:
            errors.append('订单回执暂未核实，请刷新或核对交易所。')
        rows.append(row)
    rate = None
    try:
        rate = float(
            number(
                bot.exchange.get_rate(
                    trade.pair, side='exit', is_short=trade.is_short, refresh=True
                ),
                positive=True,
            )
        )
        mode = strategy.config.get('runmode')
        simulated = strategy.config.get('dry_run', getattr(mode, 'value', mode) == 'dry_run')
        ensure_position_matches(
            bot.exchange, trade.pair, trade if trade.amount > 0 else None, dry_run=simulated
        )
    except Exception:
        errors.append('最新价格或仓位暂未核实，请刷新或核对交易所。')
    entries = [r for r in rows if r['side'] == trade.entry_side]
    kind = (
        'partial'
        if entries and (trade.amount > 0 or any(r['filled'] > 0 for r in entries))
        else ('pending' if entries else 'position')
    )
    data = dict(
        id=trade.id,
        pair=trade.pair,
        short=trade.is_short,
        kind=kind,
        amount=float(trade.amount),
        entry=float(trade.open_rate),
        leverage=float(trade.leverage),
        plan=plan,
        orders=rows,
        stage=state.get('stage', 0),
        extension=state.get('extension'),
        full_exit=state.get('full_exit'),
        tp_filled=state.get('tp_filled', {}),
        final_tp_started=state.get('final_tp_started', False),
    )
    try:
        data['stop'] = float(effective_stop(strategy, trade, plan))
    except ValueError:
        data['stop'] = None
    data['revision'] = hashlib.sha256(
        json.dumps(data, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    data.update(
        rate=rate,
        fresh=not errors,
        error='\n'.join(dict.fromkeys(errors)),
        checked=datetime.now(timezone.utc).strftime('%m-%d %H:%M:%S UTC'),
    )
    data.update(position_profit(trade, rate))
    data['pause_ref'] = next(
        (m['ref'] for m in pause_plans(owner) if pause_trade(m, [trade]) is not None), None
    )
    data['deadline'] = None
    if len(entries) == 1:
        order = trade.select_order_by_order_id(entries[0]['id'])
        try:
            ttl = entry_ttl(order)
            if ttl is not None:
                data['deadline'] = int(order.order_date_utc.timestamp() + ttl)
        except ValueError:
            data.update(fresh=False, error='挂单有效期无法核实，请核对 /order。')
    return data


def checked(owner, ident, revision):
    data = snapshot(owner, ident)
    if data['revision'] != revision:
        raise ValueError('订单或计划已变化，请刷新详情后重试。')
    if not data['fresh']:
        raise ValueError('最新订单状态未核实，本次未修改。请刷新或核对交易所。')
    return get_trade(ident), data


def edit_plan(owner, ident, revision, changes):
    bot = owner._rpc._freqtrade
    with bot._exit_lock:
        trade, data = checked(owner, ident, revision)
        plan, state = deepcopy(data['plan']), bot.strategy._execution(trade)
        if not plan:
            raise ValueError('没有可修改的手动计划，请先 /setmanual 设置 SL。')
        if state.get('full_exit'):
            raise ValueError('正在退出，请等待结果后刷新。')
        if data['kind'] == 'partial' or any(r['side'] == trade.exit_side for r in data['orders']):
            raise ValueError('仍有成交待处理或退出订单，请等待完成后修改。')
        if not changes or set(changes) - {'stop_loss', 'tp1', 'tp2', 'tp3', *RULE_FIELDS}:
            raise ValueError('不支持的修改字段')
        if state.get('extension') and any(k.startswith(('tp', 'extend')) for k in changes):
            raise ValueError('延伸止盈已生效，不能修改 TP 或延伸条件。')
        for key, value in changes.items():
            if key.startswith('tp'):
                index = int(key[-1]) - 1
                targets = plan.setdefault('exit_points', [])
                if (
                    index < state.get('stage', 0)
                    or state.get('tp_filled', {}).get(str(index + 1), 0) > 0
                ):
                    raise ValueError('该档 TP 已完成或已有成交，不能修改；可修改后续档。')
                if state.get('final_tp_started'):
                    raise ValueError('最终止盈已开始，不能修改 TP。')
                if index > len(targets) or (
                    index == len(targets)
                    and (state.get('stage', 0) or any(state.get('tp_filled', {}).values()))
                ):
                    raise ValueError('请按 TP1、TP2、TP3 顺序设置；止盈开始后不能增加档数。')
                value = float(number(value, positive=True))
                if index == len(targets):
                    targets.append(value)
                else:
                    targets[index] = value
            elif key == 'stop_loss':
                value = float(number(value, positive=True))
                stop = data['stop']
                if stop and (value > stop if trade.is_short else value < stop):
                    raise ValueError(f'保护只能收紧，当前有效 SL 为 {stop:g}。')
                reference = (
                    data['orders'][0]['price'] if data['kind'] == 'pending' else data['rate']
                )
                if value <= reference if trade.is_short else value >= reference:
                    raise ValueError('SL 已越过参考价；立即退出请使用快速全平。')
                plan[key] = value
            else:
                plan[key] = value
        targets = plan.get('exit_points', [])
        if targets != sorted(set(targets), reverse=trade.is_short):
            raise ValueError('TP 必须按盈利方向严格递进，不能交换档位。')
        if any(p >= trade.open_rate if trade.is_short else p <= trade.open_rate for p in targets):
            raise ValueError('TP 必须位于开仓价的盈利方向。')
        validate_rules(plan)
        clear_units(
            plan, stop='stop_loss' in changes, tp=any(k in changes for k in ('tp1', 'tp2', 'tp3'))
        )
        plan['trade_id'] = trade.id
        # Persist intent in the same JSON update. _plan observes this marker after a crash too.
        if targets != data['plan'].get('exit_points', []):
            plan['tp_keep_progress_from'] = data['plan'].get('exit_points', [])
        if bot.strategy.entry_pause:
            bot.strategy.entry_pause.revise_trade(trade.id, plan)
        bot.strategy.set_manual_plan(trade.pair, plan)
        bot.strategy._plan(trade)
        return '已保存；保护只收紧，收盘规则从下一次新收盘开始判断。'


def replace_entry(owner, ident, revision, price, ref):
    bot, store = owner._rpc._freqtrade, owner._rpc._freqtrade.strategy.entry_requests
    previous = store.get(ref)
    if previous:
        return owner._entry_request_status(ref)
    trade, data = checked(owner, ident, revision)
    entries = [r for r in data['orders'] if r['side'] == trade.entry_side]
    if (
        data['kind'] != 'pending'
        or len(entries) != 1
        or len(data['orders']) != 1
        or trade.nr_of_successful_entries
        or entries[0]['filled']
        or entries[0]['type'] != 'limit'
    ):
        raise ValueError('仅可修改尚未成交的初始限价单；已成交价不可修改。')
    deadline = data['deadline']
    remaining = int((deadline or 0) - datetime.now(timezone.utc).timestamp())
    if remaining < 60:
        raise ValueError('挂单已到期或剩余不足一分钟，不能改价重挂。')
    price = float(number(price, positive=True))
    plan = data['plan']
    if not plan.get('stop_loss'):
        raise ValueError('缺少 SL，不能改价。')
    if (plan['stop_loss'] <= price if trade.is_short else plan['stop_loss'] >= price) or any(
        p >= price if trade.is_short else p <= price for p in plan.get('exit_points', [])
    ):
        raise ValueError('新开仓价必须位于 SL 和 TP 之间。')
    request = {
        k: v
        for k, v in plan.items()
        if k in {'direction', 'stop_loss', 'exit_points', *RULE_FIELDS}
    }
    request.update(
        pair=trade.pair,
        size=float(trade.stake_amount),
        leverage=float(trade.leverage),
        entry_price=price,
        ttl_seconds=remaining,
        entry_deadline=deadline,
    )
    validate_rules(request)
    owner._rpc._force_entry_validations(trade.pair, SignalDirection(request['direction']))
    for other in store.unresolved(trade.pair):
        if owner._entry_request_status(other['ref'])['state'] == 'unknown':
            raise ValueError('该币种有待核实请求，请先 /request 核对。')
    _, fresh = store.reserve(ref, request)
    if not fresh:
        return owner._entry_request_status(ref)
    if bot.strategy.entry_pause:
        bot.strategy.entry_pause.cancel_trade(trade.id, 'replaced')
    store.set_status(ref, 'submitting', {'message': '正在撤旧单并核对成交，未确认前不会重挂。'})
    try:
        with bot._exit_lock:
            trade, data = checked(owner, ident, revision)
            order = trade.select_order_by_order_id(entries[0]['id'])
            install_entry_receipt_guard(bot.strategy, bot)
            _reconcile_intent(bot.strategy, trade, order.order_id, True)
            receipt = _receipt(
                bot.exchange.cancel_order_with_result(
                    order.order_id, trade.pair, order.safe_amount
                ),
                trade,
                order.order_id,
            )
            if receipt['status'] not in TERMINAL:
                raise ValueError('撤单结果未确认')
            if receipt['filled'] > 0:
                bot.update_trade_state(trade, order.order_id, receipt)
                _reconcile_intent(bot.strategy, trade, order.order_id, False)
                return store.set_status(
                    ref, 'rejected', {'message': '旧单已有成交，已停止改价重挂；已成交仓位交由原保护流程管理。'}
                )
            bot.handle_cancel_enter(trade, receipt, order, 'Telegram 修改开仓价')
            Trade.commit()
            if Trade.get_trades([Trade.id == int(ident)]).first() is not None:
                raise ValueError('旧单清理尚未确认')
            bot.strategy.delete_manual_plan(request['pair'], trade_id=int(ident))
            if bot.strategy.entry_pause:
                bot.strategy.entry_pause.retire_replaced(int(ident))
        if datetime.now(timezone.utc).timestamp() >= deadline - 60:
            return store.set_status(ref, 'rejected', {'message': '旧单已撤；临近原到期时间，未重挂。'})
        try:
            with bot.strategy.risk_request(ref):
                result = owner._submit_entry(request, request_ref=ref)
        except bot.strategy.risk_error as exc:
            return store.set_status(ref, 'rejected', {'message': '旧单已撤，新单风险检查未通过：' + str(exc)})
        return store.set_status(ref, 'submitted', result)
    except Exception:
        Trade.rollback()
        return store.set_status(ref, 'unknown', {'message': '改价结果待核实，不会自动重挂；请 /request 查询并核对交易所。'})


def close_or_cancel(owner, ident, revision, action):
    trade, data = checked(owner, ident, revision)
    if action == 'close':
        if data['amount'] <= 0 or data['orders'] or data['full_exit']:
            raise ValueError('当前不能快速全平：请先等待在途订单结束或撤销入场余量，再刷新。')
        strategy = owner._rpc._freqtrade.strategy
        strategy.request_full_exit(trade, '用户确认快速全平', exit_tag='manual_panel_close')
        strategy.mark_full_exit_attempt(trade, datetime.now(timezone.utc))
        result = owner._close_position(str(ident), 100)
        return '已提交全平请求，请刷新查看实际成交。\n' + result['result']
    if action == 'cancel':
        if not data['orders'] or any(r['side'] != trade.entry_side for r in data['orders']):
            raise ValueError('当前没有可撤的入场挂单，请刷新。')
        bot = owner._rpc._freqtrade
        if bot.strategy.entry_pause:
            bot.strategy.entry_pause.cancel_trade(trade.id)
        with bot._exit_lock:
            install_entry_receipt_guard(bot.strategy, bot)
            for row in data['orders']:
                order = trade.select_order_by_order_id(row['id'])
                if order.order_type != 'limit':
                    raise ValueError('市价订单正在处理，请等待成交后刷新。')
                _reconcile_intent(bot.strategy, trade, order.order_id, True)
                receipt = _receipt(
                    bot.exchange.cancel_order_with_result(
                        order.order_id, trade.pair, order.safe_amount
                    ),
                    trade,
                    order.order_id,
                )
                if receipt['status'] not in TERMINAL:
                    raise ValueError('撤单尚未确认，已保留核对记录；请刷新，勿重复撤单。')
                if receipt['filled'] > 0:
                    bot.update_trade_state(trade, order.order_id, receipt)
                    _reconcile_intent(bot.strategy, trade, order.order_id, False)
                else:
                    bot.handle_cancel_enter(trade, receipt, order, 'Telegram 撤销入场余量')
                    if not trade.amount:
                        bot.strategy.delete_manual_plan(trade.pair, trade_id=trade.id)
            Trade.commit()
        return '撤单回执已确认，已有成交继续按原计划保护；请刷新核对持仓。'
    raise ValueError('未知操作')
