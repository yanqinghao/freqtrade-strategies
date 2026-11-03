"""Read-only, cost-basis overview. Never query an exchange or mutate trade state."""
from datetime import datetime, timedelta, timezone

from entry_lifecycle import entry_ttl
from entry_risk import effective_stop, estimate_exposure, load_limits, number, saved_plan


def _protection(trade, state, quantity, goal, now):
    if not quantity:
        return '交易所SL：未成交，暂无已观察仓位的保护确认。'
    row = state.get('protection') or {}
    if not isinstance(row, dict):
        return '交易所SL：核验数据不可用，请核对交易所。'
    if row.get('status') == 'simulated':
        return '交易所SL：模拟状态，不代表真实交易所已保护。'
    if row.get('status') != 'confirmed':
        return '交易所SL：待确认/尚未确认，请核对 /order 与交易所。'
    try:
        price = number(row.get('stop_price'), positive=True)
        covered = number(row.get('covered_amount'), positive=True)
        checked = number(row.get('checked_at'), positive=True)
        checked_at = datetime.fromtimestamp(float(checked), timezone.utc)
        age = (now - checked_at).total_seconds()
        if age < -5:
            label = '核验时间异常'
        elif age > 60:
            label = '核验已过期'
        elif covered < number(quantity):
            label = '最近核验覆盖不足'
        elif goal is None:
            label = '目标不可用，不能确认当前保护'
        elif (price > goal) if trade.is_short else (price < goal):
            label = '核验价落后当前目标'
        else:
            label = '最近核验快照'
        return f'交易所SL：{label}，触发 {price:g}，覆盖 {covered:g}；' f'{checked_at:%Y-%m-%d %H:%M:%S} UTC。'
    except Exception:
        return '交易所SL：核验数据不可用，请核对交易所。'


def _orders(trade, state, now):
    lines = []
    rows = [o for o in trade.orders if o.ft_is_open and o.ft_order_side == trade.entry_side]
    for order in rows[:10]:
        try:
            filled = max(
                number(order.safe_filled),
                number(state.get('entry_observed', {}).get(str(order.order_id), 0)),
            )
            remaining = number(order.safe_amount) - filled
            if remaining < 0:
                raise ValueError('quantity')
            ttl = entry_ttl(order) if order.order_type == 'limit' else None
            if order.order_type != 'limit':
                timing = '市价单等待原生核实'
            elif ttl is None:
                timing = '无逐单TTL，沿用原生超时'
            else:
                deadline = order.order_date_utc + timedelta(seconds=ttl)
                seconds = int((deadline - now).total_seconds())
                if seconds <= 0:
                    timing = '已到期，待核对撤单'
                else:
                    hours, rest = divmod(seconds, 3600)
                    minutes, seconds = divmod(rest, 60)
                    timing = f'剩余 {hours}h {minutes}m {seconds}s（{deadline:%m-%d %H:%M} UTC）'
            lines.append(f'入场单 {order.order_id}：已观察成交 {filled:g}，余量 {remaining:g}；{timing}。')
        except Exception:
            lines.append(f'入场单 {order.order_id}：数量或有效期数据不可用，请核对 /order。')
    if len(rows) > 10:
        lines.append(f'另有 {len(rows) - 10} 笔入场单，请查看 /order。')
    exits = sum(o.ft_is_open and o.ft_order_side == trade.exit_side for o in trade.orders)
    if exits:
        lines.append(f'待核实的普通退出订单：{exits}笔。')
    return lines


def _trade_message(strategy, trade, snapshot, now):
    lines = [f"{trade.pair} #{trade.id} {'空' if trade.is_short else '多'}"]
    try:
        native = number(trade.amount)
        leverage = number(trade.leverage, positive=True)
        lines.append(f'原生持仓数量 {native:g}，杠杆 {leverage:g}倍。')
    except Exception:
        native = None
        lines.append('原生持仓数量/杠杆不可用。')
    quantity = None
    if snapshot is None:
        lines.append('该币种参考敞口不可用，请核对持仓记录。')
    else:
        quantity = snapshot['pairs'][trade.pair]['quantity']
        lines.append(
            f"成本口径：持仓 {snapshot['position_notional']:g}U＋入场挂单 {snapshot['pending_notional']:g}U。"
        )
        if native is not None and number(quantity) > native:
            lines.append(f'已观察数量 {quantity:g}，原生尚未完成记账；不能据此认为已足额保护。')
    state = strategy._execution(trade)
    if not isinstance(state, dict):
        state = {}
        lines.append('执行状态不可用。')
    goal = None
    try:
        plan = saved_plan(strategy, trade)
        if not plan:
            raise ValueError('plan')
        goal = effective_stop(strategy, trade, plan)
        lines.append(f'目标保护SL {goal:g}（只收紧）。')
        targets = plan.get('exit_points', [])
        stage = state.get('stage', 0)
        if not isinstance(stage, int) or isinstance(stage, bool) or not 0 <= stage <= 3:
            lines.append('TP执行阶段不可用。')
        else:
            lines.append(f'已记录TP阶段 {stage}；目标 {targets}。')
        applied = trade.get_custom_data('assistant_plan', default={}) or {}
        if applied and applied.get('exit_points') != targets:
            lines.append('TP计划存在待应用更新，上述阶段仍为原生已记录阶段。')
    except Exception:
        lines.append('目标SL/TP计划不可用，请核对 /manual。')
    if quantity is None:
        lines.append('交易所SL核验快照无法与当前数量比对，请核对交易所。')
    else:
        lines.append(_protection(trade, state, quantity, goal, now))
    if state.get('full_exit'):
        lines.append('存在待处理全平兜底意图，请核对 /order、/status。')
    lines.extend(_orders(trade, state, now))
    return '\n'.join(lines)


def overview_messages(strategy, trades, bot_state, now):
    trades = [trade for trade in trades if trade.is_open]
    details = []
    position, pending = 0.0, 0.0
    unavailable = False
    for trade in trades:
        try:
            snapshot = estimate_exposure(strategy.dp._exchange, [trade], fresh=False)
            position += snapshot['position_notional']
            pending += snapshot['pending_notional']
        except Exception:
            snapshot = None
            unavailable = True
        details.append((trade, snapshot))
    summary = [f'交易总览｜{bot_state}｜{now:%Y-%m-%d %H:%M:%S} UTC', '范围：仅助手持仓和入场挂单。']
    if unavailable:
        summary.append('助手完整敞口不可用，至少一笔记录需核实。')
    else:
        summary.append(f'参考敞口：持仓 {position:g}U＋挂单 {pending:g}U＝{position + pending:g}U。')
    summary.append('本页按成交/挂单价粗估；实际下单按最新参考价复核。')
    try:
        limits = load_limits(strategy.risk_limits_file)
        if limits:
            summary.append(
                f"上限：单笔保证金 {limits['max_margin_usdt']:g}U，杠杆 {limits['max_leverage']:g}倍，"
                f"总敞口 {limits['max_total_notional_usdt']:g}U，"
                f"估算止损亏损 {limits['max_stop_loss_usdt']:g}U。"
            )
        else:
            summary.append('额度限制未启用。')
    except Exception:
        summary.append('风险配置不可用，新入场会被拒绝，请核对 assistant_risk.json。')
    if not trades:
        summary.append('暂无助手持仓或挂单。')
    try:
        rows = strategy.entry_requests.unresolved() if strategy.entry_requests.path.exists() else []
        if rows:
            summary.append(f'待核实请求 {len(rows)} 条（可能包含尚未关联敞口），请用 /request 核对：')
            summary.extend(f"{row['ref']} {row['request']['pair']}" for row in rows[:5])
            if len(rows) > 5:
                summary.append('另有请求待核实，请查看 /request。')
    except Exception:
        summary.append('请求日志不可用；不会把缺少日志当成没有待核实订单。')
    messages = ['\n'.join(summary)]
    for trade, snapshot in details:
        try:
            messages.append(_trade_message(strategy, trade, snapshot, now))
        except Exception:
            messages.append(f'{trade.pair} #{trade.id}：交易状态数据不可用，请核对 /status、/order。')
    return messages
