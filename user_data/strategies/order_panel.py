"""Native Telegram order list, detail, scoped edits and explicit confirmations."""
import logging
import secrets
import time
from math import fsum
from datetime import datetime, timezone

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ForceReply
from freqtrade.persistence import Trade
from freqtrade.rpc._telegram_base import authorized_only
from candle_exits import RULE_GROUPS, TIMEFRAMES, describe_rules
from entry_risk import number
from freqtrade.rpc import order_management as management

logger = logging.getLogger(__name__)
LABELS = {
    'stop_loss': '硬止损 SL',
    'tp1': 'TP1',
    'tp2': 'TP2',
    'tp3': 'TP3',
    'price': '开仓价',
    'soft_sl': '软止损价',
    'extend_trigger': '延伸触发价',
    'extend_tp3': '延伸 TP3',
}
KINDS = {'pending': '挂单未成交', 'partial': '部分成交 / 入场余量', 'position': '持仓'}


def order_list(owner, scope, page):
    rows, profits = [], []
    missing, positions = 0, 0
    for trade in Trade.get_open_trades():
        pending = any(o.ft_order_side == trade.entry_side for o in trade.open_orders)
        rate = None
        if trade.amount > 0:
            try:
                rate = owner._rpc._freqtrade.exchange.get_rate(
                    trade.pair, side='exit', is_short=trade.is_short, refresh=True
                )
            except Exception:
                pass  # Keep the row unavailable, never silently count a missing quote as zero.
        pnl = management.position_profit(trade, rate)
        if pnl['pnl_status'] != '未成交':
            positions += 1
            if pnl['profit_abs'] is None:
                missing += 1
            else:
                profits.append(pnl['profit_abs'])
        if scope == 'pending' and not pending or scope == 'position' and pnl['pnl_status'] == '未成交':
            continue
        kind = '部分成交' if pending and trade.amount > 0 else ('挂单' if pending else '持仓')
        if pnl['pnl_status'] == '成交待记账':
            kind = '成交待记账'
        rows.append(
            dict(
                id=trade.id,
                label=(
                    f"#{trade.id} {trade.pair.split('/')[0]} "
                    f"{'空' if trade.is_short else '多'} · {kind}"
                ),
                **pnl,
            )
        )
    rows.sort(key=lambda row: row['id'], reverse=True)
    pages = max(1, (len(rows) + 5) // 6)
    page = min(max(0, page), pages - 1)
    return dict(
        rows=rows[page * 6 : (page + 1) * 6],
        page=page,
        pages=pages,
        total=len(rows),
        profit_total=fsum(profits),
        profit_missing=missing,
        positions=positions,
        checked=datetime.now(timezone.utc).strftime('%m-%d %H:%M:%S UTC'),
    )


def profit_text(row):
    if row['profit_abs'] is None:
        return '未成交 · 不计浮盈亏' if row['pnl_status'] == '未成交' else row['pnl_status']
    # Avoid displaying a tiny signed loss as -0.0000.
    amount = round(row['profit_abs'], 4) or 0.0
    percent = round(row['profit'], 2) or 0.0
    return f'浮盈亏 {amount:+.4f} USDT · 收益率 {percent:+.2f}%'


class OrderPanelMixin:
    @staticmethod
    def _panel_identity(update):
        return (
            update.effective_chat.id,
            update.effective_user.id,
            update.effective_message.message_thread_id,
        )

    def _panel_token(self, update, payload):
        now = time.monotonic()
        tokens = getattr(self, '_panel_tokens', {})
        tokens = {k: v for k, v in tokens.items() if v['expires'] > now}
        # ponytail: bounded in-memory drafts; restart intentionally invalidates old buttons.
        while len(tokens) >= 512:
            tokens.pop(next(iter(tokens)))
        token = secrets.token_hex(12)
        tokens[token] = dict(owner=self._panel_identity(update), expires=now + 600, payload=payload)
        self._panel_tokens = tokens
        return token

    def _panel_take(self, update, token, consume=False):
        tokens = getattr(self, '_panel_tokens', {})
        row = tokens.get(token)
        if (
            not row
            or row['expires'] <= time.monotonic()
            or row['owner'] != self._panel_identity(update)
            or (
                row.get('message') is not None
                and row['message'] != update.effective_message.message_id
            )
        ):
            raise ValueError('按钮已过期或不属于你，请重新发送 /orders。')
        if consume:
            tokens.pop(token)
        return row['payload']

    def _panel_button(self, update, label, **payload):
        return InlineKeyboardButton(label, callback_data='om:' + self._panel_token(update, payload))

    async def _panel_send(self, update, text, buttons=None, *, prompt=False):
        markup = (
            ForceReply(selective=False, input_field_placeholder='回复价格，或回复“取消”')
            if prompt
            else InlineKeyboardMarkup(buttons or [])
        )
        current = {
            b.callback_data[3:]
            for row in (buttons or [])
            for b in row
            if b.callback_data and b.callback_data.startswith('om:')
        }
        tokens = getattr(self, '_panel_tokens', {})
        identity = self._panel_identity(update)
        # A newly rendered view or prompt supersedes older mutation buttons, even if
        # the old callback was already queued. Read-only navigation remains usable.
        for token, row in list(tokens.items()):
            if (
                row['owner'] == identity
                and token not in current
                and row['payload']['action'] not in ('list', 'detail')
            ):
                tokens.pop(token)
        query = update.callback_query
        if query and not prompt:
            result = await query.edit_message_text(text=text, reply_markup=markup, parse_mode=None)
            message_id = query.message.message_id
        else:
            result = await self._app.bot.send_message(
                chat_id=update.effective_chat.id,
                text=text,
                message_thread_id=update.effective_message.message_thread_id,
                reply_markup=markup,
                parse_mode=None,
            )
            message_id = result.message_id
        for token in current:
            if token in tokens:
                tokens[token]['message'] = message_id
        return result

    async def _panel_list(self, update, scope='all', page=0, notice=''):
        data = await self._await_command(order_list, self, scope, page)
        rows, page, pages, total = (data[key] for key in ('rows', 'page', 'pages', 'total'))

        def button(label, **data):
            return self._panel_button(update, label, **data)

        labels = {'all': '全部', 'pending': '挂单', 'position': '持仓'}
        buttons = [
            [
                button(('· ' if scope == key else '') + label, action='list', scope=key, page=0)
                for key, label in labels.items()
            ]
        ]
        buttons += [[button(row['label'], action='detail', id=row['id'])] for row in rows]
        navigation = []
        if page:
            navigation.append(button('上一页', action='list', scope=scope, page=page - 1))
        if page + 1 < pages:
            navigation.append(button('下一页', action='list', scope=scope, page=page + 1))
        navigation.append(button('刷新', action='list', scope=scope, page=page))
        buttons.append(navigation)
        text = f'订单管理 · {labels[scope]} {total} 笔 · {page + 1}/{pages}\n'
        text += '\n\n'.join(row['label'] + '\n' + profit_text(row) for row in rows)
        if not rows:
            text += '当前没有此类订单。新挂单可用 /forcelong 或 /forceshort。'
        amount = round(data['profit_total'], 4) or 0.0
        if data['profit_missing']:
            text += (
                f"\n\n汇总不完整：{data['profit_missing']} 笔盈亏待更新。"
                f"\n已取得小计：{amount:+.4f} USDT"
                f"（{data['positions'] - data['profit_missing']}/{data['positions']} 笔）"
            )
        else:
            text += f'\n\n全部持仓浮盈亏：{amount:+.4f} USDT'
        text += (
            '\n汇总含全部持仓（跨页、不受筛选影响）。'
            '\n按已记账余仓与最新参考价估算，含手续费口径，不含已实现止盈。'
            '\n查询时间 ' + data['checked'] + '\n点击下方订单管理，刷新可更新盈亏。'
        )
        if notice:
            text = notice + '\n\n' + text
        await self._panel_send(update, text, buttons)

    async def _panel_detail(self, update, ident, notice=''):
        try:
            data = await self._await_command(management.snapshot, self, ident)
        except ValueError:
            await self._panel_list(update, notice=notice or '该订单已结束，请选择当前订单。')
            return

        def button(label, **kw):
            return self._panel_button(update, label, **kw)

        def edit(label, field):
            current = data['plan'].get(field)
            if field.startswith('tp'):
                index = int(field[-1]) - 1
                targets = data['plan'].get('exit_points', [])
                current = targets[index] if index < len(targets) else None
            if field == 'price':
                current = data['orders'][0]['price']
            group = 'soft_sl' if field.startswith('soft') else 'extend_trigger'
            existing = data['plan'].get(group) is not None
            values = {k: data['plan'][k] for k in RULE_GROUPS[group][:-1]} if existing else {}
            return button(
                label,
                action='edit',
                id=ident,
                field=field,
                revision=data['revision'],
                current=current,
                existing=existing,
                values=values if field.endswith('_tf') else {},
            )

        plan = data['plan']
        lines = [
            f"#{ident} {data['pair']} {'空单' if data['short'] else '多单'} · {KINDS[data['kind']]}",
            f"持仓数量 {data['amount']:g} · 杠杆 {data['leverage']:g}倍",
        ]
        if data['amount']:
            lines.append(f"实际成交均价 {data['entry']:g}")
        for order in data['orders']:
            side = '入场' if order['side'] == ('sell' if data['short'] else 'buy') else '退出'
            lines.append(
                f"{side}挂单价 {order['price']:g} · 已成交 {order['filled']:g} / {order['amount']:g}"
            )
        if data['deadline']:
            deadline = datetime.fromtimestamp(data['deadline'], timezone.utc)
            lines.append(f'挂单到期 {deadline:%m-%d %H:%M:%S} UTC')
        if data['rate'] is not None:
            lines.append(f"最新平仓参考价 {data['rate']:g}")
        lines.append(profit_text(data))
        if data['profit_abs'] is not None:
            lines.append('按已记账余仓估算；收益率含杠杆，浮盈亏不含已实现止盈。')
        lines.append('查询时间 ' + data['checked'])
        lines.append(f"硬 SL {plan.get('stop_loss', '未设置')} · 有效保护价 {data['stop'] or '未确认'}")
        lines.append('有效保护价为策略目标；交易所保护状态见 /overview。')
        targets = plan.get('exit_points', [])
        for i, price in enumerate(targets):
            progress = (
                '已完成'
                if i < data['stage']
                else ('部分成交' if data['tp_filled'].get(str(i + 1), 0) else '待触发')
            )
            lines.append(f'TP{i + 1} {price:g} · {progress}')
        lines.append(describe_rules(plan, {'extension': data['extension']}))
        if plan.get('soft_sl') is None:
            lines.append('软止损：未设置')
        if plan.get('extend_trigger') is None:
            lines.append('延伸 TP3：未设置')
        buttons = []
        if data['fresh'] and not data['full_exit']:
            editable = data['kind'] != 'partial' and not any(
                o['side'] == ('buy' if data['short'] else 'sell') for o in data['orders']
            )
            if editable and plan:
                tp_buttons = [
                    edit(f'TP{i + 1}', f'tp{i + 1}')
                    for i in range(min(3, len(targets) + 1))
                    if i >= data['stage']
                    and not data['tp_filled'].get(str(i + 1), 0)
                    and not data['extension']
                    and not data['final_tp_started']
                    and (i < len(targets) or data['stage'] == 0)
                ]
                if tp_buttons:
                    buttons.append(tp_buttons)
                buttons.append([edit('硬止损 SL', 'stop_loss'), edit('软止损', 'soft_sl')])
                if not data['extension']:
                    buttons.append([edit('延伸触发价', 'extend_trigger')])
                    if plan.get('extend_trigger') is not None:
                        buttons.append([edit('延伸 TP3', 'extend_tp3'), edit('延伸周期', 'extend_tf')])
                if plan.get('soft_sl') is not None:
                    buttons.append([edit('软止损周期', 'soft_tf')])
                if data['kind'] == 'pending':
                    buttons.append([edit('开仓价', 'price')])
            if data['orders'] and all(
                o['side'] == ('sell' if data['short'] else 'buy') for o in data['orders']
            ):
                buttons.append(
                    [
                        button(
                            '撤销入场余量',
                            action='confirm',
                            id=ident,
                            operation='cancel',
                            revision=data['revision'],
                        )
                    ]
                )
            elif data['amount'] > 0 and not data['orders']:
                buttons.append(
                    [
                        button(
                            '快速全平',
                            action='confirm',
                            id=ident,
                            operation='close',
                            revision=data['revision'],
                        )
                    ]
                )
        if not data['fresh']:
            lines.append('\n' + data['error'] + '\n交易按钮暂不可用。')
        elif data['full_exit']:
            lines.append('已有全平意图，请核对 /order 与 /request，等待实际结果。')
        buttons.append([button('刷新', action='detail', id=ident), button('返回列表', action='list')])
        await self._panel_send(
            update, (notice + '\n\n' if notice else '') + '\n'.join(filter(None, lines)), buttons
        )

    async def _panel_prompt_value(self, update, draft):
        field = draft['field']
        text = f"#{draft['id']} 修改 {LABELS[field]}\n请回复这条消息，输入新价格；回复“取消”放弃。"
        if draft.get('current') is not None:
            text += f"\n当前值 {draft['current']}"
        if field in ('soft_sl', 'extend_trigger'):
            text += '\n回复“关闭”可关闭这组尚未生效的规则。'
        if field == 'price':
            text += '\n改价会撤旧单再重挂，保持原到期时间；输入后还需确认。'
        if draft.get('values'):
            text += '\n已填写：' + '，'.join(f'{LABELS[k]} {v:g}' for k, v in draft['values'].items())
        msg = await self._panel_send(update, text, prompt=True)
        edits = getattr(self, '_panel_edits', {})
        now = time.monotonic()
        edits = {k: v for k, v in edits.items() if v['expires'] > now}
        edits[self._panel_identity(update)] = dict(draft, prompt=msg.message_id, expires=now + 600)
        self._panel_edits = edits

    async def _panel_choose_tf(self, update, draft):
        choices = [
            self._panel_button(update, tf, action='timeframe', draft=draft, value=tf)
            for tf in TIMEFRAMES
        ]
        buttons = [choices[:2], choices[2:]]
        buttons.append([self._panel_button(update, '取消修改', action='detail', id=draft['id'])])
        values = '，'.join(f'{LABELS[k]} {v:g}' for k, v in draft['values'].items())
        await self._panel_send(update, f"#{draft['id']} {values}\n选择收盘周期后保存；尚未修改计划。", buttons)

    @authorized_only
    async def _orders(self, update, context):
        try:
            await self._panel_list(update)
        except Exception:
            await self._panel_send(update, '订单列表暂不可用，请稍后 /orders 重试。')

    @authorized_only
    async def _status_table(self, update, context):
        if update.callback_query:
            await update.callback_query.answer()
        await self._orders.__wrapped__(self, update, context)

    @authorized_only
    async def _panel_callback(self, update, context):
        await update.callback_query.answer()
        try:
            token = update.callback_query.data.removeprefix('om:')
            payload = self._panel_take(update, token, consume=True)
            action = payload['action']
            if action == 'list':
                getattr(self, '_panel_edits', {}).pop(self._panel_identity(update), None)
                await self._panel_list(update, payload.get('scope', 'all'), payload.get('page', 0))
            elif action == 'detail':
                getattr(self, '_panel_edits', {}).pop(self._panel_identity(update), None)
                await self._panel_detail(update, payload['id'])
            elif action == 'edit':
                if payload['field'].endswith('_tf'):
                    await self._panel_choose_tf(update, payload)
                else:
                    await self._panel_prompt_value(update, payload)
            elif action == 'timeframe':
                draft = payload['draft']
                changes = dict(draft['values'])
                changes['soft_tf' if 'soft_sl' in changes else 'extend_tf'] = payload['value']
                async with self._manual_order_lock:
                    result = await self._await_command(
                        management.edit_plan, self, draft['id'], draft['revision'], changes
                    )
                await self._panel_detail(update, draft['id'], result)
            elif action == 'confirm':
                data = await self._await_command(management.snapshot, self, payload['id'])
                if not data['fresh'] or data['revision'] != payload['revision']:
                    raise ValueError('订单状态已变化或未核实，请刷新详情。')
                name = '确认全平' if payload['operation'] == 'close' else '确认撤单'
                text = (
                    f"#{data['id']} {data['pair']} {'空单' if data['short'] else '多单'}\n"
                    f"持仓 {data['amount']:g}；入场挂单 {len(data['orders'])} 笔。\n"
                )
                text += (
                    '将按市价退出全部持仓，以实际成交为准。'
                    if payload['operation'] == 'close'
                    else '将撤销未成交入场余量，已有成交继续按原计划保护。'
                )
                await self._panel_send(
                    update,
                    text,
                    [
                        [self._panel_button(update, name, **dict(payload, action='do'))],
                        [self._panel_button(update, '返回详情', action='detail', id=payload['id'])],
                    ],
                )
            elif action == 'do':
                async with self._manual_order_lock:
                    if payload['operation'] == 'price':
                        row = await self._await_command(
                            management.replace_entry,
                            self,
                            payload['id'],
                            payload['revision'],
                            payload['value'],
                            token,
                        )
                        result = self._format_entry_request(row)
                        ident = row['result'].get('id', payload['id'])
                    else:
                        result = await self._await_command(
                            management.close_or_cancel,
                            self,
                            payload['id'],
                            payload['revision'],
                            payload['operation'],
                        )
                        ident = payload['id']
                await self._panel_detail(update, ident, result)
        except ValueError as exc:
            await self._panel_send(
                update, str(exc), [[self._panel_button(update, '返回列表', action='list')]]
            )
        except Exception as exc:
            logger.warning('Order panel operation unavailable (%s)', type(exc).__name__)
            await self._panel_send(
                update,
                '操作结果暂未确认，请 /orders 刷新并核对 /order、/request；不要重复提交。',
                [[self._panel_button(update, '返回列表', action='list')]],
            )

    @authorized_only
    async def _panel_reply(self, update, context):
        edits = getattr(self, '_panel_edits', {})
        key = self._panel_identity(update)
        draft = edits.get(key)
        reply = update.message.reply_to_message
        if not draft or not reply or reply.message_id != draft['prompt']:
            return
        edits.pop(key)
        if draft['expires'] <= time.monotonic():
            await self._panel_send(update, '输入已过期，未修改。请 /orders 重新选择。')
            return
        try:
            saving = False
            value = update.message.text.strip()
            if value in ('取消', 'cancel'):
                await self._panel_detail(update, draft['id'], '已取消修改。')
                return
            field = draft['field']
            if value in ('关闭', 'off') and field in RULE_GROUPS:
                changes = {k: None for k in RULE_GROUPS[field]}
            else:
                value = float(number(value, positive=True))
                values = dict(draft.get('values', {}), **{field: value})
                if field == 'extend_trigger' and not draft.get('existing'):
                    await self._panel_prompt_value(
                        update, dict(draft, field='extend_tp3', values=values)
                    )
                    return
                if field in ('soft_sl', 'extend_tp3') and not draft.get('existing'):
                    await self._panel_choose_tf(update, dict(draft, values=values))
                    return
                if field == 'price':
                    await self._panel_send(
                        update,
                        f"#{draft['id']} 新开仓价 {value:g}\n"
                        '确认后撤旧单、按原保证金和杠杆重挂，保留原到期时间。\n'
                        '如已有成交或结果不明，将停止重挂；新单仍须通过风险检查。',
                        [
                            [
                                self._panel_button(
                                    update,
                                    '确认改价重挂',
                                    action='do',
                                    operation='price',
                                    id=draft['id'],
                                    revision=draft['revision'],
                                    value=value,
                                )
                            ],
                            [self._panel_button(update, '取消修改', action='detail', id=draft['id'])],
                        ],
                    )
                    return
                changes = {field: value}
            saving = True
            async with self._manual_order_lock:
                result = await self._await_command(
                    management.edit_plan, self, draft['id'], draft['revision'], changes
                )
            await self._panel_detail(update, draft['id'], result)
        except ValueError as exc:
            if saving:
                await self._panel_detail(update, draft['id'], str(exc))
            else:
                await self._panel_send(update, str(exc))
                await self._panel_prompt_value(update, draft)
        except Exception as exc:
            logger.warning('Order panel edit unavailable (%s)', type(exc).__name__)
            await self._panel_send(update, '保存结果暂未确认，请 /orders 刷新核对，勿重复提交。')
