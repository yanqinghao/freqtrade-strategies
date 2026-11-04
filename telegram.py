"""Small Telegram extension on top of the pinned, unmodified Freqtrade runtime."""
import asyncio
import logging
import re
import time
from copy import deepcopy
from datetime import datetime, timezone
from functools import partial
from html import escape
from math import isfinite

from telegram import InlineKeyboardButton
from telegram.ext import CallbackQueryHandler, CommandHandler
from freqtrade.enums import SignalDirection
from freqtrade.persistence import Trade, Order
from freqtrade.rpc import RPCException
from freqtrade.rpc._telegram_base import Telegram as UpstreamTelegram, authorized_only

logger = logging.getLogger(__name__)


def entry_command(request):
    """Round-trip stored parameters without rounding prices or inventing omitted SL/TP."""

    def numeric(value):
        return (
            str(int(value))
            if float(value).is_integer() and abs(value) < 1e16
            else repr(float(value))
        )

    parts = [
        f"/force{request['direction']}",
        request['pair'],
        numeric(request['size']),
        numeric(request['leverage']),
    ]
    for field, name in (('stop_loss', 'sl'), ('entry_price', 'price')):
        if field in request:
            parts.append(f'{name}={numeric(request[field])}')
    if request.get('exit_points'):
        parts.append('tp=' + ','.join(numeric(value) for value in request['exit_points']))
    if 'ttl_seconds' in request:
        parts.append(f"ttl={request['ttl_seconds'] // 60}m")
    return ' '.join(parts)


def request_html(text):
    return '\n'.join(
        f'<code>{escape(line)}</code>'
        if line.startswith(('/forcelong ', '/forceshort ', '/preview '))
        else escape(line)
        for line in text.splitlines()
    )


def normalize_pair(value):
    value = value.upper()
    if '/' not in value:
        value = value[:-4] if value.endswith('USDT') else value
        value += '/USDT'
    if ':' not in value:
        value += ':USDT'
    return value


def positive(value, name):
    number = float(value)
    if not isfinite(number) or number <= 0:
        raise ValueError(f'{name} 必须是有限正数')
    return number


def ttl_seconds(value):
    match = re.fullmatch(r'([1-9]\d*)(m|h|d)', value.lower())
    if not match:
        raise ValueError('ttl 请使用整数分钟/小时/天，例如 30m、4h、1d')
    seconds = int(match[1]) * {'m': 60, 'h': 3600, 'd': 86400}[match[2]]
    if not 60 <= seconds <= 5 * 86400:
        raise ValueError('ttl 范围为1分钟至5天')
    return seconds


def parse_entry(args, direction):
    if len(args) < 3:
        raise ValueError(
            '用法：/forcelong 或 /forceshort BTC <保证金USDT> <杠杆> '
            'sl=<止损> [price=<限价>] [ttl=24h] [tp=<TP1,TP2,TP3>]'
        )
    result = {
        'pair': normalize_pair(args[0]),
        'direction': direction,
        'size': positive(args[1], '保证金'),
        'leverage': positive(args[2], '杠杆'),
    }
    if result['leverage'] < 1:
        raise ValueError('杠杆必须至少为1')
    tokens = args[3:]
    if tokens and all('=' not in t for t in tokens):
        if len(tokens) not in (4, 5):
            raise ValueError('位置参数仅支持 TP1 TP2 TP3 SL [单个限价]')
        values = dict(zip(('tp1', 'tp2', 'tp3', 'sl', 'price'), tokens))
    else:
        values = {}
        for token in tokens:
            if '=' not in token:
                raise ValueError('请使用 key=value 参数')
            key, value = token.split('=', 1)
            if key not in ('sl', 'price', 'tp', 'tp1', 'tp2', 'tp3', 'ttl') or key in values:
                raise ValueError(f'不支持或重复的参数：{key}；不支持自动分段入场')
            values[key] = value
    if 'tp' in values and any(k in values for k in ('tp1', 'tp2', 'tp3')):
        raise ValueError('tp 和 tp1/tp2/tp3 不能混用')
    for key, field in (('sl', 'stop_loss'), ('price', 'entry_price')):
        if key in values:
            result[field] = positive(values[key], key)
    tps = (
        values['tp'].split(',')
        if 'tp' in values
        else [values[k] for k in ('tp1', 'tp2', 'tp3') if k in values]
    )
    if len(tps) > 3:
        raise ValueError('最多支持三个止盈点')
    if tps:
        result['exit_points'] = sorted(
            set(positive(p, 'TP') for p in tps), reverse=direction == 'short'
        )
    if 'ttl' in values and 'entry_price' not in result:
        raise ValueError('ttl 仅适用于带 price 的限价单')
    if 'entry_price' in result:
        result['ttl_seconds'] = ttl_seconds(values.get('ttl', '24h'))
    return result


class Telegram(UpstreamTelegram):
    def __init__(self, rpc, config):
        self._manual_order_lock = asyncio.Lock()
        rpc._freqtrade.strategy.execution_bot = rpc._freqtrade
        super().__init__(rpc, config)

    def _init_keyboard(self):
        self._keyboard = [
            ['/overview', '/preview', '/request'],
            ['/status table', '/balance', '/manual'],
            ['/forcelong', '/forceshort', '/forceexit'],
            ['/setmanual', '/close', '/coo'],
            ['/monitor', '/profit', '/help'],
        ]

    def _assistant_handlers(self):
        pause_entries = getattr(self, '_pause', None) or self._stopentry
        commands = {
            'status': self._status,
            'balance': self._balance,
            'profit': self._profit,
            'start': self._start,
            'stop': self._stop,
            'stopentry': pause_entries,
            'pause': pause_entries,
            'forcelong': partial(self._force_enter, order_side=SignalDirection.LONG),
            'forceshort': partial(self._force_enter, order_side=SignalDirection.SHORT),
            'forceexit': self._force_exit,
            'fx': self._force_exit,
            'close': self._close,
            'manual': self._manual,
            'request': self._request,
            'preview': self._preview,
            'overview': self._overview,
            'setmanual': self._set_manual,
            'monitor': self._monitor,
            'coo': self._cancel_open_order,
            'cancel_open_order': self._cancel_open_order,
            'order': self._order,
            'trades': self._trades,
            'performance': self._performance,
            'daily': self._daily,
            'weekly': self._weekly,
            'monthly': self._monthly,
            'count': self._count,
            'health': self._health,
            'logs': self._logs,
            'reload_config': self._reload_config,
            'show_config': self._show_config,
            'help': self._help,
            'version': self._version,
        }
        handlers = [
            CommandHandler(
                name,
                callback,
                block=name
                not in {
                    'monitor',
                    'request',
                    'preview',
                    'overview',
                    'forcelong',
                    'forceshort',
                    'setmanual',
                    'close',
                    'forceexit',
                    'fx',
                    'coo',
                    'cancel_open_order',
                },
            )
            for name, callback in commands.items()
        ]
        for name in (
            'status_table',
            'balance',
            'profit',
            'performance',
            'daily',
            'weekly',
            'monthly',
            'count',
        ):
            handlers.append(
                CallbackQueryHandler(getattr(self, '_' + name), pattern=f'^update_{name}$')
            )
        handlers.append(
            CallbackQueryHandler(
                self._preview_history, pattern=r'^preview_request:[0-9a-f]{24}$', block=False
            )
        )
        return handlers

    def _init(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._app = self._init_telegram_app()
        for handler in self._assistant_handlers():
            self._app.add_handler(handler)
        self._loop.run_until_complete(self._startup_with_alerts())

    async def _startup_with_alerts(self):
        worker = asyncio.create_task(self._alert_worker())
        try:
            await self._startup_telegram()
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

    async def _alert_worker(self):
        while True:
            await asyncio.sleep(10)
            if self._app.running:
                try:
                    await self._deliver_alerts_once()
                except Exception as error:
                    logger.warning('Critical alert queue unavailable (%s)', type(error).__name__)

    async def _deliver_alerts_once(self, now=None):
        stamp = time.time() if now is None else now
        queue = self._rpc._freqtrade.strategy.alert_queue
        for row in queue.pending(now=stamp, limit=3):
            updated = datetime.fromtimestamp(row['updated'], timezone.utc)
            message = row['message'] + f'\n状态时间：{updated:%m-%d %H:%M:%S} UTC'
            if stamp - row['updated'] > 60:
                message += '\n延迟送达，请用 /overview 核对当前状态。'
            try:
                await self._app.bot.send_message(
                    self._config['telegram']['chat_id'],
                    text=message,
                    parse_mode=None,
                    message_thread_id=self._config['telegram'].get('topic_id'),
                )
            except Exception as error:
                logger.warning('Critical alert remains pending (%s)', type(error).__name__)
                delay = getattr(error, 'retry_after', 60)
                if hasattr(delay, 'total_seconds'):
                    delay = delay.total_seconds()
                queue.defer(row, stamp + max(60, float(delay)))
            else:
                queue.acknowledge(row)

    async def _await_command(self, function, *args):
        from freqtrade.enums import State

        bot = self._rpc._freqtrade
        if bot.state not in (State.RUNNING, State.PAUSED):
            raise ValueError('机器人尚未处理交易，请先 /start 或 /pause')
        future = bot.strategy.submit_command(function, *args)
        wrapped = asyncio.wrap_future(future)
        try:
            return await asyncio.wait_for(asyncio.shield(wrapped), timeout=30)
        except TimeoutError:
            if future.cancel():
                raise ValueError('请求尚未执行且等待超时，已取消，可在检查 /health 后重试')
            # A running request cannot safely be retried until its status is known.
            wrapped.add_done_callback(
                lambda task: task.exception() if not task.cancelled() else None
            )
            raise RPCException('执行结果暂未确认，请先查看 /status 与 /order，勿直接重复指令')

    @authorized_only
    async def _force_exit(self, update, context):
        args = context.args or []
        if len(args) != 1 or (args[0] != 'all' and not args[0].isdigit()):
            await self._send_msg('用法：/forceexit <交易ID或all>', parse_mode=None)
            return
        try:
            result = await self._await_command(self._rpc._rpc_force_exit, args[0])
            await self._send_msg(result['result'], parse_mode=None)
        except (ValueError, RPCException) as exc:
            await self._send_msg(str(exc), parse_mode=None)

    @authorized_only
    async def _cancel_open_order(self, update, context):
        args = context.args or []
        if len(args) != 1 or not args[0].isdigit():
            await self._send_msg('用法：/coo <交易ID>', parse_mode=None)
            return
        try:
            result = await self._await_command(self._rpc._rpc_cancel_open_order, int(args[0]))
            await self._send_msg(str(result), parse_mode=None)
        except (ValueError, RPCException) as exc:
            await self._send_msg(str(exc), parse_mode=None)

    @authorized_only
    async def _force_enter(self, update, context, order_side):
        request_ref = None
        try:
            request = parse_entry(context.args or [], order_side.value)
            request_ref = self._rpc._freqtrade.strategy.entry_requests.reference(
                update.effective_chat.id, update.effective_message.message_id
            )
            async with self._manual_order_lock:
                row = await self._await_command(self._execute_entry_request, request, request_ref)
            await self._send_msg(self._format_entry_request(row), parse_mode=None)
        except (ValueError, RPCException) as exc:
            suffix = f'\n请求 {request_ref}，可用 /request {request_ref} 查询。' if request_ref else ''
            await self._send_msg(str(exc) + suffix, parse_mode=None)
        except Exception as exc:
            logger.warning('Entry request unavailable (%s)', type(exc).__name__)
            ref_label = request_ref or '未建立'
            await self._send_msg(
                f'请求 {ref_label} 状态暂无法确认，请先用 /request、/status 和 /order 核实，勿直接重复下单。',
                parse_mode=None,
            )

    def _overview_snapshot(self):
        bot = self._rpc._freqtrade
        state = getattr(bot.state, 'name', str(bot.state))
        return bot.strategy.overview(state, datetime.now(timezone.utc))

    @authorized_only
    async def _overview(self, update, context):
        if context.args:
            await self._send_msg('用法：/overview，查看助手持仓、挂单、TP和止损核验快照。', parse_mode=None)
            return
        try:
            messages = await self._await_command(self._overview_snapshot)
            for message in messages:
                await self._send_msg(message, parse_mode=None)
        except (ValueError, RPCException) as error:
            await self._send_msg(str(error), parse_mode=None)
        except Exception as error:
            logger.warning('Trade overview unavailable (%s)', type(error).__name__)
            await self._send_msg('总览暂不可用，本命令未改变持仓或订单，请用 /status、/order 核对。', parse_mode=None)

    def _preview_entry(self, request):
        trade = Trade.get_trades([Trade.is_open.is_(True), Trade.pair == request['pair']]).first()
        result = self._rpc._freqtrade.strategy.preview_entry(request, trade)
        if getattr(self._rpc._freqtrade.state, 'name', '') != 'RUNNING':
            result['reasons'].append('机器人当前未允许新入场')
            result['allowed'] = False
        return result

    @staticmethod
    def _format_risk_preview(row):
        status = '本地风险预检通过' if row['allowed'] else '当前会被拦截'
        scope = '该币种加仓后' if row.get('risk_complete', True) else '已能计算部分'
        loss_text = (
            f"估算止损亏损 {row['estimated_loss']:.4f}U"
            if row['estimated_loss'] is not None
            else '整体估算不可用（另有待记账成交或入场余量）'
        )
        text = (
            f"风险预览（未下单）：{row['pair']} {row['direction']}\n"
            f"请求保证金 {row['margin']:g}U，杠杆 {row['leverage']:g}倍\n"
            f"参考入场 {row['reference_price']:g}，预计数量 {row['amount']:g}\n"
            f"名义仓位 {row['notional']:g}U，保护止损 {row['stop_price']:g}，"
            f"距离 {row['stop_distance_pct']:.3f}%\n"
            f"{scope}价格损失 {row['price_loss']:.4f}U＋费用参考 {row['fees']:.4f}U\n"
            f'{loss_text}\n'
            f"助手敞口：持仓 {row['exposure']['position_notional']:g}U＋"
            f"挂单 {row['exposure']['pending_notional']:g}U；"
            f"新增后约 {row['projected_total_notional']:g}U\n"
        )
        limits = row['limits']
        if limits:
            text += (
                f"上限：保证金 {limits['max_margin_usdt']:g}U／杠杆 {limits['max_leverage']:g}倍／"
                f"总敞口 {limits['max_total_notional_usdt']:g}U／"
                f"估算亏损 {limits['max_stop_loss_usdt']:g}U\n"
            )
        else:
            text += '额度限制未启用，不能把本预览当作额度保护。\n'
        text += status + ('：' + '；'.join(row['reasons']) if row['reasons'] else '')
        return text + '\n实际提交会重新核对；余额和交易所规则仍适用。估算不含未来滑点、跳价与资金费，不保证最终亏损上限。'

    @authorized_only
    async def _preview(self, update, context):
        await self._preview_args(context.args or [])

    @authorized_only
    async def _preview_history(self, update, context):
        query = update.callback_query
        await query.answer()
        await self._preview_args([query.data.removeprefix('preview_request:')])

    async def _preview_args(self, args):
        try:
            source = None
            if len(args) == 1 and re.fullmatch('[0-9a-f]{24}', args[0]):
                source = args[0]
                row = await asyncio.to_thread(
                    self._rpc._freqtrade.strategy.entry_requests.get, source
                )
                if row is None:
                    raise ValueError('未找到历史请求，请用 /request 重新选择。')
                saved = row['request']
                # Reuse the normal parser; never reconcile, reserve or mutate the old request.
                request = parse_entry(entry_command(saved).split()[1:], saved['direction'])
            elif len(args) >= 4 and args[0] in ('long', 'short'):
                request = parse_entry(args[1:], args[0])
            else:
                raise ValueError(
                    '风险预览（只读，不下单）\n'
                    '/request 选择历史单的风险预览按钮，或 /preview <请求编号> 带入历史参数。\n'
                    '用法：/preview long|short <币种> <保证金USDT> <杠杆> '
                    'sl=<止损> [price=<限价>] [tp=<TP1,TP2,TP3>] [ttl=24h]\n'
                    'long 做多，short 做空；不填price按市价估算。'
                    '限价单默认24h有效，ttl仅用于限价单。\n'
                    '示例（价格仅演示格式，请替换）：\n'
                    '/preview long BTC 10 20 sl=90000 price=95000\n'
                    '/preview short ETH 10 20 sl=3200 price=3100\n'
                    '预览不会开仓；实际开仓请使用 /forcelong 或 /forceshort。'
                )
            await self._show_preview(request, source)
        except (ValueError, RPCException) as error:
            await self._send_msg(str(error), parse_mode=None)
        except Exception as error:
            logger.warning('Entry preview unavailable (%s)', type(error).__name__)
            await self._send_msg('风险预览暂不可用，本命令未下单。', parse_mode=None)

    async def _show_preview(self, request, source=None):
        command = entry_command(request)
        draft = '/preview ' + request['direction'] + ' ' + command.split(' ', 1)[1]
        text = f'历史参数来自请求 {source}；按当前持仓和额度重新估算。\n' if source else ''
        text += '复制下面命令，修改金额、杠杆、price、sl、tp或ttl后发送，即可重新预览：\n' + draft
        if 'stop_loss' not in request:
            text += '\n未填写SL：有同向持仓时按当前计划估算；新开仓必须补 sl=...。'
        try:
            row = await self._await_command(self._preview_entry, request)
            text += '\n\n' + self._format_risk_preview(row)
            if row['allowed']:
                text += (
                    '\n\n手动开单指令（自行核对后发送）：\n' + command + '\n发送会创建新开仓/加仓请求；同方向已有持仓会申请加仓，实际提交重新检查。'
                )
        except (ValueError, RPCException) as error:
            text += '\n\n预览未通过（未下单）：' + str(error)
        except Exception as error:
            logger.warning('Entry preview unavailable (%s)', type(error).__name__)
            text += '\n\n风险数据暂不可用，保留上面的预览命令供修改重试；未下单。'
        await self._send_request_text(text)

    @staticmethod
    def _preview_buttons(rows):
        return [
            [
                InlineKeyboardButton(
                    f"风险预览 {i} · {row['request']['pair'].split('/')[0][:12]} "
                    + ('多' if row['request']['direction'] == 'long' else '空'),
                    callback_data='preview_request:' + row['ref'],
                )
            ]
            for i, row in enumerate(rows, 1)
        ] or None

    def _entry_request_status(self, ref):
        store = self._rpc._freqtrade.strategy.entry_requests
        row = store.get(ref)
        if row is None:
            raise ValueError('未找到请求；可能尚未进入交易循环或未成功建立记录')
        if row['state'] in ('submitting', 'unknown'):
            orders = (
                Trade.session.query(Order).filter(Order.ft_order_tag.endswith('|req=' + ref)).all()
            )
            orders = [
                o
                for o in orders
                if o.ft_pair == row['request']['pair']
                and o.ft_order_side == ('sell' if row['request']['direction'] == 'short' else 'buy')
            ]
            if orders:
                trade = orders[0].trade
                return store.set_status(
                    ref,
                    'submitted',
                    {
                        'id': orders[0].ft_trade_id,
                        'leverage': trade.leverage if trade else None,
                        'order_ids': [o.order_id for o in orders],
                    },
                )
            return store.set_status(
                ref, 'unknown', {'message': '提交结果尚未确认，该币种暂停新增入场；核实交易所后可显式 resolve，旧请求不会重发。'}
            )
        if row['state'] == 'reserved':
            return store.set_status(ref, 'rejected', {'message': '请求在提交前中断，未自动重放；请核实后发送新指令。'})
        return row

    def _execute_entry_request(self, request, ref):
        store = self._rpc._freqtrade.strategy.entry_requests
        row, fresh = store.reserve(ref, request)
        if not fresh:
            return self._entry_request_status(ref)
        try:
            for other in store.unresolved(request['pair']):
                if self._entry_request_status(other['ref'])['state'] == 'unknown':
                    raise ValueError(f"该币种请求 {other['ref']} 尚待核实，本次未提交；请先 /request 查询")
            # Native eligibility (especially PAUSED) must fail before plan changes.
            self._rpc._force_entry_validations(
                request['pair'], SignalDirection(request['direction'])
            )
            result = self._submit_entry(request, request_ref=ref)
            return store.set_status(ref, 'submitted', result)
        except Exception as exc:
            Trade.rollback()
            state = store.get(ref)['state']
            if state == 'reserved' or isinstance(exc, self._rpc._freqtrade.strategy.risk_error):
                message = str(exc) if isinstance(exc, ValueError) else '提交前检查失败，未下单，请核对运行状态。'
                return store.set_status(ref, 'rejected', {'message': message})
            logger.warning('Entry submission requires reconciliation (%s)', type(exc).__name__)
            store.set_status(ref, 'unknown', {'message': '交易所提交结果待核实，不会自动重下。'})
            return self._entry_request_status(ref)

    def _resolve_entry_request(self, ref):
        row = self._entry_request_status(ref)
        if row['state'] != 'unknown':
            return row
        pair = row['request']['pair']
        trade = Trade.get_trades([Trade.is_open.is_(True), Trade.pair == pair]).first()
        self._rpc._freqtrade.strategy.ensure_entry_position(pair, trade)
        return self._rpc._freqtrade.strategy.entry_requests.set_status(
            ref, 'resolved', {'message': '已按你的指令核对当前仓位和挂单并解除阻塞；旧请求不会重新提交，新入场请发新指令。'}
        )

    @staticmethod
    def _format_entry_request(row, compact=False):
        states = {
            'reserved': '待执行',
            'submitting': '提交待核实',
            'submitted': '已提交',
            'rejected': '未提交/已拒绝',
            'unknown': '结果待核实',
            'resolved': '已核实解除',
        }
        request, result = row['request'], row['result']
        created = datetime.fromtimestamp(row['created'], timezone.utc)
        command = entry_command(request)
        reuse = f'\n参数还原（复制后修改）：\n{command}'
        if 'stop_loss' not in request:
            reuse += '\n原请求未填写SL：当时沿用已有计划；新开仓请补 sl=...。'
        if compact:
            return (
                f"{created:%m-%d %H:%M} UTC · {states[row['state']]}\n" f"编号 {row['ref']}" + reuse
            )
        text = f"请求 {row['ref']}：{states[row['state']]}\n{request['pair']} {request['direction']}\n"
        text += f'创建时间：{created:%Y-%m-%d %H:%M:%S} UTC\n'
        if row['state'] == 'submitted':
            text += f"交易 #{result['id']}，关联订单 {', '.join(result['order_ids'])}。\n"
            leverage = result.get('leverage')
            leverage_text = f'{leverage:g}x' if leverage is not None else '未核实'
            text += f"请求保证金 {request['size']:g} USDT，实际杠杆 {leverage_text}。\n"
            if 'ttl_seconds' in request:
                text += f"限价有效期 {request['ttl_seconds'] // 60} 分钟，从订单创建时计算。\n"
            text += '本请求不会重复下单；实际成交/退出以 /order、/status 和 /trades 为准。'
        else:
            text += result.get('message', '请稍后查询。')
        return text + reuse + '\n重新发送是新请求；同方向已有持仓时会申请加仓。'

    @authorized_only
    async def _request(self, update, context):
        args = context.args or []
        if not args or args[0] == 'page':
            try:
                page = 1 if not args else int(args[1]) if len(args) == 2 else 0
                store = self._rpc._freqtrade.strategy.entry_requests
                rows = await asyncio.to_thread(store.recent, page)
                text = f'开单历史 · 第{page}页\n'
                text += '\n\n'.join(
                    f'{i}. ' + self._format_entry_request(row, compact=True)
                    for i, row in enumerate(rows, 1)
                )
                if not rows:
                    text += '这一页暂无开单请求记录。'
                text += (
                    '\n\n/request <请求编号> 查看详情\n'
                    f'/request page {page + 1} 下一页'
                    + (f' · /request page {page - 1} 上一页' if page > 1 else '')
                    + '\n复制命令修改后重新发送，会成为新开仓/加仓请求；先核对当前持仓。'
                )
                await self._send_request_text(text, keyboard=self._preview_buttons(rows))
            except (ValueError, IndexError):
                await self._send_msg('用法：/request 或 /request page <正整数页码>', parse_mode=None)
            except Exception as error:
                logger.warning('Entry history unavailable (%s)', type(error).__name__)
                await self._send_msg('开单历史暂不可用，本命令未下单。', parse_mode=None)
            return
        if len(args) > 2 or (len(args) == 2 and args[1] != 'resolve'):
            await self._send_msg(
                '用法：/request [请求编号]；核实后解除未知阻塞：/request <编号> resolve', parse_mode=None
            )
            return
        try:
            function = self._resolve_entry_request if len(args) == 2 else self._entry_request_status
            row = await self._await_command(function, args[0])
            await self._send_request_text(
                self._format_entry_request(row), keyboard=self._preview_buttons([row])
            )
        except (ValueError, RPCException) as exc:
            await self._send_msg(str(exc), parse_mode=None)
        except Exception as exc:
            logger.warning('Entry request lookup unavailable (%s)', type(exc).__name__)
            await self._send_msg('请求记录暂不可用，请核对 /order、/status 与交易所；本命令未提交新订单。', parse_mode=None)

    async def _send_request_text(self, text, keyboard=None):
        # 1500 Unicode code points fit even when every character uses two UTF-16 units.
        chunks, current = [], ''
        for line in text.splitlines():
            if len(line) > 1500 and line.startswith(('/forcelong ', '/forceshort ', '/preview ')):
                line = '原命令过长，无法提供完整复制块；请核对原消息参数。'
            for part in [line[i : i + 1500] for i in range(0, len(line), 1500)] or ['']:
                if len(current) + len(part) + 1 > 1500:
                    chunks.append(current)
                    current = ''
                current += ('\n' if current else '') + part
        if current:
            chunks.append(current)
        for i, chunk in enumerate(chunks):
            await self._send_msg(
                request_html(chunk),
                parse_mode='HTML',
                keyboard=keyboard if i == len(chunks) - 1 else None,
            )

    def _submit_entry(self, request, request_ref=None):
        pair = request['pair']
        bot = self._rpc._freqtrade
        strategy = bot.strategy
        existing = Trade.get_trades([Trade.is_open.is_(True), Trade.pair == pair]).first()
        if existing:
            if strategy._execution(existing).get('full_exit'):
                raise ValueError('该持仓正在全平兜底处理中，请确认退出完成后再开仓')
            if existing.is_short != (request['direction'] == 'short'):
                raise ValueError('同一币种只允许单方向持仓，请先平掉原方向')
            if existing.has_open_orders:
                raise ValueError('该持仓尚有未完成订单，请先等待成交或撤单')
            if request['leverage'] != existing.leverage:
                raise ValueError('加仓必须沿用现有持仓杠杆')
        strategy.ensure_entry_position(pair, existing)
        strategy.ensure_entry_protection(existing)
        old = deepcopy(strategy.manual_open.get(pair))
        plan = deepcopy(strategy._plan(existing) if existing else {})
        if not existing:
            plan.pop('trade_id', None)
        plan.update(
            {k: v for k, v in request.items() if k not in ('pair', 'entry_price', 'ttl_seconds')}
        )
        plan.setdefault('exit_points', [])
        if not plan.get('stop_loss'):
            raise ValueError('开仓必须提供 sl=<止损价>；本助手不自动生成止损')
        try:
            rate = request.get('entry_price') or bot.exchange.get_rate(
                pair, side='entry', is_short=request['direction'] == 'short', refresh=True
            )
        except Exception:
            raise ValueError('无法取得入场参考价格，本次未提交') from None
        sl = float(plan['stop_loss'])
        if sl <= rate if request['direction'] == 'short' else sl >= rate:
            raise ValueError('止损必须位于当前入场价的亏损方向')
        if not existing and any(
            (p >= rate if request['direction'] == 'short' else p <= rate)
            for p in plan['exit_points']
        ):
            raise ValueError('止盈必须位于入场价的盈利方向')
        strategy.enforce_entry_risk(request, existing, plan, rate)
        if request_ref:
            # Account queries can take time while a Telegram pause arrives.
            self._rpc._force_entry_validations(pair, SignalDirection(request['direction']))
            strategy.entry_requests.set_status(request_ref, 'submitting', {})
        strategy.set_manual_plan(pair, plan)
        try:
            with strategy.risk_request(request_ref):
                trade = self._rpc._rpc_force_entry(
                    pair,
                    request.get('entry_price'),
                    order_side=SignalDirection(request['direction']),
                    order_type='limit' if request.get('entry_price') is not None else 'market',
                    stake_amount=request['size'],
                    leverage=request['leverage'],
                    enter_tag='manual_'
                    + request['direction']
                    + ('|ttl=' + str(request['ttl_seconds']) if 'entry_price' in request else '')
                    + ('|req=' + request_ref if request_ref else ''),
                )
            if trade is None:
                raise RPCException('未创建订单')
        except (ValueError, RPCException) as exc:
            if request_ref and not isinstance(exc, strategy.risk_error):
                raise
            if old is None:
                strategy.delete_manual_plan(pair)
            else:
                strategy.set_manual_plan(pair, old)
            raise
        bound = deepcopy(strategy.manual_open.get(pair) or plan)
        bound['trade_id'] = trade.id
        strategy.set_manual_plan(pair, bound)
        order_ids = [
            o.order_id
            for o in trade.orders
            if request_ref and (o.ft_order_tag or '').endswith('|req=' + request_ref)
        ]
        if request_ref and not order_ids:
            raise RPCException('订单关联尚未确认')
        return {'id': trade.id, 'leverage': trade.leverage, 'order_ids': order_ids}

    @authorized_only
    async def _manual(self, update, context):
        plans = self._rpc._freqtrade.strategy.manual_open
        if not plans:
            await self._send_msg('当前没有手动交易计划。', parse_mode=None)
        for pair, plan in list(plans.items()):
            await self._send_msg(
                f"{pair} {plan.get('direction')}\n"
                f"SL：{plan.get('stop_loss')}\nTP：{plan.get('exit_points', [])}\n"
                '实际持仓与移动保护价请查看 /status。',
                parse_mode=None,
            )

    def _update_plan(self, ident, tokens):
        with self._rpc._freqtrade._exit_lock:
            filters = [Trade.is_open.is_(True)]
            filters.append(
                Trade.id == int(ident) if ident.isdigit() else Trade.pair == normalize_pair(ident)
            )
            trade = Trade.get_trades(filters).first()
            if not trade:
                raise ValueError('未找到该持仓')
            direction = 'short' if trade.is_short else 'long'
            if tokens and tokens[0] in ('long', 'short'):
                if tokens[0] != direction:
                    raise ValueError('不能通过修改计划改变持仓方向')
                tokens = tokens[1:]
            if any(t.startswith('price=') for t in tokens):
                raise ValueError('/setmanual 不修改入场价，请使用开仓指令提交新订单')
            request = parse_entry(
                [trade.pair, str(trade.stake_amount), str(trade.leverage)] + tokens, direction
            )
            strategy = self._rpc._freqtrade.strategy
            plan = deepcopy(strategy.manual_open.get(trade.pair) or strategy._plan(trade))
            if trade.has_open_orders and request.get(
                'exit_points', plan.get('exit_points')
            ) != plan.get('exit_points'):
                raise ValueError('仍有未完成订单，请先撤单或等待成交再修改TP；可以单独收紧SL')
            plan.update({k: v for k, v in request.items() if k not in ('pair', 'entry_price')})
            if not plan.get('stop_loss'):
                raise ValueError('请提供 sl=<止损价>')
            plan['trade_id'] = trade.id
            strategy.set_manual_plan(trade.pair, plan)
            return trade.pair

    @authorized_only
    async def _set_manual(self, update, context):
        args = context.args or []
        if len(args) < 2:
            await self._send_msg(
                '用法：/setmanual <交易ID或币种> sl=<止损> [tp=<TP1,TP2,TP3>]；也可只修改TP。', parse_mode=None
            )
            return
        try:
            async with self._manual_order_lock:
                pair = await self._await_command(self._update_plan, args[0], args[1:])
            await self._send_msg(f'{pair} 计划已更新；保护止损只向盈利方向收紧。', parse_mode=None)
        except (ValueError, RPCException) as exc:
            await self._send_msg(str(exc), parse_mode=None)

    def _close_position(self, ident, percentage):
        trade = Trade.get_trades([Trade.is_open.is_(True), Trade.id == int(ident)]).first()
        if not trade:
            raise ValueError('未找到该持仓')
        bot = self._rpc._freqtrade
        rate = bot.exchange.get_rate(trade.pair, side='exit', is_short=trade.is_short, refresh=True)
        requested = (
            trade.amount
            if bot.strategy._execution(trade).get('full_exit')
            else trade.amount * percentage / 100
        )
        amount, reason = bot.strategy.exit_decision(trade, rate, requested, manual=True)
        if reason:
            bot.strategy.request_full_exit(trade, reason, blocked=amount is None)
            if amount is None:
                raise ValueError('整仓也暂不满足平仓规则，已保留记录并请求核验止损，请检查交易所')
        if amount >= trade.amount:
            from datetime import datetime, timezone

            bot.strategy.mark_full_exit_attempt(trade, datetime.now(timezone.utc))
            return self._rpc._rpc_force_exit(ident, ordertype='market')
        return self._rpc._rpc_force_exit(ident, ordertype='market', amount=amount)

    @authorized_only
    async def _close(self, update, context):
        try:
            args = context.args or []
            if len(args) not in (1, 2) or not args[0].isdigit():
                raise ValueError('用法：/close <交易ID> [平仓百分比，默认100]')
            percentage = positive(args[1].removesuffix('%'), '平仓比例') if len(args) == 2 else 100
            if percentage > 100:
                raise ValueError('平仓比例不能大于100%')
            result = await self._await_command(self._close_position, args[0], percentage)
            await self._send_msg(result['result'], parse_mode=None)
        except (ValueError, RPCException) as exc:
            await self._send_msg(str(exc), parse_mode=None)

    @authorized_only
    async def _monitor(self, update, context):
        reports = await asyncio.get_running_loop().run_in_executor(
            None, self._rpc._freqtrade.strategy.get_market_reports
        )
        for report in reports:
            await self._send_msg(report, parse_mode=None)

    @authorized_only
    async def _help(self, update, context):
        await self._send_msg(
            '手动交易助手\n\n'
            '底部键盘可直接选择命令；点击 /preview 查看参数说明和示例。\n'
            '/forcelong BTC 100 3 sl=90000 price=95000 tp=98000,100000,105000\n'
            '限价单默认24h有效，可填 ttl=4h 或 ttl=2d（1m至5d）；市价单不填ttl。\n'
            '到期撤单依赖机器人在线和网络；检测到部分成交后撤余量，优先保护已成交仓位。\n'
            '加仓前须实时确认交易所止损；未确认时拒绝加仓，平仓照常。\n'
            '新入场前核对同币种普通/条件挂单；只减仓或整仓退出单可保留，不自动撤外部单。\n'
            '同一开单消息重放只返回原请求；主动发送新消息视为新开仓/加仓请求。\n'
            '/preview long|short BTC <保证金> <杠杆> sl=... [price=...] [tp=...] [ttl=24h] 只读预览\n'
            '/overview 汇总助手持仓、挂单TTL、TP及止损核验快照\n'
            '风险上限覆盖保证金、杠杆、助手总敞口和该币种加仓后估算止损亏损；超限拒绝，不自动缩量。\n'
            '/request 历史开单、复制命令与风险预览按钮；/request page 2 翻页\n'
            '/preview <请求编号> 带入历史参数，复制预览命令修改后再算；预览不会下单\n'
            '/request <编号> resolve 核实后解除未知请求阻塞（不重发旧请求）\n'
            '/forceshort BTC 100 3 sl=105000 tp=98000,95000,90000\n'
            '金额为保证金USDT，后接杠杆；不填price则市价，只接受单个入场价。以上数字仅示范格式。\n'
            '同方向已有持仓时，同一命令用于明确加仓，杠杆须与原仓一致。\n\n'
            '/setmanual <ID或币种> sl=... tp=... 修改持仓计划\n'
            '/manual 查看手动计划；/status table 查看实际仓位与移动止损\n'
            '/forceexit <ID或all> 全平；/close <ID> 50 部分平仓\n'
            '/coo <ID> 撤销未成交订单\n'
            '/monitor 查看三币详细报告；每小时自动一条简报，突破/跌破关键位即时提醒\n'
            '/balance /profit /order /trades /performance 查询账户与成交\n'
            '/start /stopentry /stop /reload_config /health /logs 运行控制\n\n'
            'TP1平初始仓位30%，TP2平剩余50%，TP3清仓；两个TP各平50%，一个TP全平。\n'
            '部分止盈实际成交后，保护位依次移到含交易费的成本价、TP1；空头方向相反。\n'
            '关闭指标开仓、到价补仓、自动选方向、回测和AI调用。监控参考点位不触发交易。',
            parse_mode=None,
        )
