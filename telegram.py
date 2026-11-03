"""Small Telegram extension on top of the pinned, unmodified Freqtrade runtime."""
import asyncio
import logging
from copy import deepcopy
from functools import partial
from math import isfinite

from telegram.ext import CallbackQueryHandler, CommandHandler
from freqtrade.enums import SignalDirection
from freqtrade.persistence import Trade
from freqtrade.rpc import RPCException
from freqtrade.rpc._telegram_base import Telegram as UpstreamTelegram, authorized_only

logger = logging.getLogger(__name__)


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


def parse_entry(args, direction):
    if len(args) < 3:
        raise ValueError(
            '用法：/forcelong 或 /forceshort BTC <保证金USDT> <杠杆> sl=<止损> [price=<限价>] [tp=<TP1,TP2,TP3>]'
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
            if key not in ('sl', 'price', 'tp', 'tp1', 'tp2', 'tp3') or key in values:
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
    return result


class Telegram(UpstreamTelegram):
    def __init__(self, rpc, config):
        self._manual_order_lock = asyncio.Lock()
        rpc._freqtrade.strategy.execution_bot = rpc._freqtrade
        super().__init__(rpc, config)

    def _init_keyboard(self):
        self._keyboard = [
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
        return handlers

    def _init(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._app = self._init_telegram_app()
        for handler in self._assistant_handlers():
            self._app.add_handler(handler)
        self._loop.run_until_complete(self._startup_telegram())

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
        try:
            request = parse_entry(context.args or [], order_side.value)
            async with self._manual_order_lock:
                result = await self._await_command(self._submit_entry, request)
            await self._send_msg(
                f"已提交交易 #{result['id']}：{request['pair']} {request['direction']}\n"
                f"本次保证金 {request['size']:g} USDT，杠杆 {result['leverage']:g}x。\n"
                '成交情况以订单通知和 /status 为准；没有后续自动补仓。',
                parse_mode=None,
            )
        except (ValueError, RPCException) as exc:
            await self._send_msg(str(exc), parse_mode=None)

        except Exception:
            logger.exception('Manual entry submission could not be confirmed')
            await self._send_msg(
                '提交状态暂无法确认，保护计划已保留。请先用 /status 和 /order 核实，勿直接重复下单。', parse_mode=None
            )

    def _submit_entry(self, request):
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
        old = deepcopy(strategy.manual_open.get(pair))
        plan = deepcopy(strategy._plan(existing) if existing else {})
        if not existing:
            plan.pop('trade_id', None)
        plan.update({k: v for k, v in request.items() if k not in ('pair', 'entry_price')})
        plan.setdefault('exit_points', [])
        if not plan.get('stop_loss'):
            raise ValueError('开仓必须提供 sl=<止损价>；本助手不自动生成止损')
        rate = request.get('entry_price') or bot.exchange.get_rate(
            pair, side='entry', is_short=request['direction'] == 'short', refresh=True
        )
        sl = float(plan['stop_loss'])
        if sl <= rate if request['direction'] == 'short' else sl >= rate:
            raise ValueError('止损必须位于当前入场价的亏损方向')
        if not existing and any(
            (p >= rate if request['direction'] == 'short' else p <= rate)
            for p in plan['exit_points']
        ):
            raise ValueError('止盈必须位于入场价的盈利方向')
        strategy.set_manual_plan(pair, plan)
        try:
            trade = self._rpc._rpc_force_entry(
                pair,
                request.get('entry_price'),
                order_side=SignalDirection(request['direction']),
                order_type='limit' if request.get('entry_price') is not None else 'market',
                stake_amount=request['size'],
                leverage=request['leverage'],
                enter_tag='manual_' + request['direction'],
            )
            if trade is None:
                raise RPCException('未创建订单')
        except (ValueError, RPCException):
            if old is None:
                strategy.delete_manual_plan(pair)
            else:
                strategy.set_manual_plan(pair, old)
            raise
        bound = deepcopy(strategy.manual_open.get(pair) or plan)
        bound['trade_id'] = trade.id
        strategy.set_manual_plan(pair, bound)
        return {'id': trade.id, 'leverage': trade.leverage}

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
            '/forcelong BTC 100 3 sl=90000 price=95000 tp=98000,100000,105000\n'
            '/forceshort BTC 100 3 sl=105000 tp=98000,95000,90000\n'
            '金额为保证金USDT，后接杠杆；不填price则市价，只接受单个入场价。以上数字仅示范格式。\n'
            '同方向已有持仓时，同一命令用于明确加仓，杠杆须与原仓一致。\n\n'
            '/setmanual <ID或币种> sl=... tp=... 修改持仓计划\n'
            '/manual 查看手动计划；/status table 查看实际仓位与移动止损\n'
            '/forceexit <ID或all> 全平；/close <ID> 50 部分平仓\n'
            '/coo <ID> 撤销未成交订单\n'
            '/monitor 推送BTC、ETH、SOL三份日线/4小时报告；运行时每小时也会自动推送\n'
            '/balance /profit /order /trades /performance 查询账户与成交\n'
            '/start /stopentry /stop /reload_config /health /logs 运行控制\n\n'
            'TP1平初始仓位30%，TP2平剩余50%，TP3清仓；两个TP各平50%，一个TP全平。\n'
            '部分止盈实际成交后，保护位依次移到含交易费的成本价、TP1；空头方向相反。\n'
            '关闭指标开仓、到价补仓、自动选方向、回测和AI调用。监控参考点位不触发交易。',
            parse_mode=None,
        )
