"""Telegram condition-watch wizard using the shared native panel primitives."""
import asyncio
import logging
import secrets
import time
from copy import deepcopy
from functools import partial

from telegram import InlineKeyboardButton, ForceReply
from freqtrade.rpc._telegram_base import authorized_only
from condition_watch import MODES, STATES, ACTIVE, MONITOR_PAIRS, describe, validate, price, stamp

logger = logging.getLogger(__name__)
LABELS = {
    'mode': '监控模式',
    'pair': '币种',
    'side': '关注方向',
    'tf': '收盘周期',
    'level': '突破／刺穿关键位',
    'low': '观察区域下限',
    'high': '观察区域上限',
    'confirm': '收盘确认价',
    'window': '确认窗口',
    'invalid': '即时失效价',
    'ttl': '有效期',
}


def steps(spec):
    result = ['mode', 'pair', 'side']
    mode = spec.get('mode', 'touch')
    if mode != 'touch':
        result.append('tf')
    if mode in ('retest', 'fake'):
        result.append('level')
    if mode in ('touch', 'reject', 'retest'):
        result += ['low', 'high']
    if mode != 'touch':
        result.append('confirm')
    if mode in ('reject', 'retest', 'fake'):
        result.append('window')
    return result + ['invalid', 'ttl']


def selected(spec):
    lines = []
    for field in steps(spec):
        if field not in spec:
            continue
        v = spec[field]
        if field == 'mode':
            v = MODES[v]
        elif field == 'pair':
            v = v.split('/')[0]
        elif field == 'side':
            v = '关注做多' if v == 'long' else '关注做空'
        elif field == 'ttl':
            v = f'{v // 3600}小时'
        elif field == 'invalid' and v is None:
            v = '不设置'
        lines.append(f'{LABELS[field]}：{v}')
    return '\n'.join(lines)


class WatchPanelMixin:
    def _watch_store(self):
        return self._rpc._freqtrade.strategy.watches

    def _watch_lock_get(self):
        if not hasattr(self, '_watch_lock'):
            self._watch_lock = asyncio.Lock()
        return self._watch_lock

    def _watch_button(self, update, label, op, **kw):
        payload = dict(action=op if op in ('list', 'detail') else 'watch', op=op, **kw)
        return InlineKeyboardButton(label, callback_data='cw:' + self._panel_token(update, payload))

    def _watch_clear_input(self, update):
        getattr(self, '_watch_inputs', {}).pop(self._panel_identity(update), None)

    async def _watch_list(self, update, ended=False, page=0, notice=''):
        self._watch_clear_input(update)
        rows = await asyncio.to_thread(self._watch_store().list, self._panel_identity(update))
        rows = [w for w in rows if (w['state'] not in ACTIVE) == ended]
        pages = max(1, (len(rows) + 5) // 6)
        page = min(max(0, page), pages - 1)
        visible = rows[page * 6 : (page + 1) * 6]
        b = partial(self._watch_button, update)
        buttons = [[b('新建监控', 'new')], [b('进行中', 'list'), b('已结束', 'list', ended=True)]]
        text = f"条件监控 · {'已结束' if ended else '进行中'} {len(rows)}条 · {page + 1}/{pages}\n"
        if notice:
            text = notice + '\n\n' + text
        if not visible:
            text += '\n当前没有此类监控。点击“新建监控”，分步设置价格条件。'
        for w in visible:
            title = f"#{w['id']} {w['spec']['pair'].split('/')[0]} · {STATES[w['state']]}"
            text += '\n\n' + title + '\n' + MODES[w['spec']['mode']]
            if w['state'] in ACTIVE:
                text += '\n' + w.get('health', '等待行情')
            buttons.append([b(title, 'detail', id=w['id'])])
        navigation = []
        if page:
            navigation.append(b('上一页', 'list', ended=ended, page=page - 1))
        if page + 1 < pages:
            navigation.append(b('下一页', 'list', ended=ended, page=page + 1))
        navigation.append(b('刷新', 'list', ended=ended, page=page))
        buttons.append(navigation)
        text += '\n\n仅提醒，不自动开单。小时简报与极端放量提醒继续独立运行。'
        await self._panel_send(update, text, buttons)

    async def _watch_detail(self, update, ident, notice=''):
        self._watch_clear_input(update)
        rows = await asyncio.to_thread(self._watch_store().list, self._panel_identity(update))
        w = next((w for w in rows if w['id'] == ident), None)
        if w is None:
            raise ValueError('监控不存在或不属于当前用户，请 /watch 刷新')
        s = w['spec']
        text = (
            f"#{w['id']} {s['pair'].split('/')[0]} · {STATES[w['state']]}\n"
            f"{MODES[s['mode']]} · {'关注做空' if s['side'] == 'short' else '关注做多'}\n"
            + describe(s)
            + f"\n创建：{stamp(w['created'])}\n到期：{stamp(w['expires'])}"
        )
        if w.get('deadline'):
            text += '\n确认窗口结束：' + stamp(w['deadline'])
        if w.get('checked'):
            text += f"\n最近观测：{w['last_price']:.8g}（{stamp(w['checked'])}）"
        if w.get('last_close_at'):
            text += f"\n最近核对收盘：{w['last_close']:.8g}（{stamp(w['last_close_at'])}）"
        text += '\n' + w.get('health', '') + '\n' + w.get('reason', '')
        text += '\n每10秒轮询；到价可能错过短暂触碰。仅提醒，不自动开单。'
        b = partial(self._watch_button, update)
        buttons = [[b('刷新', 'detail', id=ident), b('复制新建', 'copy', id=ident)]]
        if w['state'] in ACTIVE:
            buttons.append([b('取消监控', 'cancel', id=ident)])
        buttons.append([b('返回监控', 'list')])
        await self._panel_send(update, (notice + '\n\n' if notice else '') + text, buttons)

    async def _watch_render(self, update, draft, index=0, error=''):
        self._watch_clear_input(update)
        s = deepcopy(draft['spec'])
        fields = steps(s)
        index = min(max(0, index), len(fields))
        draft = dict(draft, spec=s, index=index)
        b = partial(self._watch_button, update)
        if index == len(fields):
            try:
                s = validate({'tf': '1h', 'window': 1, **s})
            except ValueError as exc:
                error = str(exc)
            buttons = []
            if not error:
                buttons.append([b('确认创建', 'create', draft=dict(draft, spec=s))])
            for field in fields:
                if field not in ('mode', 'pair'):
                    buttons.append(
                        [
                            b(
                                '修改' + LABELS[field],
                                'step',
                                draft=dict(draft, reviewing=True),
                                index=fields.index(field),
                            )
                        ]
                    )
            buttons.append([b('返回修改', 'step', draft=draft, index=index - 1), b('取消创建', 'list')])
            if error:
                await self._panel_send(update, '条件需要修改：' + error + '\n\n' + selected(s), buttons)
                return
            text = (
                f"确认创建 · {s['pair'].split('/')[0]} · {MODES[s['mode']]}\n"
                + ('关注做空' if s['side'] == 'short' else '关注做多')
                + '\n'
                + describe(s)
                + f"\n有效期：从确认起 {s['ttl'] // 3600}小时\n仅提醒，不自动开单。点位固定；创建前的影线不计入。"
            )
            await self._panel_send(update, text, buttons)
            return
        field = fields[index]
        text = f'新建监控 · 第{index + 1}/{len(fields)}步 · {LABELS[field]}\n'
        if s:
            text += '\n已填条件\n' + selected(s) + '\n'
        if error:
            text += '\n' + error + '\n'
        choices = {
            'mode': list(MODES.items()),
            'pair': [(p, p.split('/')[0]) for p in MONITOR_PAIRS],
            'side': [('long', '关注做多'), ('short', '关注做空')],
            'tf': [(tf, tf) for tf in ('15m', '1h', '4h', '1d')],
            'window': [(n, f'{n}根') for n in (1, 3, 6)],
            'ttl': [(14400, '4小时'), (86400, '24小时'), (259200, '3天'), (432000, '5天')],
        }
        buttons = []
        if field in choices:
            for value, label in choices[field]:
                buttons.append([b(label, 'choose', draft=draft, field=field, value=value)])
            text += '\n请选择' + LABELS[field] + '。'
        else:
            text += '\n请回复下一条输入提示，填写一个价格；回复“取消”退出。'
            if field == 'high':
                text += '\n单点监控：上限与下限填写相同价格。'
            if field == 'confirm':
                text += '\n' + ('空头等待收盘严格低于此价。' if s['side'] == 'short' else '多头等待收盘严格高于此价。')
            if field == 'invalid':
                buttons.append([b('不设价格失效', 'choose', draft=draft, field=field, value=None)])
        nav = [b('取消创建', 'list')]
        if index:
            nav.insert(0, b('上一步', 'step', draft=draft, index=index - 1))
        buttons.append(nav)
        await self._panel_send(update, text, buttons)
        if field not in choices:
            prompt = await self._app.bot.send_message(
                chat_id=update.effective_chat.id,
                message_thread_id=update.effective_message.message_thread_id,
                text=f"请输入{LABELS[field]}（只回复一个数字，例如 95000.5；示例不是建议点位）。\n回复“取消”退出。",
                parse_mode=None,
                reply_markup=ForceReply(selective=False, input_field_placeholder=LABELS[field]),
            )
            inputs = getattr(self, '_watch_inputs', {})
            # ponytail: one draft per authorized user/topic; restart discards unfinished input.
            inputs = {k: v for k, v in inputs.items() if v['expires'] > time.monotonic()}
            inputs[self._panel_identity(update)] = dict(
                draft=draft, field=field, prompt=prompt.message_id, expires=time.monotonic() + 600
            )
            self._watch_inputs = inputs

    async def _watch_choose(self, update, draft, field, value):
        s = dict(draft['spec'])
        if field == 'mode' and s.get('mode') != value:
            s = {k: v for k, v in s.items() if k in ('pair', 'side', 'tf', 'ttl')}
        s[field] = value
        if field == 'high' and s['high'] < s['low']:
            raise ValueError('区域上限不能低于下限，请重新输入上限')
        if field == 'confirm':
            ref = s.get('low') if s['side'] == 'short' else s.get('high')
            if s['mode'] == 'fake':
                ref = s['level']
            if ref is not None and (
                s['side'] == 'short' and value > ref or s['side'] == 'long' and value < ref
            ):
                raise ValueError('确认价方向不正确：空头须≤区域下限／关键位，多头须≥区域上限／关键位')
        if field == 'high' and s['mode'] == 'retest' and not s['low'] <= s['level'] <= s['high']:
            raise ValueError('回踩区域须包含突破位，请修改上限或返回修改下限')
        if field == 'invalid' and value is not None:
            validate({'tf': '1h', 'window': 1, 'ttl': 86400, **s})
        next_index = len(steps(s)) if draft.get('reviewing') else draft['index'] + 1
        await self._watch_render(update, dict(draft, spec=s), next_index)

    @authorized_only
    async def _watch(self, update, context):
        async with self._watch_lock_get():
            try:
                await self._watch_list(update)
            except Exception as exc:
                logger.warning('Watch list unavailable (%s)', type(exc).__name__)
                await self._panel_send(update, '监控列表暂不可用，请稍后 /watch 重试。')

    @authorized_only
    async def _watch_callback(self, update, context):
        await update.callback_query.answer()
        async with self._watch_lock_get():
            try:
                token = update.callback_query.data.removeprefix('cw:')
                p = self._panel_take(update, token, consume=True)
                self._watch_clear_input(update)
                op = p['op']
                if op == 'list':
                    await self._watch_list(update, p.get('ended', False), p.get('page', 0))
                elif op == 'detail':
                    await self._watch_detail(update, p['id'])
                elif op == 'new':
                    await self._watch_render(update, dict(spec={}, request=secrets.token_hex(12)))
                elif op == 'step':
                    await self._watch_render(update, p['draft'], p['index'])
                elif op == 'choose':
                    await self._watch_choose(update, p['draft'], p['field'], p['value'])
                elif op == 'create':
                    w = await asyncio.to_thread(
                        self._watch_store().create,
                        p['draft']['spec'],
                        self._panel_identity(update),
                        p['draft']['request'],
                    )
                    await self._watch_detail(update, w['id'], '已创建监控，开始等待条件。')
                elif op == 'cancel':
                    await asyncio.to_thread(
                        self._watch_store().cancel, p['id'], self._panel_identity(update)
                    )
                    await self._watch_detail(update, p['id'], '已结束本条监控；其他监控继续运行。')
                elif op == 'copy':
                    rows = await asyncio.to_thread(
                        self._watch_store().list, self._panel_identity(update)
                    )
                    w = next((w for w in rows if w['id'] == p['id']), None)
                    if w is None:
                        raise ValueError('原监控不存在，请刷新')
                    draft = dict(spec=w['spec'], request=secrets.token_hex(12))
                    await self._watch_render(update, draft, len(steps(draft['spec'])))
            except ValueError as exc:
                await self._panel_send(
                    update, str(exc), [[self._watch_button(update, '返回监控', 'list')]]
                )
            except Exception as exc:
                logger.warning('Watch operation unavailable (%s)', type(exc).__name__)
                await self._panel_send(
                    update,
                    '结果暂未确认，请 /watch 刷新核对已有监控，避免重复创建。',
                    [[self._watch_button(update, '返回监控', 'list')]],
                )

    async def _watch_reply(self, update):
        async with self._watch_lock_get():
            inputs = getattr(self, '_watch_inputs', {})
            key = self._panel_identity(update)
            row = inputs.get(key)
            reply = update.message.reply_to_message
            if not row or not reply or reply.message_id != row['prompt']:
                return False
            inputs.pop(key)
            if row['expires'] <= time.monotonic():
                await self._watch_list(update, notice='输入已过期，未创建监控。请重新开始。')
                return True
            value = update.message.text.strip()
            if value in ('取消', 'cancel'):
                await self._watch_list(update, notice='已取消创建。')
                return True
            try:
                await self._watch_choose(update, row['draft'], row['field'], price(value))
            except ValueError as exc:
                await self._watch_render(update, row['draft'], row['draft']['index'], str(exc))
            except Exception as exc:
                logger.warning('Watch input unavailable (%s)', type(exc).__name__)
                await self._panel_send(update, '输入处理失败，尚未创建监控，请 /watch 重新开始。')
            return True
