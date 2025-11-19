# pragma pylint: disable=unused-argument, unused-variable, protected-access, invalid-name

"""
This module manage Telegram communication
"""
import asyncio
import json
import logging
import re
import io
import ccxt
import time

# Generate chart image
import pandas as pd
from io import BytesIO
import mplfinance as mpf
from collections.abc import Callable, Coroutine
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import partial, wraps
from html import escape
import html as _py_html
from itertools import chain
from math import isnan
from threading import Thread
from typing import Any, Literal

from tabulate import tabulate
from telegram import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
    Update,
    InputMediaPhoto,
    InputMediaDocument,
    ForceReply,
)
from telegram.constants import MessageLimit, ParseMode
from telegram.error import BadRequest, NetworkError, TelegramError
from telegram.ext import Application, CallbackContext, CallbackQueryHandler, CommandHandler, MessageHandler, filters
from telegram.helpers import escape_markdown

from freqtrade.__init__ import __version__
from freqtrade.constants import DUST_PER_COIN, Config
from freqtrade.enums import (
    MarketDirection,
    RPCMessageType,
    SignalDirection,
    TradingMode,
    State,
)
from freqtrade.exceptions import OperationalException
from freqtrade.misc import chunks, plural
from freqtrade.persistence import Trade
from freqtrade.rpc import RPC, RPCException, RPCHandler
from freqtrade.rpc.rpc_types import RPCEntryMsg, RPCExitMsg, RPCOrderMsg, RPCSendMsg
from freqtrade.util import (
    dt_from_ts,
    dt_humanize_delta,
    fmt_coin,
    fmt_coin2,
    format_date,
    round_value,
)
from freqtrade.freqllm.analysis_agent import CryptoTechnicalAnalyst
from freqtrade.freqllm.ai_agent import run_two_phase
from freqtrade.freqllm.pagination_utils import split_html_pages, save_pages, get_page
from freqtrade.freqllm.html_sanitizer import sanitize_telegram_html
from freqtrade.freqllm.key_level_agent import TradingSignalExtractor
from freqtrade.freqllm.db_manager import connect_to_db, get_todays_analysis, insert_analysis_result
from contextlib import contextmanager

MAX_MESSAGE_LENGTH = MessageLimit.MAX_TEXT_LENGTH


logger = logging.getLogger(__name__)

logger.debug('Included module rpc.telegram ...')

def _parse_price_list(token: str) -> list[float | None]:
    parts = [p.strip() for p in str(token).split(',')]
    out: list[float | None] = []
    for p in parts:
        if p == '' or p.lower() in ('none', 'null', 'market'):
            out.append(None)
        else:
            out.append(float(p))
    return out

def _allocations_for(n: int) -> list[int]:
    if n == 3: return [50, 30, 20]
    if n == 2: return [60, 40]
    if n <= 1: return [100]
    base = 100 // n
    arr = [base] * n
    arr[0] += 100 - base * n
    return arr

def _parse_overrides(args):
    """解析 open_rate=xxx amount=xxx stake_amount=xxx"""
    out = {}
    for tok in args or []:
        m = re.match(r'^(open_rate|amount|stake_amount)=([\d.]+)$', tok.strip())
        if m:
            out[m.group(1)] = float(m.group(2))
    return out

def _build_exchange_from_config(config):
    """使用 self.config / original_config 构建 ccxt 实例（USDT-M 永续）"""
    exchange_name = config['exchange']['name'].lower()
    exchange_class = getattr(ccxt, exchange_name)
    api_config = {
        'apiKey': config['original_config']['exchange'].get('key', ''),
        'secret': config['original_config']['exchange'].get('secret', ''),
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',
            'defaultMarket': 'future',
        },
    }
    # 过滤空值（和你提供的一致）
    api_config = {k: v for k, v in api_config.items() if v}
    return exchange_class(api_config)

def _fetch_position(exchange, pair: str):
    """
    读取该 pair 的第一条非零仓位，返回:
      {'pair','side','contracts','entryPrice'}
    """
    symbol = _normalize_pair(pair)
    try:
        positions = exchange.fetch_positions([symbol]) or []
    except Exception as e:
        print(f"[sync_trade] fetch_positions error: {e}")
        return None

    for pos in positions:
        size = pos['contracts']
        entry = pos['entryPrice']
        side = pos['side']

        return {
            'pair': pair,
            'side': side,
            'contracts': float(size),
            'entryPrice': float(entry),
        }
    return None

@contextmanager
def _temp_entry_type(strategy, new_type='limit'):
    old_entry = strategy.order_types.get('entry', 'market')
    old_force = strategy.order_types.get('force_entry', old_entry)
    strategy.order_types['entry'] = new_type
    strategy.order_types['force_entry'] = new_type
    try:
        yield
    finally:
        strategy.order_types['entry'] = old_entry
        strategy.order_types['force_entry'] = old_force


def _parse_kv_tokens(tokens: list[str]) -> dict:
    """
    解析形如 ["price=2700", "sl=2500", "tps=71000,72000,73500", "tp1=...", "tp2=...", "tp3=..."]
    的 KV 风格参数。返回 dict，未出现的键不包含。
    """
    out = {}
    for t in tokens:
        if '=' not in t:
            continue
        k, v = t.split('=', 1)
        k = k.strip().lower()
        v = v.strip()
        if not k:
            continue
        out[k] = v
    return out


def _maybe_float(x: str | None) -> float | None:
    if x is None or x == '':
        return None
    try:
        return float(x)
    except Exception:
        return None


def _extract_tps_from_kv(kv: dict) -> list[float]:
    """
    从 kv 中提取 tps。支持：
      - tps=71000,72000,73500
      - tp1=..., tp2=..., tp3=...
      - tp=...（单个）
    返回: 已转成 float 的列表（未提供返回 []）
    """
    tps: list[float] = []

    # tps=comma,separated
    if 'tps' in kv:
        parts = [p.strip() for p in kv['tps'].split(',') if p.strip() != '']
        for p in parts:
            v = _maybe_float(p)
            if v is not None:
                tps.append(v)

    # 单个/分散的写法
    for key in ('tp', 'tp1', 'tp2', 'tp3'):
        if key in kv:
            v = _maybe_float(kv[key])
            if v is not None:
                tps.append(v)

    return tps


def _normalize_pair(p: str) -> str:
    p = p.upper()
    if ':' in p:
        return p
    if p.endswith('/USDT'):
        return p + ':USDT'
    if '/' not in p:
        return p + '/USDT:USDT'
    return p


def _to_clean_plain(text: str) -> str:
    """
    将可能包含 HTML / 实体 / 占位残渣的文本转为干净纯文本：
      - 去标签
      - 去我们占位符（\uFFF0idx\uFFF0）
      - 解实体 (&amp; -> &)
      - 合理压缩空白
    """
    if not text:
        return ''
    # 1) 去标签
    plain = re.sub(r'<[^>]+>', '', text)
    # 2) 去占位符（如果还有）
    plain = re.sub(r'\uFFF0\d+\uFFF0', '', plain)
    # 3) 解实体
    plain = _py_html.unescape(plain)
    # 4) 压缩多空格与空行
    plain = re.sub(r'[ \t]{2,}', ' ', plain)
    plain = re.sub(r'\n{3,}', '\n\n', plain).strip()
    return plain[:3900]  # Telegram 安全边界


def _beautify_inequalities(html_text: str) -> str:
    # 把 &lt;number 转成 ≤number ，把 &gt;number 转成 ≥number
    # 仅替换普通正文里的模式，不碰标签
    html_text = re.sub(r'&lt;\s*(\d+(\.\d+)?\s*\w*)', r'≤ \1', html_text)
    html_text = re.sub(r'&gt;\s*(\d+(\.\d+)?\s*\w*)', r'≥ \1', html_text)
    # 去掉替换后多余空格，例如 "≤ 1h" → "≤1h"
    html_text = re.sub(r'≤\s+', '≤', html_text)
    html_text = re.sub(r'≥\s+', '≥', html_text)
    return html_text

# 允许的模式集合
ALLOWED_MODES = {0, 1, 2, 3, 4}  # 你用到几个就留几个

MODE_PREFIX_PATTERNS = [
    re.compile(r'^\s*\[(?P<mode>\d)\]\s*', re.I),     # [2] Prompt...
    re.compile(r'^\s*(?P<mode>\d)\s*::\s*', re.I),    # 2:: Prompt...
]
MODE_SUFFIX_PATTERNS = [
    re.compile(r'\s*::\s*(?P<mode>\d)\s*$', re.I),    # Prompt ... ::2
]

def extract_mode_and_prompt(raw: str, default_mode: int = 0):
    """
    仅在消息“开头/结尾”解析模式，避免误伤正文。
    格式：
      前缀:  [2] ...   或   2:: ...
      后缀:  ... ::2
    若无 → 使用 default_mode
    """
    text = raw or ''
    # 1) 前缀
    for rx in MODE_PREFIX_PATTERNS:
        m = rx.match(text)
        if m:
            mode = int(m.group('mode'))
            if mode in ALLOWED_MODES:
                return mode, text[m.end():].strip()
    # 2) 后缀
    for rx in MODE_SUFFIX_PATTERNS:
        m = rx.search(text)
        if m:
            mode = int(m.group('mode'))
            if mode in ALLOWED_MODES:
                # 去掉后缀
                cleaned = text[:m.start()].rstrip()
                return mode, cleaned
    # 3) 默认
    return default_mode, text.strip()

def safe_async_db(func: Callable[..., Any]):
    """
    Decorator to safely handle sessions when switching async context
    :param func: function to decorate
    :return: decorated function
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        """Decorator logic"""
        try:
            return func(*args, **kwargs)
        finally:
            Trade.session.remove()

    return wrapper


@dataclass
class TimeunitMappings:
    header: str
    message: str
    message2: str
    callback: str
    default: int
    dateformat: str


def authorized_only(command_handler: Callable[..., Coroutine[Any, Any, None]]):
    """
    Decorator to check if the message comes from the correct chat_id
    can only be used with Telegram Class to decorate instance methods.
    :param command_handler: Telegram CommandHandler
    :return: decorated function
    """

    @wraps(command_handler)
    async def wrapper(self, *args, **kwargs):
        """Decorator logic"""
        update = kwargs.get('update') or args[0]

        # Reject unauthorized messages
        if update.callback_query:
            cchat_id = int(update.callback_query.message.chat.id)
            ctopic_id = update.callback_query.message.message_thread_id
        else:
            cchat_id = int(update.message.chat_id)
            ctopic_id = update.message.message_thread_id

        chat_id = int(self._config['telegram']['chat_id'])
        if cchat_id != chat_id:
            logger.info(f"Rejected unauthorized message from: {cchat_id}")
            return None
        if (topic_id := self._config['telegram'].get('topic_id')) is not None:
            if str(ctopic_id) != topic_id:
                # This can be quite common in multi-topic environments.
                logger.debug(f"Rejected message from wrong channel: {cchat_id}, {ctopic_id}")
                return None

        # Rollback session to avoid getting data stored in a transaction.
        Trade.rollback()
        logger.debug('Executing handler: %s for chat_id: %s', command_handler.__name__, chat_id)
        try:
            return await command_handler(self, *args, **kwargs)
        except RPCException as e:
            await self._send_msg(str(e))
        except BaseException:
            logger.exception('Exception occurred within Telegram module')
        finally:
            Trade.session.remove()

    return wrapper


class Telegram(RPCHandler):
    """This class handles all telegram communication"""

    def __init__(self, rpc: RPC, config: Config) -> None:
        """
        Init the Telegram call, and init the super class RPCHandler
        :param rpc: instance of RPC Helper class
        :param config: Configuration object
        :return: None
        """
        super().__init__(rpc, config)

        self._app: Application
        self._loop: asyncio.AbstractEventLoop
        self._init_keyboard()
        self._start_thread()
        self._pending_force: dict[tuple[int, int], dict] = {}
        self._pending_manual_edit: dict[tuple[int, int], dict] = {}
        self._pending_monitor_edit: dict[tuple[int, int], dict] = {}
        self._pending_hedge_edit: dict[tuple[int, int], dict] = {}

    def _start_thread(self):
        """
        Creates and starts the polling thread
        """
        self._thread = Thread(target=self._init, name='FTTelegram')
        self._thread.start()

    def _init_keyboard(self) -> None:
        """
        Validates the keyboard configuration from telegram config
        section.
        """
        self._keyboard: list[list[str | KeyboardButton]] = [
            ['/daily', '/profit', '/balance'],
            ['/status', '/status table', '/performance'],
            ['/count', '/start', '/stop', '/help'],
            ['/chart', '/analysis', '/ai', '/prompt', '/promptjson', '/manual', '/monitor'],
            ['/addpair', '/delpair', '/setmanual'],
            ['/setpairstrategy', '/delpairstrategy', '/showpairstrategy', '/setpairstrategyauto'],
        ]
        # do not allow commands with mandatory arguments and critical cmds
        # TODO: DRY! - its not good to list all valid cmds here. But otherwise
        #       this needs refactoring of the whole telegram module (same
        #       problem in _help()).
        valid_keys: list[str] = [
            r'/start$',
            r'/stop$',
            r'/status$',
            r'/status table$',
            r'/trades$',
            r'/performance$',
            r'/buys',
            r'/entries',
            r'/sells',
            r'/exits',
            r'/mix_tags',
            r'/daily$',
            r'/daily \d+$',
            r'/profit$',
            r'/profit \d+',
            r'/stats$',
            r'/count$',
            r'/locks$',
            r'/balance$',
            r'/stopbuy$',
            r'/stopentry$',
            r'/reload_config$',
            r'/show_config$',
            r'/logs$',
            r'/whitelist$',
            r'/whitelist(\ssorted|\sbaseonly)+$',
            r'/blacklist$',
            r'/bl_delete$',
            r'/weekly$',
            r'/weekly \d+$',
            r'/monthly$',
            r'/monthly \d+$',
            r'/forcebuy$',
            r'/forcelong$',
            r'/forceshort$',
            r'/forcesell$',
            r'/forceexit$',
            r'/edge$',
            r'/health$',
            r'/help$',
            r'/version$',
            r'/marketdir (long|short|even|none)$',
            r'/marketdir$',
            r'/manual$',
            r'/monitor$',
            r'/chart$',  # chart命令格式
            r'/ai$',  # ai命令格式
            r'/analysis$',  # analysis命令格式
            r'/prompt$',  # analysis命令格式
            r'/promptjson$',  # analysis命令格式
            r'/addpair$',
            r'/delpair$',
            r'/setpairstrategy$',
            r'/setpairstrategyauto$',
            r'/delpairstrategy$',
            r'/showpairstrategy$',
            r'/setmanual$',
            r'/restoremanual$',
            r'/coo$',
            r'/fx$',
            r'/rtd$',
            r'/st$',
            r'/hglong$',
            r'/hgshort$',
            r'/hedge$',
            r'/fehg$',
            r'/fxhg$',
            r'/statushg$',
        ]
        # Create keys for generation
        valid_keys_print = [k.replace('$', '') for k in valid_keys]

        # custom keyboard specified in config.json
        cust_keyboard = self._config['telegram'].get('keyboard', [])
        if cust_keyboard:
            combined = '(' + ')|('.join(valid_keys) + ')'
            # check for valid shortcuts
            invalid_keys = [
                b for b in chain.from_iterable(cust_keyboard) if not re.match(combined, b)
            ]
            if len(invalid_keys):
                err_msg = (
                    'config.telegram.keyboard: Invalid commands for '
                    f"custom Telegram keyboard: {invalid_keys}"
                    f"\nvalid commands are: {valid_keys_print}"
                )
                raise OperationalException(err_msg)
            else:
                self._keyboard = cust_keyboard
                logger.info(f"using custom keyboard from config.json: {self._keyboard}")

    def _init_telegram_app(self):
        return Application.builder().token(self._config['telegram']['token']).build()

    def _init(self) -> None:
        """
        Initializes this module with the given config,
        registers all known command handlers
        and starts polling for message updates
        Runs in a separate thread.
        """
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

        self._app = self._init_telegram_app()

        # Register command handler and start telegram message polling
        handles = [
            CommandHandler('status', self._status),
            CommandHandler('profit', self._profit),
            CommandHandler('balance', self._balance),
            CommandHandler('start', self._start),
            CommandHandler('stop', self._stop),
            CommandHandler(['forcesell', 'forceexit', 'fx'], self._force_exit),
            CommandHandler(
                ['forcebuy', 'forcelong'],
                partial(self._force_enter, order_side=SignalDirection.LONG, is_hedge=False),
            ),
            CommandHandler(
                'forceshort', partial(self._force_enter, order_side=SignalDirection.SHORT, is_hedge=False)
            ),
            # hedge：只写配置，不下单
            CommandHandler('hglong',
                partial(self._force_enter, order_side=SignalDirection.LONG, is_hedge=True)),
            CommandHandler('hgshort',
                partial(self._force_enter, order_side=SignalDirection.SHORT, is_hedge=True)),
            CommandHandler('reload_trade', self._reload_trade_from_exchange),
            CommandHandler('trades', self._trades),
            CommandHandler('delete', self._delete_trade),
            CommandHandler(['coo', 'cancel_open_order'], self._cancel_open_order),
            CommandHandler(['st', 'sync_trade'], self._sync_trade),
            CommandHandler(['rtd', 'reset_trade_data'], self._reset_trade_data),
            CommandHandler('performance', self._performance),
            CommandHandler(['buys', 'entries'], self._enter_tag_performance),
            CommandHandler(['sells', 'exits'], self._exit_reason_performance),
            CommandHandler('mix_tags', self._mix_tag_performance),
            CommandHandler('stats', self._stats),
            CommandHandler('daily', self._daily),
            CommandHandler('weekly', self._weekly),
            CommandHandler('monthly', self._monthly),
            CommandHandler('count', self._count),
            CommandHandler('locks', self._locks),
            CommandHandler(['unlock', 'delete_locks'], self._delete_locks),
            CommandHandler(['reload_config', 'reload_conf'], self._reload_config),
            CommandHandler(['show_config', 'show_conf'], self._show_config),
            CommandHandler(['stopbuy', 'stopentry'], self._stopentry),
            CommandHandler('whitelist', self._whitelist),
            CommandHandler('blacklist', self._blacklist),
            CommandHandler(['blacklist_delete', 'bl_delete'], self._blacklist_delete),
            CommandHandler('logs', self._logs),
            CommandHandler('edge', self._edge),
            CommandHandler('health', self._health),
            CommandHandler('help', self._help),
            CommandHandler('version', self._version),
            CommandHandler('marketdir', self._changemarketdir),
            CommandHandler('order', self._order),
            CommandHandler('list_custom_data', self._list_custom_data),
            CommandHandler('tg_info', self._tg_info),
            CommandHandler('chart', self._chart),
            CommandHandler('ai', self._cmd_two_phase),
            CommandHandler('analysis', self._analysis),
            CommandHandler('prompt', self._prompt),
            CommandHandler('promptjson', self._prompt_json),
            CommandHandler('addpair', self._add_pair),
            CommandHandler('delpair', self._del_pair),
            CommandHandler('setpairstrategy', self._set_pair_strategy),
            CommandHandler('setpairstrategyauto', self._set_pair_strategy_auto),
            CommandHandler('delpairstrategy', self._del_pair_strategy),
            CommandHandler('showpairstrategy', self._show_pair_strategy),
            CommandHandler('manualopen', self._manual_open),
            CommandHandler('manual', self._manual_open),
            CommandHandler('hedge', self._hedge_open),
            CommandHandler('monitor', self._monitor_list),
            CommandHandler('monitoring', self._monitor_list),
            CommandHandler('setmanual', self._set_manual),
            CommandHandler('restoremanual', self._restore_manual),
            CommandHandler('fehg', self._hedge_force_open),
            CommandHandler('fxhg', self._hedge_force_close),
            CommandHandler('statushg', self._hedge_status),
        ]
        callbacks = [
            CallbackQueryHandler(self._status_table, pattern='update_status_table'),
            CallbackQueryHandler(self._chart, pattern=r'update_chart(?::(.+))?'),
            CallbackQueryHandler(self._prompt, pattern=r'update_prompt(?::(.+))?'),
            CallbackQueryHandler(self._daily, pattern='update_daily'),
            CallbackQueryHandler(self._weekly, pattern='update_weekly'),
            CallbackQueryHandler(self._monthly, pattern='update_monthly'),
            CallbackQueryHandler(self._profit, pattern='update_profit'),
            CallbackQueryHandler(self._balance, pattern='update_balance'),
            CallbackQueryHandler(self._performance, pattern='update_performance'),
            CallbackQueryHandler(
                self._enter_tag_performance, pattern='update_enter_tag_performance'
            ),
            CallbackQueryHandler(
                self._exit_reason_performance, pattern='update_exit_reason_performance'
            ),
            CallbackQueryHandler(self._mix_tag_performance, pattern='update_mix_tag_performance'),
            CallbackQueryHandler(self._count, pattern='update_count'),
            CallbackQueryHandler(self._force_exit_inline, pattern=r'force_exit__\S+'),
            CallbackQueryHandler(self._force_enter_inline, pattern=r'force_enter__\S+'),
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._text_router),
            CallbackQueryHandler(self._manual_open, pattern='update_manual_list'),
            CallbackQueryHandler(self._manual_open_view, pattern=r'manual_select__.+'),
            CallbackQueryHandler(self._manual_edit_inline, pattern=r'manual_edit__.+'),
            CallbackQueryHandler(self._monitor_list, pattern='update_monitor_list'),
            CallbackQueryHandler(self._monitor_view, pattern=r'monitor_select__.+'),
            CallbackQueryHandler(self._monitor_edit_inline, pattern=r'monitor_edit__.+'),
            CallbackQueryHandler(self._monitor_recalculate_inline, pattern=r'monitor_recalculate__.+'),
            CallbackQueryHandler(self._hedge_open,        pattern='update_hedge_list'),
            CallbackQueryHandler(self._hedge_open_view,   pattern=r'hedge_select__.+'),
            CallbackQueryHandler(self._hedge_edit_inline, pattern=r'hedge_edit__.+'),
            CallbackQueryHandler(self._hedge_delete,      pattern=r'hedge_delete__.+'),
            CallbackQueryHandler(self._ai_pagination_handler, pattern=r'^ai_(page|copy):'),
        ]
        for handle in handles:
            self._app.add_handler(handle)

        for callback in callbacks:
            self._app.add_handler(callback)

        logger.info(
            'rpc.telegram is listening for following commands: %s',
            [[x for x in sorted(h.commands)] for h in handles],
        )
        self._loop.run_until_complete(self._startup_telegram())

    def _get_open_trade_by_pair_or_id(self, ident: str):
        """
        ident 可以是 'BTC/USDT' 这样的 pair，也可以是整数 trade_id。
        返回: Trade 或 None
        """
        from freqtrade.persistence import Trade
        q = Trade.get_trades_query([Trade.is_open.is_(True)])
        if '/' in ident.upper():
            return Trade.session.scalars(q.filter(Trade.pair == ident.upper())).first()
        else:
            try:
                tid = int(ident)
                return Trade.session.scalars(q.filter(Trade.id == tid)).first()
            except ValueError:
                return None

    def _clear_manual_open_for_pair(self, pair: str):
        """
        从 strategy_state_production.json 删除 manual_open[pair]，并同步内存 strategy.manual_open
        """
        state_file = 'user_data/strategy_state_production.json'
        try:
            with open(state_file, 'r') as f:
                strategy_state = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            strategy_state = {}

        manual_open = strategy_state.get('manual_open', {})
        changed = False
        if pair in manual_open:
            manual_open.pop(pair, None)
            strategy_state['manual_open'] = manual_open
            with open(state_file, 'w') as f:
                json.dump(strategy_state, f, indent=4)
            changed = True

        # 同步到内存
        if hasattr(self._rpc._freqtrade, 'strategy'):
            self._rpc._freqtrade.strategy.manual_open = manual_open

        return changed

    def _find_monitor_first_entry(self, pair: str, side_str: str):
        """
        在内存中的 coin_monitoring[pair]（array）里，找到第一个 direction 匹配 side_str 的配置，
        返回 config['entry_points'][0]；找不到返回 None
        """
        strat = getattr(self._rpc._freqtrade, 'strategy', None)
        if not strat:
            return None
        cm = getattr(strat, 'coin_monitoring', {}) or {}
        arr = cm.get(pair, [])
        for conf in arr:
            if (str(conf.get('direction', '')).lower() == side_str.lower()):
                eps = conf.get('entry_points') or []
                if eps:
                    return eps[0]
        return None

    def _fmt_price_str(self, val) -> str:
        """
        将价格安全地转成字符串，避免 2.7101834999999994 这种长尾。
        规则：尽量保留到 8 位小数，去掉尾部多余 0 和小数点。
        """
        try:
            from decimal import Decimal, ROUND_DOWN, InvalidOperation
            d = Decimal(str(val)).quantize(Decimal('0.00000001'), rounding=ROUND_DOWN)
            s = format(d, 'f').rstrip('0').rstrip('.')
            return s if s else '0'
        except Exception:
            # 兜底
            return str(val)

    async def _startup_telegram(self) -> None:
        await self._app.initialize()
        await self._app.start()
        if self._app.updater:
            await self._app.updater.start_polling(
                bootstrap_retries=-1,
                timeout=20,
                # read_latency=60,  # Assumed transmission latency
                drop_pending_updates=True,
                # stop_signals=[],  # Necessary as we don't run on the main thread
            )
            while True:
                await asyncio.sleep(10)
                if not self._app.updater.running:
                    break

    async def _cleanup_telegram(self) -> None:
        if self._app.updater:
            await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()

    def cleanup(self) -> None:
        """
        Stops all running telegram threads.
        :return: None
        """
        # This can take up to `timeout` from the call to `start_polling`.
        asyncio.run_coroutine_threadsafe(self._cleanup_telegram(), self._loop)
        self._thread.join()

    def _exchange_from_msg(self, msg: RPCOrderMsg) -> str:
        """
        Extracts the exchange name from the given message.
        :param msg: The message to extract the exchange name from.
        :return: The exchange name.
        """
        return f"{msg['exchange']}{' (dry)' if self._config['dry_run'] else ''}"

    def _add_analyzed_candle(self, pair: str) -> str:
        candle_val = (
            self._config['telegram'].get('notification_settings', {}).get('show_candle', 'off')
        )
        if candle_val != 'off':
            if candle_val == 'ohlc':
                analyzed_df, _ = self._rpc._freqtrade.dataprovider.get_analyzed_dataframe(
                    pair, self._config['timeframe']
                )
                candle = analyzed_df.iloc[-1].squeeze() if len(analyzed_df) > 0 else None
                if candle is not None:
                    return (
                        f"*Candle OHLC*: `{candle['open']}, {candle['high']}, "
                        f"{candle['low']}, {candle['close']}`\n"
                    )

        return ''

    def _format_entry_msg(self, msg: RPCEntryMsg) -> str:
        is_fill = msg['type'] in [RPCMessageType.ENTRY_FILL]
        emoji = '\N{CHECK MARK}' if is_fill else '\N{LARGE BLUE CIRCLE}'

        terminology = {
            '1_enter': 'New Trade',
            '1_entered': 'New Trade filled',
            'x_enter': 'Increasing position',
            'x_entered': 'Position increase filled',
        }

        key = f"{'x' if msg['sub_trade'] else '1'}_{'entered' if is_fill else 'enter'}"
        wording = terminology[key]

        message = (
            f"{emoji} *{self._exchange_from_msg(msg)}:*"
            f" {wording} (#{msg['trade_id']})\n"
            f"*Pair:* `{msg['pair']}`\n"
        )
        message += self._add_analyzed_candle(msg['pair'])
        message += f"*Enter Tag:* `{msg['enter_tag']}`\n" if msg.get('enter_tag') else ''
        message += f"*Amount:* `{round_value(msg['amount'], 8)}`\n"
        message += f"*Direction:* `{msg['direction']}"
        if msg.get('leverage') and msg.get('leverage', 1.0) != 1.0:
            message += f" ({msg['leverage']:.3g}x)"
        message += '`\n'
        message += f"*Open Rate:* `{fmt_coin2(msg['open_rate'], msg['quote_currency'])}`\n"
        if msg['type'] == RPCMessageType.ENTRY and msg['current_rate']:
            message += (
                f"*Current Rate:* `{fmt_coin2(msg['current_rate'], msg['quote_currency'])}`\n"
            )

        profit_fiat_extra = self.__format_profit_fiat(msg, 'stake_amount')  # type: ignore
        total = fmt_coin(msg['stake_amount'], msg['quote_currency'])

        message += f"*{'New ' if msg['sub_trade'] else ''}Total:* `{total}{profit_fiat_extra}`"

        return message

    def _format_exit_msg(self, msg: RPCExitMsg) -> str:
        duration = msg['close_date'].replace(microsecond=0) - msg['open_date'].replace(
            microsecond=0
        )
        duration_min = duration.total_seconds() / 60

        leverage_text = (
            f" ({msg['leverage']:.3g}x)"
            if msg.get('leverage') and msg.get('leverage', 1.0) != 1.0
            else ''
        )

        profit_fiat_extra = self.__format_profit_fiat(msg, 'profit_amount')

        profit_extra = (
            f" ({msg['gain']}: {fmt_coin(msg['profit_amount'], msg['quote_currency'])}"
            f"{profit_fiat_extra})"
        )

        is_fill = msg['type'] == RPCMessageType.EXIT_FILL
        is_sub_trade = msg.get('sub_trade')
        is_sub_profit = msg['profit_amount'] != msg.get('cumulative_profit')
        is_final_exit = msg.get('is_final_exit', False) and is_sub_profit
        profit_prefix = 'Sub ' if is_sub_trade else ''
        cp_extra = ''
        exit_wording = 'Exited' if is_fill else 'Exiting'
        if is_sub_trade or is_final_exit:
            cp_fiat = self.__format_profit_fiat(msg, 'cumulative_profit')

            if is_final_exit:
                profit_prefix = 'Sub '
                cp_extra = (
                    f"*Final Profit:* `{msg['final_profit_ratio']:.2%} "
                    f"({msg['cumulative_profit']:.8f} {msg['quote_currency']}{cp_fiat})`\n"
                )
            else:
                exit_wording = f"Partially {exit_wording.lower()}"
                if msg['cumulative_profit']:
                    cp_extra = (
                        f"*Cumulative Profit:* `"
                        f"{fmt_coin(msg['cumulative_profit'], msg['stake_currency'])}{cp_fiat}`\n"
                    )
        enter_tag = f"*Enter Tag:* `{msg['enter_tag']}`\n" if msg.get('enter_tag') else ''
        message = (
            f"{self._get_exit_emoji(msg)} *{self._exchange_from_msg(msg)}:* "
            f"{exit_wording} {msg['pair']} (#{msg['trade_id']})\n"
            f"{self._add_analyzed_candle(msg['pair'])}"
            f"*{f'{profit_prefix}Profit' if is_fill else f'Unrealized {profit_prefix}Profit'}:* "
            f"`{msg['profit_ratio']:.2%}{profit_extra}`\n"
            f"{cp_extra}"
            f"{enter_tag}"
            f"*Exit Reason:* `{msg['exit_reason']}`\n"
            f"*Direction:* `{msg['direction']}"
            f"{leverage_text}`\n"
            f"*Amount:* `{round_value(msg['amount'], 8)}`\n"
            f"*Open Rate:* `{fmt_coin2(msg['open_rate'], msg['quote_currency'])}`\n"
        )
        if msg['type'] == RPCMessageType.EXIT and msg['current_rate']:
            message += (
                f"*Current Rate:* `{fmt_coin2(msg['current_rate'], msg['quote_currency'])}`\n"
            )
            if msg['order_rate']:
                message += f"*Exit Rate:* `{fmt_coin2(msg['order_rate'], msg['quote_currency'])}`"
        elif msg['type'] == RPCMessageType.EXIT_FILL:
            message += f"*Exit Rate:* `{fmt_coin2(msg['close_rate'], msg['quote_currency'])}`"

        if is_sub_trade:
            stake_amount_fiat = self.__format_profit_fiat(msg, 'stake_amount')

            rem = fmt_coin(msg['stake_amount'], msg['quote_currency'])
            message += f"\n*Remaining:* `{rem}{stake_amount_fiat}`"
        else:
            message += f"\n*Duration:* `{duration} ({duration_min:.1f} min)`"
        return message

    def __format_profit_fiat(
        self, msg: RPCExitMsg, key: Literal['stake_amount', 'profit_amount', 'cumulative_profit']
    ) -> str:
        """
        Format Fiat currency to append to regular profit output
        """
        profit_fiat_extra = ''
        if self._rpc._fiat_converter and (fiat_currency := msg.get('fiat_currency')):
            profit_fiat = self._rpc._fiat_converter.convert_amount(
                msg[key], msg['stake_currency'], fiat_currency
            )
            profit_fiat_extra = f" / {profit_fiat:.3f} {fiat_currency}"
        return profit_fiat_extra

    def compose_message(self, msg: RPCSendMsg) -> str | None:
        if msg['type'] == RPCMessageType.ENTRY or msg['type'] == RPCMessageType.ENTRY_FILL:
            message = self._format_entry_msg(msg)

        elif msg['type'] == RPCMessageType.EXIT or msg['type'] == RPCMessageType.EXIT_FILL:
            message = self._format_exit_msg(msg)

        elif (
            msg['type'] == RPCMessageType.ENTRY_CANCEL or msg['type'] == RPCMessageType.EXIT_CANCEL
        ):
            message_side = 'enter' if msg['type'] == RPCMessageType.ENTRY_CANCEL else 'exit'
            message = (
                f"\N{WARNING SIGN} *{self._exchange_from_msg(msg)}:* "
                f"Cancelling {'partial ' if msg.get('sub_trade') else ''}"
                f"{message_side} Order for {msg['pair']} "
                f"(#{msg['trade_id']}). Reason: {msg['reason']}."
            )

        elif msg['type'] == RPCMessageType.PROTECTION_TRIGGER:
            message = (
                f"*Protection* triggered due to {msg['reason']}. "
                f"`{msg['pair']}` will be locked until `{msg['lock_end_time']}`."
            )

        elif msg['type'] == RPCMessageType.PROTECTION_TRIGGER_GLOBAL:
            message = (
                f"*Protection* triggered due to {msg['reason']}. "
                f"*All pairs* will be locked until `{msg['lock_end_time']}`."
            )

        elif msg['type'] == RPCMessageType.STATUS:
            message = f"*Status:* `{msg['status']}`"

        elif msg['type'] == RPCMessageType.WARNING:
            message = f"\N{WARNING SIGN} *Warning:* `{msg['status']}`"
        elif msg['type'] == RPCMessageType.EXCEPTION:
            # Errors will contain exceptions, which are wrapped in triple ticks.
            message = f"\N{WARNING SIGN} *ERROR:* \n {msg['status']}"

        elif msg['type'] == RPCMessageType.STARTUP:
            message = f"{msg['status']}"
        elif msg['type'] == RPCMessageType.STRATEGY_MSG:
            message = f"{msg['msg']}"
        else:
            logger.debug('Unknown message type: %s', msg['type'])
            return None
        return message

    def _message_loudness(self, msg: RPCSendMsg) -> str:
        """Determine the loudness of the message - on, off or silent"""
        default_noti = 'on'

        msg_type = msg['type']
        noti = ''
        if msg['type'] == RPCMessageType.EXIT or msg['type'] == RPCMessageType.EXIT_FILL:
            sell_noti = (
                self._config['telegram'].get('notification_settings', {}).get(str(msg_type), {})
            )

            # For backward compatibility sell still can be string
            if isinstance(sell_noti, str):
                noti = sell_noti
            else:
                default_noti = sell_noti.get('*', default_noti)
                noti = sell_noti.get(str(msg['exit_reason']), default_noti)
        else:
            noti = (
                self._config['telegram']
                .get('notification_settings', {})
                .get(str(msg_type), default_noti)
            )

        return noti

    def send_msg(self, msg: RPCSendMsg) -> None:
        """Send a message to telegram channel"""
        noti = self._message_loudness(msg)

        if noti == 'off':
            logger.info(f"Notification '{msg['type']}' not sent.")
            # Notification disabled
            return

        message = self.compose_message(deepcopy(msg))
        if message:
            asyncio.run_coroutine_threadsafe(
                self._send_msg(message, disable_notification=(noti == 'silent')), self._loop
            )

    def _get_exit_emoji(self, msg):
        """
        Get emoji for exit-messages
        """

        if float(msg['profit_ratio']) >= 0.05:
            return '\N{ROCKET}'
        elif float(msg['profit_ratio']) >= 0.0:
            return '\N{EIGHT SPOKED ASTERISK}'
        elif msg['exit_reason'] == 'stop_loss':
            return '\N{WARNING SIGN}'
        else:
            return '\N{CROSS MARK}'

    def _prepare_order_details(self, filled_orders: list, quote_currency: str, is_open: bool):
        """
        Prepare details of trade with entry adjustment enabled
        """
        lines_detail: list[str] = []
        if len(filled_orders) > 0:
            first_avg = filled_orders[0]['safe_price']
        order_nr = 0
        for order in filled_orders:
            lines: list[str] = []
            if order['is_open'] is True:
                continue
            order_nr += 1
            wording = 'Entry' if order['ft_is_entry'] else 'Exit'

            cur_entry_amount = order['filled'] or order['amount']
            cur_entry_average = order['safe_price']
            lines.append('  ')
            lines.append(f"*{wording} #{order_nr}:*")
            if order_nr == 1:
                lines.append(
                    f"*Amount:* {round_value(cur_entry_amount, 8)} "
                    f"({fmt_coin(order['cost'], quote_currency)})"
                )
                lines.append(f"*Average Price:* {round_value(cur_entry_average, 8)}")
            else:
                # TODO: This calculation ignores fees.
                price_to_1st_entry = (cur_entry_average - first_avg) / first_avg
                if is_open:
                    lines.append('({})'.format(dt_humanize_delta(order['order_filled_date'])))
                lines.append(
                    f"*Amount:* {round_value(cur_entry_amount, 8)} "
                    f"({fmt_coin(order['cost'], quote_currency)})"
                )
                lines.append(
                    f"*Average {wording} Price:* {round_value(cur_entry_average, 8)} "
                    f"({price_to_1st_entry:.2%} from 1st entry rate)"
                )
                lines.append(f"*Order Filled:* {order['order_filled_date']}")

            lines_detail.append('\n'.join(lines))

        return lines_detail

    @authorized_only
    async def _order(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /order.
        Returns the orders of the trade
        :param bot: telegram bot
        :param update: message update
        :return: None
        """

        trade_ids = []
        if context.args and len(context.args) > 0:
            trade_ids = [int(i) for i in context.args if i.isnumeric()]

        results = self._rpc._rpc_trade_status(trade_ids=trade_ids)
        for r in results:
            lines = ['*Order List for Trade #*`{trade_id}`']

            lines_detail = self._prepare_order_details(
                r['orders'], r['quote_currency'], r['is_open']
            )
            lines.extend(lines_detail if lines_detail else '')
            await self.__send_order_msg(lines, r)

    async def __send_order_msg(self, lines: list[str], r: dict[str, Any]) -> None:
        """
        Send status message.
        """
        msg = ''

        for line in lines:
            if line:
                if (len(msg) + len(line) + 1) < MAX_MESSAGE_LENGTH:
                    msg += line + '\n'
                else:
                    await self._send_msg(msg.format(**r))
                    msg = '*Order List for Trade #*`{trade_id}` - continued\n' + line + '\n'

        await self._send_msg(msg.format(**r))

    @authorized_only
    async def _status(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /status.
        Returns the current TradeThread status
        :param bot: telegram bot
        :param update: message update
        :return: None
        """

        if context.args and 'table' in context.args:
            await self._status_table(update, context)
            return
        else:
            await self._status_msg(update, context)

    async def _status_msg(self, update: Update, context: CallbackContext) -> None:
        """
        handler for `/status` and `/status <id>`.

        """
        # Check if there's at least one numerical ID provided.
        # If so, try to get only these trades.
        trade_ids = []
        if context.args and len(context.args) > 0:
            trade_ids = [int(i) for i in context.args if i.isnumeric()]

        results = self._rpc._rpc_trade_status(trade_ids=trade_ids)
        position_adjust = self._config.get('position_adjustment_enable', False)
        max_entries = self._config.get('max_entry_position_adjustment', -1)
        for r in results:
            r['open_date_hum'] = dt_humanize_delta(r['open_date'])
            r['num_entries'] = len([o for o in r['orders'] if o['ft_is_entry']])
            r['num_exits'] = len(
                [
                    o
                    for o in r['orders']
                    if not o['ft_is_entry'] and not o['ft_order_side'] == 'stoploss'
                ]
            )
            r['exit_reason'] = r.get('exit_reason', '')
            r['stake_amount_r'] = fmt_coin(r['stake_amount'], r['quote_currency'])
            r['max_stake_amount_r'] = fmt_coin(
                r['max_stake_amount'] or r['stake_amount'], r['quote_currency']
            )
            r['profit_abs_r'] = fmt_coin(r['profit_abs'], r['quote_currency'])
            r['realized_profit_r'] = fmt_coin(r['realized_profit'], r['quote_currency'])
            r['total_profit_abs_r'] = fmt_coin(r['total_profit_abs'], r['quote_currency'])
            lines = [
                '*Trade ID:* `{trade_id}`' + (' `(since {open_date_hum})`' if r['is_open'] else ''),
                '*Current Pair:* {pair}',
                (
                    f"*Direction:* {'`Short`' if r.get('is_short') else '`Long`'}"
                    + ' ` ({leverage}x)`'
                    if r.get('leverage')
                    else ''
                ),
                '*Amount:* `{amount} ({stake_amount_r})`',
                '*Total invested:* `{max_stake_amount_r}`' if position_adjust else '',
                '*Enter Tag:* `{enter_tag}`' if r['enter_tag'] else '',
                '*Exit Reason:* `{exit_reason}`' if r['exit_reason'] else '',
            ]

            if position_adjust:
                max_buy_str = f"/{max_entries + 1}" if (max_entries > 0) else ''
                lines.extend(
                    [
                        '*Number of Entries:* `{num_entries}' + max_buy_str + '`',
                        '*Number of Exits:* `{num_exits}`',
                    ]
                )

            lines.extend(
                [
                    f"*Open Rate:* `{round_value(r['open_rate'], 8)}`",
                    f"*Close Rate:* `{round_value(r['close_rate'], 8)}`" if r['close_rate'] else '',
                    '*Open Date:* `{open_date}`',
                    '*Close Date:* `{close_date}`' if r['close_date'] else '',
                    (
                        f" \n*Current Rate:* `{round_value(r['current_rate'], 8)}`"
                        if r['is_open']
                        else ''
                    ),
                    ('*Unrealized Profit:* ' if r['is_open'] else '*Close Profit: *')
                    + '`{profit_ratio:.2%}` `({profit_abs_r})`',
                ]
            )

            if r['is_open']:
                if r.get('realized_profit'):
                    lines.extend(
                        [
                            '*Realized Profit:* `{realized_profit_ratio:.2%} '
                            '({realized_profit_r})`',
                            '*Total Profit:* `{total_profit_ratio:.2%} ({total_profit_abs_r})`',
                        ]
                    )

                # Append empty line to improve readability
                lines.append(' ')
                if (
                    r['stop_loss_abs'] != r['initial_stop_loss_abs']
                    and r['initial_stop_loss_ratio'] is not None
                ):
                    # Adding initial stoploss only if it is different from stoploss
                    lines.append(
                        '*Initial Stoploss:* `{initial_stop_loss_abs:.8f}` '
                        '`({initial_stop_loss_ratio:.2%})`'
                    )

                # Adding stoploss and stoploss percentage only if it is not None
                lines.append(
                    f"*Stoploss:* `{round_value(r['stop_loss_abs'], 8)}` "
                    + ('`({stop_loss_ratio:.2%})`' if r['stop_loss_ratio'] else '')
                )
                lines.append(
                    f"*Stoploss distance:* `{round_value(r['stoploss_current_dist'], 8)}` "
                    '`({stoploss_current_dist_ratio:.2%})`'
                )
                if r.get('open_orders'):
                    lines.append(
                        '*Open Order:* `{open_orders}`'
                        + ('- `{exit_order_status}`' if r['exit_order_status'] else '')
                    )

            await self.__send_status_msg(lines, r)

    async def __send_status_msg(self, lines: list[str], r: dict[str, Any]) -> None:
        """
        Send status message.
        """
        msg = ''

        for line in lines:
            if line:
                if (len(msg) + len(line) + 1) < MAX_MESSAGE_LENGTH:
                    msg += line + '\n'
                else:
                    await self._send_msg(msg.format(**r))
                    msg = '*Trade ID:* `{trade_id}` - continued\n' + line + '\n'

        await self._send_msg(msg.format(**r))

    @authorized_only
    async def _status_table(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /status table.
        Returns the current TradeThread status in table format
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        fiat_currency = self._config.get('fiat_display_currency', '')
        statlist, head, fiat_profit_sum, fiat_total_profit_sum = self._rpc._rpc_status_table(
            self._config['stake_currency'], fiat_currency
        )

        show_total = not isnan(fiat_profit_sum) and len(statlist) > 1
        show_total_realized = (
            not isnan(fiat_total_profit_sum) and len(statlist) > 1 and fiat_profit_sum
        ) != fiat_total_profit_sum
        max_trades_per_msg = 50
        """
        Calculate the number of messages of 50 trades per message
        0.99 is used to make sure that there are no extra (empty) messages
        As an example with 50 trades, there will be int(50/50 + 0.99) = 1 message
        """
        messages_count = max(int(len(statlist) / max_trades_per_msg + 0.99), 1)
        for i in range(0, messages_count):
            trades = statlist[i * max_trades_per_msg : (i + 1) * max_trades_per_msg]
            if show_total and i == messages_count - 1:
                # append total line
                trades.append(['Total', '', '', f"{fiat_profit_sum:.2f} {fiat_currency}"])
                if show_total_realized:
                    trades.append(
                        [
                            'Total',
                            '(incl. realized Profits)',
                            '',
                            f"{fiat_total_profit_sum:.2f} {fiat_currency}",
                        ]
                    )

            message = tabulate(trades, headers=head, tablefmt='simple')
            if show_total and i == messages_count - 1:
                # insert separators line between Total
                lines = message.split('\n')
                offset = 2 if show_total_realized else 1
                message = '\n'.join(lines[:-offset] + [lines[1]] + lines[-offset:])
            await self._send_msg(
                f"<pre>{message}</pre>",
                parse_mode=ParseMode.HTML,
                reload_able=True,
                callback_path='update_status_table',
                query=update.callback_query,
            )

    async def _timeunit_stats(self, update: Update, context: CallbackContext, unit: str) -> None:
        """
        Handler for /daily <n>
        Returns a daily profit (in BTC) over the last n days.
        :param bot: telegram bot
        :param update: message update
        :return: None
        """

        vals = {
            'days': TimeunitMappings('Day', 'Daily', 'days', 'update_daily', 7, '%Y-%m-%d'),
            'weeks': TimeunitMappings(
                'Monday', 'Weekly', 'weeks (starting from Monday)', 'update_weekly', 8, '%Y-%m-%d'
            ),
            'months': TimeunitMappings('Month', 'Monthly', 'months', 'update_monthly', 6, '%Y-%m'),
        }
        val = vals[unit]

        stake_cur = self._config['stake_currency']
        fiat_disp_cur = self._config.get('fiat_display_currency', '')
        try:
            timescale = int(context.args[0]) if context.args else val.default
        except (TypeError, ValueError, IndexError):
            timescale = val.default
        stats = self._rpc._rpc_timeunit_profit(timescale, stake_cur, fiat_disp_cur, unit)
        stats_tab = tabulate(
            [
                [
                    f"{period['date']:{val.dateformat}} ({period['trade_count']})",
                    f"{fmt_coin(period['abs_profit'], stats['stake_currency'])}",
                    f"{period['fiat_value']:.2f} {stats['fiat_display_currency']}",
                    f"{period['rel_profit']:.2%}",
                ]
                for period in stats['data']
            ],
            headers=[
                f"{val.header} (count)",
                f"{stake_cur}",
                f"{fiat_disp_cur}",
                'Profit %',
                'Trades',
            ],
            tablefmt='simple',
        )
        message = (
            f"<b>{val.message} Profit over the last {timescale} {val.message2}</b>:\n"
            f"<pre>{stats_tab}</pre>"
        )
        await self._send_msg(
            message,
            parse_mode=ParseMode.HTML,
            reload_able=True,
            callback_path=val.callback,
            query=update.callback_query,
        )

    @authorized_only
    async def _daily(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /daily <n>
        Returns a daily profit (in BTC) over the last n days.
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        await self._timeunit_stats(update, context, 'days')

    @authorized_only
    async def _weekly(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /weekly <n>
        Returns a weekly profit (in BTC) over the last n weeks.
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        await self._timeunit_stats(update, context, 'weeks')

    @authorized_only
    async def _monthly(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /monthly <n>
        Returns a monthly profit (in BTC) over the last n months.
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        await self._timeunit_stats(update, context, 'months')

    @authorized_only
    async def _profit(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /profit.
        Returns a cumulative profit statistics.
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        stake_cur = self._config['stake_currency']
        fiat_disp_cur = self._config.get('fiat_display_currency', '')

        start_date = datetime.fromtimestamp(0)
        timescale = None
        try:
            if context.args:
                timescale = int(context.args[0]) - 1
                today_start = datetime.combine(date.today(), datetime.min.time())
                start_date = today_start - timedelta(days=timescale)
        except (TypeError, ValueError, IndexError):
            pass

        stats = self._rpc._rpc_trade_statistics(stake_cur, fiat_disp_cur, start_date)
        profit_closed_coin = stats['profit_closed_coin']
        profit_closed_ratio_mean = stats['profit_closed_ratio_mean']
        profit_closed_percent = stats['profit_closed_percent']
        profit_closed_fiat = stats['profit_closed_fiat']
        profit_all_coin = stats['profit_all_coin']
        profit_all_ratio_mean = stats['profit_all_ratio_mean']
        profit_all_percent = stats['profit_all_percent']
        profit_all_fiat = stats['profit_all_fiat']
        trade_count = stats['trade_count']
        first_trade_date = f"{stats['first_trade_humanized']} ({stats['first_trade_date']})"
        latest_trade_date = f"{stats['latest_trade_humanized']} ({stats['latest_trade_date']})"
        avg_duration = stats['avg_duration']
        best_pair = stats['best_pair']
        best_pair_profit_ratio = stats['best_pair_profit_ratio']
        winrate = stats['winrate']
        expectancy = stats['expectancy']
        expectancy_ratio = stats['expectancy_ratio']

        if stats['trade_count'] == 0:
            markdown_msg = f"No trades yet.\n*Bot started:* `{stats['bot_start_date']}`"
        else:
            # Message to display
            if stats['closed_trade_count'] > 0:
                markdown_msg = (
                    '*ROI:* Closed trades\n'
                    f"∙ `{fmt_coin(profit_closed_coin, stake_cur)} "
                    f"({profit_closed_ratio_mean:.2%}) "
                    f"({profit_closed_percent} \N{GREEK CAPITAL LETTER SIGMA}%)`\n"
                    f"∙ `{fmt_coin(profit_closed_fiat, fiat_disp_cur)}`\n"
                )
            else:
                markdown_msg = '`No closed trade` \n'
            fiat_all_trades = (
                f"∙ `{fmt_coin(profit_all_fiat, fiat_disp_cur)}`\n" if fiat_disp_cur else ''
            )
            markdown_msg += (
                f"*ROI:* All trades\n"
                f"∙ `{fmt_coin(profit_all_coin, stake_cur)} "
                f"({profit_all_ratio_mean:.2%}) "
                f"({profit_all_percent} \N{GREEK CAPITAL LETTER SIGMA}%)`\n"
                f"{fiat_all_trades}"
                f"*Total Trade Count:* `{trade_count}`\n"
                f"*Bot started:* `{stats['bot_start_date']}`\n"
                f"*{'First Trade opened' if not timescale else 'Showing Profit since'}:* "
                f"`{first_trade_date}`\n"
                f"*Latest Trade opened:* `{latest_trade_date}`\n"
                f"*Win / Loss:* `{stats['winning_trades']} / {stats['losing_trades']}`\n"
                f"*Winrate:* `{winrate:.2%}`\n"
                f"*Expectancy (Ratio):* `{expectancy:.2f} ({expectancy_ratio:.2f})`"
            )
            if stats['closed_trade_count'] > 0:
                markdown_msg += (
                    f"\n*Avg. Duration:* `{avg_duration}`\n"
                    f"*Best Performing:* `{best_pair}: {best_pair_profit_ratio:.2%}`\n"
                    f"*Trading volume:* `{fmt_coin(stats['trading_volume'], stake_cur)}`\n"
                    f"*Profit factor:* `{stats['profit_factor']:.2f}`\n"
                    f"*Max Drawdown:* `{stats['max_drawdown']:.2%} "
                    f"({fmt_coin(stats['max_drawdown_abs'], stake_cur)})`\n"
                    f"    from `{stats['max_drawdown_start']} "
                    f"({fmt_coin(stats['drawdown_high'], stake_cur)})`\n"
                    f"    to `{stats['max_drawdown_end']} "
                    f"({fmt_coin(stats['drawdown_low'], stake_cur)})`\n"
                )
        await self._send_msg(
            markdown_msg,
            reload_able=True,
            callback_path='update_profit',
            query=update.callback_query,
        )

    @authorized_only
    async def _stats(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /stats
        Show stats of recent trades
        """
        stats = self._rpc._rpc_stats()

        reason_map = {
            'roi': 'ROI',
            'stop_loss': 'Stoploss',
            'trailing_stop_loss': 'Trail. Stop',
            'stoploss_on_exchange': 'Stoploss',
            'exit_signal': 'Exit Signal',
            'force_exit': 'Force Exit',
            'emergency_exit': 'Emergency Exit',
        }
        exit_reasons_tabulate = [
            [reason_map.get(reason, reason), sum(count.values()), count['wins'], count['losses']]
            for reason, count in stats['exit_reasons'].items()
        ]
        exit_reasons_msg = 'No trades yet.'
        for reason in chunks(exit_reasons_tabulate, 25):
            exit_reasons_msg = tabulate(reason, headers=['Exit Reason', 'Exits', 'Wins', 'Losses'])
            if len(exit_reasons_tabulate) > 25:
                await self._send_msg(f"```\n{exit_reasons_msg}```", ParseMode.MARKDOWN)
                exit_reasons_msg = ''

        durations = stats['durations']
        duration_msg = tabulate(
            [
                [
                    'Wins',
                    (
                        str(timedelta(seconds=durations['wins']))
                        if durations['wins'] is not None
                        else 'N/A'
                    ),
                ],
                [
                    'Losses',
                    (
                        str(timedelta(seconds=durations['losses']))
                        if durations['losses'] is not None
                        else 'N/A'
                    ),
                ],
            ],
            headers=['', 'Avg. Duration'],
        )
        msg = f"""```\n{exit_reasons_msg}```\n```\n{duration_msg}```"""

        await self._send_msg(msg, ParseMode.MARKDOWN)

    @authorized_only
    async def _balance(self, update: Update, context: CallbackContext) -> None:
        """Handler for /balance"""
        full_result = context.args and 'full' in context.args
        result = self._rpc._rpc_balance(
            self._config['stake_currency'], self._config.get('fiat_display_currency', '')
        )

        balance_dust_level = self._config['telegram'].get('balance_dust_level', 0.0)
        if not balance_dust_level:
            balance_dust_level = DUST_PER_COIN.get(self._config['stake_currency'], 1.0)

        output = ''
        if self._config['dry_run']:
            output += '*Warning:* Simulated balances in Dry Mode.\n'
        starting_cap = fmt_coin(result['starting_capital'], self._config['stake_currency'])
        output += f"Starting capital: `{starting_cap}`"
        starting_cap_fiat = (
            fmt_coin(result['starting_capital_fiat'], self._config['fiat_display_currency'])
            if result['starting_capital_fiat'] > 0
            else ''
        )
        output += (f" `, {starting_cap_fiat}`.\n") if result['starting_capital_fiat'] > 0 else '.\n'

        total_dust_balance = 0
        total_dust_currencies = 0
        for curr in result['currencies']:
            curr_output = ''
            if (curr['is_position'] or curr['est_stake'] > balance_dust_level) and (
                full_result or curr['is_bot_managed']
            ):
                if curr['is_position']:
                    curr_output = (
                        f"*{curr['currency']}:*\n"
                        f"\t`{curr['side']}: {curr['position']:.8f}`\n"
                        f"\t`Est. {curr['stake']}: "
                        f"{fmt_coin(curr['est_stake'], curr['stake'], False)}`\n"
                    )
                else:
                    est_stake = fmt_coin(
                        curr['est_stake' if full_result else 'est_stake_bot'], curr['stake'], False
                    )

                    curr_output = (
                        f"*{curr['currency']}:*\n"
                        f"\t`Available: {curr['free']:.8f}`\n"
                        f"\t`Balance: {curr['balance']:.8f}`\n"
                        f"\t`Pending: {curr['used']:.8f}`\n"
                        f"\t`Bot Owned: {curr['bot_owned']:.8f}`\n"
                        f"\t`Est. {curr['stake']}: {est_stake}`\n"
                    )

            elif curr['est_stake'] <= balance_dust_level:
                total_dust_balance += curr['est_stake']
                total_dust_currencies += 1

            # Handle overflowing message length
            if len(output + curr_output) >= MAX_MESSAGE_LENGTH:
                await self._send_msg(output)
                output = curr_output
            else:
                output += curr_output

        if total_dust_balance > 0:
            output += (
                f"*{total_dust_currencies} Other "
                f"{plural(total_dust_currencies, 'Currency', 'Currencies')} "
                f"(< {balance_dust_level} {result['stake']}):*\n"
                f"\t`Est. {result['stake']}: "
                f"{fmt_coin(total_dust_balance, result['stake'], False)}`\n"
            )
        tc = result['trade_count'] > 0
        stake_improve = f" `({result['starting_capital_ratio']:.2%})`" if tc else ''
        fiat_val = f" `({result['starting_capital_fiat_ratio']:.2%})`" if tc else ''
        value = fmt_coin(result['value' if full_result else 'value_bot'], result['symbol'], False)
        total_stake = fmt_coin(
            result['total' if full_result else 'total_bot'], result['stake'], False
        )
        output += (
            f"\n*Estimated Value{' (Bot managed assets only)' if not full_result else ''}*:\n"
            f"\t`{result['stake']}: {total_stake}`{stake_improve}\n"
            f"\t`{result['symbol']}: {value}`{fiat_val}\n"
        )
        await self._send_msg(
            output, reload_able=True, callback_path='update_balance', query=update.callback_query
        )

    @authorized_only
    async def _start(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /start.
        Starts TradeThread
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        msg = self._rpc._rpc_start()
        await self._send_msg(f"Status: `{msg['status']}`")

    @authorized_only
    async def _stop(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /stop.
        Stops TradeThread
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        msg = self._rpc._rpc_stop()
        await self._send_msg(f"Status: `{msg['status']}`")

    @authorized_only
    async def _reload_config(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /reload_config.
        Triggers a config file reload
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        msg = self._rpc._rpc_reload_config()
        await self._send_msg(f"Status: `{msg['status']}`")

    @authorized_only
    async def _stopentry(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /stop_buy.
        Sets max_open_trades to 0 and gracefully sells all open trades
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        msg = self._rpc._rpc_stopentry()
        await self._send_msg(f"Status: `{msg['status']}`")

    @authorized_only
    async def _reload_trade_from_exchange(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /reload_trade <tradeid>.
        """
        if not context.args or len(context.args) == 0:
            raise RPCException('Trade-id not set.')
        trade_id = int(context.args[0])
        msg = self._rpc._rpc_reload_trade_from_exchange(trade_id)
        await self._send_msg(f"Status: `{msg['status']}`")

    @authorized_only
    async def _force_exit(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /forceexit <id>.
        Sells the given trade at current price
        :param bot: telegram bot
        :param update: message update
        :return: None
        """

        if context.args:
            trade_id = context.args[0]
            await self._force_exit_action(trade_id)
        else:
            fiat_currency = self._config.get('fiat_display_currency', '')
            try:
                statlist, _, _, _ = self._rpc._rpc_status_table(
                    self._config['stake_currency'], fiat_currency
                )
            except RPCException:
                await self._send_msg(msg='No open trade found.')
                return
            trades = []
            for trade in statlist:
                trades.append((trade[0], f"{trade[0]} {trade[1]} {trade[2]} {trade[3]}"))

            trade_buttons = [
                InlineKeyboardButton(text=trade[1], callback_data=f"force_exit__{trade[0]}")
                for trade in trades
            ]
            buttons_aligned = self._layout_inline_keyboard(trade_buttons, cols=1)

            buttons_aligned.append(
                [InlineKeyboardButton(text='Cancel', callback_data='force_exit__cancel')]
            )
            await self._send_msg(msg='Which trade?', keyboard=buttons_aligned)

    async def _force_exit_action(self, trade_id: str):
        if trade_id != 'cancel':
            try:
                loop = asyncio.get_running_loop()
                # Workaround to avoid nested loops
                await loop.run_in_executor(None, safe_async_db(self._rpc._rpc_force_exit), trade_id)
            except RPCException as e:
                await self._send_msg(str(e))

    async def _force_exit_inline(self, update: Update, _: CallbackContext) -> None:
        if update.callback_query:
            query = update.callback_query
            if query.data and '__' in query.data:
                # Input data is "force_exit__<tradid|cancel>"
                trade_id = query.data.split('__')[1].split(' ')[0]
                if trade_id == 'cancel':
                    await query.answer()
                    await query.edit_message_text(text='Force exit canceled.')
                    return
                trade: Trade | None = Trade.get_trades(trade_filter=Trade.id == trade_id).first()
                await query.answer()
                if trade:
                    await query.edit_message_text(
                        text=f"Manually exiting Trade #{trade_id}, {trade.pair}"
                    )
                    await self._force_exit_action(trade_id)
                else:
                    await query.edit_message_text(text=f"Trade {trade_id} not found.")


    @authorized_only
    async def _hedge_open(self, update: Update, context: CallbackContext) -> None:
        """
        列出所有 hedge_open 的对冲配置，点击进入详情
        """
        state_file = 'user_data/strategy_state_production.json'
        try:
            with open(state_file, 'r') as f:
                st = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            st = {}

        data = st.get('hedge_open', {}) or {}
        if not data:
            await self._send_msg('当前没有对冲配置（hedge_open）。')
            return

        buttons = []
        for pair, v in sorted(data.items()):
            side = v.get('direction', 'long')
            ep = v.get('entry_price') or v.get('entries')[0] if v.get('entries') else None
            ep_txt = f"{ep}" if isinstance(ep, (int, float)) else '—'
            btn_text = f"{pair} [hedge:{side}] EP:{ep_txt}"
            buttons.append(
                InlineKeyboardButton(text=btn_text, callback_data=f"hedge_select__{pair}")
            )
        rows = self._layout_inline_keyboard(buttons, cols=1)
        rows.append([InlineKeyboardButton(text='刷新', callback_data='update_hedge_list')])

        await self._send_msg(
            '请选择一个对冲配置查看 / 编辑：',
            keyboard=rows,
            query=getattr(update, 'callback_query', None),
        )

    @authorized_only
    async def _hedge_open_view(self, update: Update, context: CallbackContext) -> None:
        query = update.callback_query
        _, pair = query.data.split('__', 1)

        state_file = 'user_data/strategy_state_production.json'
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                st = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            st = {}

        it = (st.get('hedge_open', {}) or {}).get(pair)
        if not it:
            await self._send_msg('未找到该对冲配置（可能已被删除）。', query=query)
            return

        side = it.get('direction', 'long')
        size = it.get('size', 0)
        lev  = it.get('leverage', 1)
        ep   = it.get('entry_price')
        eps  = it.get('entries') or []
        tps  = it.get('exit_points') or []
        sl   = it.get('stop_loss')
        ts   = it.get('timestamp')
        ts   = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') if ts else '—'

        def fmt_list(a):
            if not a:
                return '（未设）'
            out = []
            for x in a:
                out.append('—' if x is None else f'{x:g}')
            return escape(', '.join(out))

        # 所有插值 escape
        pair_e = escape(pair)
        side_e = escape(str(side))
        size_e = escape(f'{size:g}')
        lev_e  = escape(f'{lev:g}')
        ep_e   = escape('—' if ep is None else f'{ep:g}')
        eps_e  = fmt_list(eps)
        tps_e  = fmt_list(tps)
        sl_e   = escape('（未设）' if sl is None else f'{sl:g}')
        ts_e   = escape(ts)

        # ✅ 不使用 <br>，全部用 \n
        msg = (
            f"🛡 <b>{pair_e}</b> 对冲配置\n"
            f"• 方向：<code>{side_e}</code>   • 仓位：<code>{size_e}</code>   • 杠杆：<code>{lev_e}</code>\n"
            f"• 最新 entry_price：<code>{ep_e}</code>\n"
            f"• entries（多点位）：<code>{eps_e}</code>\n"
            f"• TP1/2/3：<code>{tps_e}</code>\n"
            f"• SL：<code>{sl_e}</code>\n"
            f"• 更新时间：<code>{ts_e}</code>\n\n"
            f"<b>修改方法（回消息）：</b>\n"
            f"1）五值（单入场）：<code>entry tp1 tp2 tp3 sl</code>\n"
            f"2）键值对：<code>entries=...</code>（或 <code>ep=</code> 支持多值，逗号分隔/[]） "
            f"<code>tp=...</code> / <code>tp1=</code>/<code>tp2=</code>/<code>tp3=</code> <code>sl=</code>\n"
            f"   例：<code>entries=[100750,99550,98250] tp=103050,104350,105850 sl=97450</code>\n"
            f"⚠ 对冲不会立即下单，机器人在运行中自行判断是否开启。"
        )

        kb = [
            [InlineKeyboardButton('✏️ 修改参数', callback_data=f"hedge_edit__{pair}")],
            [InlineKeyboardButton('🗑 删除对冲配置', callback_data=f"hedge_delete__{pair}")],
            [InlineKeyboardButton('⬅️ 返回列表', callback_data='update_hedge_list')],
            [InlineKeyboardButton('🔁 刷新本页', callback_data=f"hedge_select__{pair}")],
        ]

        await self._send_msg(msg, parse_mode=ParseMode.HTML, keyboard=kb, query=query)


    @authorized_only
    async def _hedge_edit_inline(self, update: Update, context: CallbackContext) -> None:
        query = update.callback_query
        _, pair = query.data.split('__', 1)

        key = (query.message.chat_id, query.from_user.id)
        if not hasattr(self, '_pending_hedge_edit'):
            self._pending_hedge_edit = {}
        self._pending_hedge_edit[key] = {'pair': pair, 'ts': time.time()}

        tip = (
            '请回复新的参数以更新对冲配置：\n'
            '• 五值（单入场）：`entry tp1 tp2 tp3 sl`\n'
            '• 或键值对：`entries=...`（或 `ep=` 支持多值） `tp=...`/`tp1=`/`tp2=`/`tp3=` `sl=`\n'
            '示例：\n'
            '  `entries=[100750,99550,98250] tp=103050,104350,105850 sl=97450`\n'
            '  `100750 103050 104350 105850 97450`\n'
            '⚠ 注意：对冲只写配置，不会立即下单。'
        )
        await query.message.reply_text(tip, parse_mode=ParseMode.MARKDOWN, reply_markup=ForceReply(selective=True))

    @authorized_only
    async def on_hedge_edit_reply(self, update: Update, context: CallbackContext) -> None:
        msg = update.message
        key = (msg.chat_id, msg.from_user.id)
        pending = getattr(self, '_pending_hedge_edit', {}).get(key)
        if not pending:
            return

        # 超时 3 分钟
        if time.time() - pending['ts'] > 180:
            self._pending_hedge_edit.pop(key, None)
            await msg.reply_text('编辑已超时，请重新点击“✏️ 修改参数”。')
            return

        pair = pending['pair']

        # 读取当前
        state_file = 'user_data/strategy_state_production.json'
        try:
            with open(state_file, 'r') as f:
                st = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            st = {}
        current = (st.get('hedge_open', {}) or {}).get(pair)
        if not current:
            self._pending_hedge_edit.pop(key, None)
            await msg.reply_text('未找到该对冲配置，可能已被删除。')
            return

        try:
            # 解析输入（沿用你已有的解析器）
            entry_price, tps, sl = self._parse_edit_input(msg.text, current)

            # 读取旧值（保留 size/lev/方向）
            size = float(current.get('size', 0))
            lev  = float(current.get('leverage', 1))
            side = SignalDirection(current.get('direction', 'long'))

            # TP 清洗与排序（long 升序 / short 降序），不足 3 个用旧值补齐
            tps_clean = [x for x in tps if x is not None]
            if len(tps_clean) != 3:
                old = current.get('exit_points', [])
                while len(tps_clean) < 3 and len(old) > len(tps_clean):
                    tps_clean.append(old[len(tps_clean)])
                if len(tps_clean) != 3:
                    raise ValueError('TP 数量不足（需要 3 个）。')
            tps_clean = sorted(tps_clean, reverse=(side == SignalDirection.SHORT))

            # entries：从 free-text 中抽。优先 ep= / entries= / 五值第一位
            entries = self._extract_entries_from_text(msg.text, fallback_entry=entry_price)

            # 保存（仅写配置，不会触发下单）
            await self._update_manual_trade_config(
                pair, size, lev, tps_clean, float(sl), side,
                entry_price=entry_price, is_update=True, is_hedge=True, entries=entries
            )

            await msg.reply_text('对冲配置已更新 ✅')

            # 回到详情页
            class _Q:
                pass
            q = _Q()
            q.data = f"hedge_select__{pair}"
            q.message = msg
            fake_cb = Update(update.update_id, callback_query=q)
            await self._hedge_open_view(update=fake_cb, context=context)

        except Exception as e:
            await msg.reply_text(f'解析或保存失败：{e}\n'
                                '示例：`100750 103050 104350 105850 97450` 或 `entries=[100750,99550] tp=103050,104350,105850 sl=97450`',
                                parse_mode=ParseMode.MARKDOWN)
        finally:
            self._pending_hedge_edit.pop(key, None)

    def _extract_entries_from_text(self, text: str, fallback_entry: float | None) -> list[float | None]:
        """
        支持 entries=[a,b] / ep=a,b / 五值第一位 / 'null'/'market' 视为 None
        """
        s = text.strip().lower()
        # entries=[...]
        m = re.search(r'entries\s*=\s*\[([^\]]*)\]', s)
        if m:
            items = [x.strip() for x in m.group(1).split(',') if x.strip()!='']
            out = []
            for x in items:
                if x in ('null', 'none', 'market'): out.append(None)
                else: out.append(float(x))
            return out

        # ep=a,b,c
        m = re.search(r'\b(ep|entries)\s*=\s*([^\s\]]+)', s)
        if m:
            seq = m.group(2).strip().strip('[]')
            items = [x.strip() for x in seq.split(',') if x.strip()!='']
            out = []
            for x in items:
                if x in ('null','none','market'): out.append(None)
                else: out.append(float(x))
            return out

        # 五值（entry tp1 tp2 tp3 sl）
        toks = re.split(r'[\s,]+', s)
        if len(toks) >= 1:
            try:
                first = toks[0]
                if first in ('null','none','market'):
                    return [None]
                return [float(first)]
            except Exception:
                pass

        # 兜底：如果有 fallback_entry，就用它作为单一 entries
        return [fallback_entry] if fallback_entry is not None else []

    @authorized_only
    async def _hedge_delete(self, update: Update, context: CallbackContext) -> None:
        query = update.callback_query
        _, pair = query.data.split('__', 1)

        state_file = 'user_data/strategy_state_production.json'
        try:
            with open(state_file, 'r') as f:
                st = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            st = {}

        hedge = st.get('hedge_open', {}) or {}
        if pair not in hedge:
            await self._send_msg(f'{pair} 的对冲配置不存在或已删除。', query=query)
            return

        # 删除并落盘
        hedge.pop(pair, None)
        st['hedge_open'] = hedge
        with open(state_file, 'w') as f:
            json.dump(st, f, indent=4)

        # 同步内存
        if hasattr(self._rpc._freqtrade, 'strategy'):
            self._rpc._freqtrade.strategy.hedge_open = hedge

        await self._send_msg(f'已删除 {pair} 的对冲配置。', query=query)

        # 返回列表
        await self._hedge_open(update=update, context=context)

    @authorized_only
    async def _manual_open(self, update: Update, context: CallbackContext) -> None:
        """
        列出所有 manual_open 的手动单，点击进入详情
        """
        state_file = 'user_data/strategy_state_production.json'
        try:
            with open(state_file, 'r') as f:
                st = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            st = {}

        data = st.get('manual_open', {})
        if not data:
            await self._send_msg('当前没有手动单（manual_open）配置。')
            return

        # 生成按钮
        buttons = []
        for pair, v in sorted(data.items()):
            side = v.get('direction', 'long')
            ep = v.get('entry_price')
            ep_txt = f"{ep}" if isinstance(ep, (int, float)) else '市价'
            btn_text = f"{pair} [{side}] EP:{ep_txt}"
            buttons.append(
                InlineKeyboardButton(text=btn_text, callback_data=f"manual_select__{pair}")
            )
        rows = self._layout_inline_keyboard(buttons, cols=1)
        rows.append([InlineKeyboardButton(text='刷新', callback_data='update_manual_list')])

        await self._send_msg(
            '请选择一个手动单查看 / 编辑：',
            keyboard=rows,
            query=update.callback_query,        # 保留即可
        )

    @authorized_only
    async def _manual_open_view(self, update: Update, context: CallbackContext) -> None:
        query = update.callback_query
        pair = query.data.split('__', 1)[1]

        state_file = 'user_data/strategy_state_production.json'
        try:
            with open(state_file, 'r') as f:
                st = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            st = {}

        item = st.get('manual_open', {}).get(pair)
        if not item:
            await self._send_msg(f'{pair} 不在 manual_open 中。', query=query)
            return

        side = item.get('direction', 'long')
        size = item.get('size')
        lev = item.get('leverage')
        ep = item.get('entry_price')
        entries = item.get('entries')
        tps = item.get('exit_points', [])
        sl = item.get('stop_loss')
        ts = datetime.fromtimestamp(item.get('timestamp', datetime.now().timestamp())).strftime('%Y-%m-%d %H:%M:%S')

        tps_fmt = ','.join([str(x) for x in tps]) if tps else '(未设)'
        ep_txt = f"{ep}" if isinstance(ep, (int, float)) else '市价'

        msg = (
            f"📌 *{pair}*  手动单详情\n"
            f"• 方向：`{side}`   • 杠杆：`{lev}`   • 仓位：`{size}`\n"
            f"• 进场价（enter price）：`{ep_txt}`\n"
            f"• entries（多点位）：`{','.join([str(x) for x in entries]) if entries else '(未设)'}`\n"
            f"• TP1/2/3：`{tps_fmt}`\n"
            f"• SL：`{sl}`\n"
            f"• 设置时间：`{ts}`\n\n"
            f"✏️ *修改方法*：\n"
            f"1）五值：`entry tp1 tp2 tp3 sl`\n"
            f"2）或键值对：`entry=... tp1=... tp2=... tp3=... sl=... size=... lev=...`\n"
            f"   （可只改其中一部分）\n"
        )

        kb = [
            [InlineKeyboardButton('✏️ 修改参数', callback_data=f"manual_edit__{pair}")],
            [InlineKeyboardButton('⬅️ 返回列表', callback_data='update_manual_list')],
            [InlineKeyboardButton('🔁 刷新本页', callback_data=f"manual_select__{pair}")],
        ]
        # 计算 chat_id 以便 fallback 发送
        chat_id = getattr(getattr(update, 'effective_chat', None), 'id', None) \
            or getattr(getattr(query, 'message', None), 'chat_id', None)

        if query is not None and hasattr(query, 'edit_message_text'):
            # 有真实的 CallbackQuery，走“编辑原消息”
            await self._send_msg(msg, parse_mode=ParseMode.MARKDOWN, keyboard=kb, query=query)
        else:
            # 否则新发一条（不要传 query）
            from telegram import InlineKeyboardMarkup
            await self._app.bot.send_message(
                chat_id=chat_id,
                text=msg,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=InlineKeyboardMarkup(kb),
            )

    @authorized_only
    async def _manual_edit_inline(self, update: Update, context: CallbackContext) -> None:
        query = update.callback_query
        pair = query.data.split('__', 1)[1]

        key = (query.message.chat_id, query.from_user.id)
        self._pending_manual_edit[key] = {'pair': pair, 'ts': time.time()}

        tip = (
            '请回复新的参数：\n'
            '• 五值：`entry tp1 tp2 tp3 sl`\n'
            '• 或键值对：`entry=... tp1=... tp2=... tp3=... sl=...`\n'
            '示例：`71000 72000 73500 75000 69500`（含 entry）\n'
            '或：`tp1=72000 tp2=73500 sl=69500`（只改部分）'
        )
        await query.message.reply_text(tip, parse_mode=ParseMode.MARKDOWN, reply_markup=ForceReply(selective=True))

    def _parse_edit_input(self, text: str, current: dict) -> tuple[list[float], list[float], float, float | None, float | None]:
        """
        解析用户输入，支持修改价格、仓位(size)和杠杆(lev)。

        返回: (entries, tps[3], sl, size, lev)

        支持的键值对 (支持 entries 逗号分隔):
        - entry=2600,2650,2700  (支持多条目)
        - tp1=... tp2=... tp3=... sl=...
        - size=... lev=...

        支持的纯数字模式 (逗号视为分隔符):
        - 7 个数: entry tp1 tp2 tp3 sl size lev
        - 6 个数: tp1 tp2 tp3 sl size lev (entry 不改)
        - 5 个数: entry tp1 tp2 tp3 sl
        - 4 个数: tp1 tp2 tp3 sl
        """

        # 1. 原始分割：先按空格分，保留 token 内部的逗号 (用于处理 entry=1,2,3)
        raw_tokens = text.strip().split()

        # --- 获取当前默认值 ---
        entries = current.get('entries')
        if not entries:
            ep = current.get('entry_price')
            entries = [ep] if ep is not None else []

        tps = list(current.get('exit_points', [None, None, None]))
        if len(tps) < 3:
            tps += [None] * (3 - len(tps))

        sl = current.get('stop_loss')
        size = current.get('size')
        lev = current.get('leverage')

        # --- 2. 模式 A：键值对优先 (包含 '=') ---
        if any('=' in t for t in raw_tokens):
            for t in raw_tokens:
                if '=' not in t:
                    continue
                k, v = t.split('=', 1)
                k = k.lower()

                # 尝试解析 Value
                # 特殊处理：entries 可能包含逗号
                if k in ('entry', 'price', 'ep', 'entries'):
                    try:
                        # 将 "2600,2650" 分割并转为 float 列表
                        new_entries = [float(x) for x in v.split(',') if x.strip()]
                        if new_entries:
                            entries = new_entries
                    except ValueError:
                        continue
                else:
                    # 其他字段通常是单数值
                    try:
                        val = float(v)
                    except ValueError:
                        continue

                    if k == 'tp1':
                        tps[0] = val
                    elif k == 'tp2':
                        tps[1] = val
                    elif k == 'tp3':
                        tps[2] = val
                    elif k in ('sl', 'stop', 'stop_loss'):
                        sl = val
                    elif k in ('size', 'amount', 'qty'):
                        size = val
                    elif k in ('lev', 'leverage'):
                        lev = int(val)

            return entries, [tps[0], tps[1], tps[2]], sl, size, lev

        # --- 3. 模式 B：纯数字序列 ---
        # 只有在没发现 '=' 时才进入这里
        # 这里将逗号替换为空格，兼容 "entry, tp1, tp2" 这种输入习惯
        normalized_text = text.replace(',', ' ')
        nums = []
        for t in normalized_text.split():
            try:
                nums.append(float(t))
            except ValueError:
                pass

        count = len(nums)

        if count == 7:
            # Entry, TP1, TP2, TP3, SL, Size, Lev
            entries = [nums[0]]
            tps = [nums[1], nums[2], nums[3]]
            sl = nums[4]
            size = nums[5]
            lev = nums[6]

        elif count == 6:
            # TP1, TP2, TP3, SL, Size, Lev
            tps = [nums[0], nums[1], nums[2]]
            sl = nums[3]
            size = nums[4]
            lev = nums[5]

        elif count == 5:
            # Entry, TP1, TP2, TP3, SL
            entries = [nums[0]]
            tps = [nums[1], nums[2], nums[3]]
            sl = nums[4]

        elif count == 4:
            # TP1, TP2, TP3, SL
            tps = [nums[0], nums[1], nums[2]]
            sl = nums[3]

        else:
            raise ValueError(f'参数数量({count})不匹配。请检查输入格式或使用键值对(key=value)。')

        return entries, tps, sl, size, lev

    def _parse_num_list(self, s: str) -> list[float]:
        s = s.strip()
        if s.startswith('[') and s.endswith(']'):
            s = s[1:-1]
        parts = re.split(r'[,\s]+', s.strip())
        out = []
        for p in parts:
            if not p:
                continue
            out.append(float(p))
        return out

    async def _text_router(self, update: Update, context: CallbackContext) -> None:
        if not update.message:
            return
        key = (update.message.chat_id, update.message.from_user.id)

        # ① /manual 编辑优先
        if getattr(self, '_pending_manual_edit', None) and self._pending_manual_edit.get(key):
            await self.on_manual_edit_reply(update, context)
            return

        # ② /monitor 编辑其次
        if getattr(self, '_pending_monitor_edit', None) and self._pending_monitor_edit.get(key):
            await self.on_monitor_edit_reply(update, context)
            return

        # ④ hedge 编辑
        if getattr(self, '_pending_hedge_edit', None) and self._pending_hedge_edit.get(key):
            await self.on_hedge_edit_reply(update, context)
            return

        # ③ 手动开单参数填写
        if getattr(self, '_pending_force', None) and self._pending_force.get(key):
            await self.on_force_enter_reply(update, context)
            return

        # 其他纯文本不处理
        return

    def _parse_monitor_edit_input(self, text: str, current: dict) -> tuple[list[float] | None, list[float] | None, float | None, bool | None]:
        """
        返回 (entries, tps, sl, auto)
        - 五值：entry tp1 tp2 tp3 sl -> entries=[entry], tps=[tp1,tp2,tp3], sl, auto=None
        - 键值对：entries=/ep=（多值可逗号或空格或 [a,b]），tp=/tp1-3=，sl=，auto=
        auto= 支持 true/false/1/0/yes/no/on/off（大小写不敏感）
        - 允许只改部分，未给的返回 None
        """
        text = text.strip()
        tokens = [t for t in re.split(r'\s+', text) if t]

        def _to_bool(v: str) -> bool:
            s = v.strip().lower()
            return s in ('1', 'true', 'yes', 'on')

        # 若有 '=' 走键值对
        if any('=' in t for t in tokens):
            entries = None
            tps = [None, None, None]
            sl = None
            auto = None
            tp_bulk = None
            for tok in tokens:
                if '=' not in tok:
                    continue
                k, v = tok.split('=', 1)
                k = k.lower().strip()
                v = v.strip()
                try:
                    if k in ('entries', 'ep', 'entry', 'eps'):
                        entries = self._parse_num_list(v)
                    elif k in ('tp', 'tps'):
                        tp_bulk = self._parse_num_list(v)
                    elif k == 'tp1':
                        tps[0] = float(v)
                    elif k == 'tp2':
                        tps[1] = float(v)
                    elif k == 'tp3':
                        tps[2] = float(v)
                    elif k in ('sl', 'stop', 'stop_loss'):
                        sl = float(v)
                    elif k in ('auto',):
                        auto = _to_bool(v)
                except ValueError:
                    pass

            # 合并单独 tp1-3 与 tp= 批量
            if tp_bulk is not None:
                while len(tp_bulk) < 3:
                    tp_bulk.append(None)
                tps = tp_bulk[:3]

            # 若 tps 三个都是 None，则表示不改
            if all(x is None for x in tps):
                tps = None
            else:
                cur = (current.get('exit_points') or [None, None, None]) + [None] * 3
                tps = [tps[i] if tps[i] is not None else cur[i] for i in range(3)]
            return entries, tps, sl, auto

        # 否则尝试纯数字
        nums = []
        for t in tokens:
            try:
                nums.append(float(t))
            except ValueError:
                pass
        if len(nums) == 5:
            entry, tp1, tp2, tp3, sl = nums
            return [entry], [tp1, tp2, tp3], sl, None
        elif len(nums) == 4:
            tp1, tp2, tp3, sl = nums
            return None, [tp1, tp2, tp3], sl, None
        else:
            raise ValueError('参数数量不正确。请输入 5 个数（含 entry）或 4 个数（不含 entry），或使用键值对。')

    async def _update_coin_monitoring_config(
        self,
        pair: str,
        idx: int,
        entry_points: list[float] | None,
        exit_points: list[float] | None,
        sl: float | None,
        auto: bool | None = None,   # <== 新增
    ):
        state_file = 'user_data/strategy_state_production.json'
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                st = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            st = {}

        if 'coin_monitoring' not in st:
            st['coin_monitoring'] = {}
        if pair not in st['coin_monitoring'] or not isinstance(st['coin_monitoring'][pair], list):
            st['coin_monitoring'][pair] = []

        while len(st['coin_monitoring'][pair]) <= idx:
            st['coin_monitoring'][pair].append({})

        item = st['coin_monitoring'][pair][idx]
        direction = item.get('direction', 'long')

        if entry_points is not None:
            item['entry_points'] = entry_points
        if exit_points is not None:
            eps = [x for x in exit_points if x is not None]
            if direction == 'short':
                eps = sorted(eps, reverse=True)
            else:
                eps = sorted(eps)
            item['exit_points'] = eps
        if sl is not None:
            item['stop_loss'] = sl
        if auto is not None:                # <== 新增
            item['auto'] = bool(auto)

        item['timestamp'] = datetime.now().timestamp()

        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(st, f, indent=4, ensure_ascii=False)

        if hasattr(self._rpc._freqtrade, 'strategy'):
            self._rpc._freqtrade.strategy.coin_monitoring = st['coin_monitoring']

        await self._send_msg(f'已更新 {pair}（#{idx}）的 coin_monitoring ✅')

    @authorized_only
    async def on_manual_edit_reply(self, update: Update, context: CallbackContext) -> None:
        msg = update.message
        key = (msg.chat_id, msg.from_user.id)
        pending = self._pending_manual_edit.get(key)
        if not pending:
            return  # 不是我们的编辑流程

        # （可选）超时 3 分钟
        if time.time() - pending['ts'] > 180:
            self._pending_manual_edit.pop(key, None)
            await msg.reply_text('编辑已超时，请重新点击“✏️ 修改参数”。')
            return

        pair = pending['pair']

        # 读现有
        state_file = 'user_data/strategy_state_production.json'
        try:
            with open(state_file, 'r') as f:
                st = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            st = {}
        current = st.get('manual_open', {}).get(pair)
        if not current:
            await msg.reply_text('未找到该手动单，可能已被移除。')
            self._pending_manual_edit.pop(key, None)
            return

        try:
            entries, tps, sl, size, lev = self._parse_edit_input(msg.text, current)

            # 用原来的 size/lev/方向
            side = SignalDirection(current.get('direction', 'long'))

            # 根据方向做排序（long 升序，short 降序）
            tps_clean = [x for x in tps if x is not None]
            if len(tps_clean) != 3:
                # 保证三档 TP
                # 缺失的直接沿用旧值
                old = current.get('exit_points', [])
                while len(tps_clean) < 3 and old:
                    tps_clean.append(old[len(tps_clean)])
                if len(tps_clean) != 3:
                    raise ValueError('TP 数量不足（需要 3 个）。')

            if side == SignalDirection.SHORT:
                tps_clean = sorted(tps_clean, reverse=True)
            else:
                tps_clean = sorted(tps_clean)

            # 调用统一的保存逻辑（会更新 JSON 和内存，并提示）
            await self._update_manual_trade_config(
                pair, size, lev, tps_clean, float(sl), side, entries=entries, is_update=True
            )

            await msg.reply_text('已更新 ✅')
            # 回到详情页
            fake_cb = deepcopy(update)
            # 构造一个简易的 callback_query 供 _manual_open_view 复用（也可以直接再次发送 /manualopen）
            if hasattr(fake_cb, 'callback_query') and fake_cb.callback_query:
                fake_cb.callback_query.data = f"manual_select__{pair}"
            else:
                # 简化：直接调用详情
                class _Q:
                    pass
                q = _Q()
                q.data = f"manual_select__{pair}"
                q.message = msg
                fake_cb = Update(update.update_id, callback_query=q)
            await self._manual_open_view(update=fake_cb, context=context)
        except Exception as e:
            await msg.reply_text(f'解析或保存失败：{e}\n'
                                 '示例：`71000 72000 73500 75000 69500` 或 `tp1=72000 tp2=73500 sl=69500`',
                                 parse_mode=ParseMode.MARKDOWN)
        finally:
            self._pending_manual_edit.pop(key, None)

    @authorized_only
    async def _monitor_view(self, update: Update, context: CallbackContext) -> None:
        query = update.callback_query
        _, pair, idx_s = query.data.split('__', 2)
        idx = int(idx_s)

        state_file = 'user_data/strategy_state_production.json'
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                st = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            st = {}

        items = (st.get('coin_monitoring', {}) or {}).get(pair) or []
        if idx < 0 or idx >= len(items):
            await self._send_msg(f'{pair} 的索引 #{idx} 不存在。', query=query)
            return
        it = items[idx]

        side = it.get('direction', 'long')
        auto = '是' if it.get('auto') else '否'
        eps = it.get('entry_points') or []
        tps = it.get('exit_points') or []
        sl = it.get('stop_loss')
        ts = it.get('timestamp')
        ts = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') if ts else '—'

        def fmt_list(a):
            return ', '.join(f'{x:g}' for x in a) if a else '（未设）'

        msg = (
            f"📈 *{pair}*  监控详情（#{idx})\n"
            f"• 方向：`{side}`   • 自动：`{auto}`\n"
            f"• 入场价（entries）：`{fmt_list(eps)}`\n"
            f"• TP1/2/3：`{fmt_list(tps)}`\n"
            f"• SL：`{sl if sl is not None else '（未设）'}`\n"
            f"• 设置时间：`{ts}`\n\n"
            f"✏️ *修改方法*：\n"
            f"1）五值（单入场）：`entry tp1 tp2 tp3 sl`\n"
            f"2）键值对：`entries=...` 或 `ep=...`（可多值），`tp=...`（或 `tp1= tp2= tp3=`），`sl=`\n"
            f"   例：`entries=[71000,70800] tp=72000,73500,75000 sl=69500`\n"
            f"3）手动入场方式：\n<pre><code>/force{side} {pair.replace('/USDT:USDT', '')} 100 1 {tps[0]} {tps[1]} {tps[2]} {sl} {eps[0]}</code></pre>\n"
        )

        kb = [
            [InlineKeyboardButton('✏️ 修改参数', callback_data=f"monitor_edit__{pair}__{idx}")],
            [InlineKeyboardButton('⚙️ 重新计算', callback_data=f"monitor_recalculate__{pair}__{idx}")],
            [InlineKeyboardButton('⬅️ 返回列表', callback_data='update_monitor_list')],
            [InlineKeyboardButton('🔁 刷新本页', callback_data=f"monitor_select__{pair}__{idx}")],
        ]
        # 计算 chat_id 以便 fallback 发送
        chat_id = getattr(getattr(update, 'effective_chat', None), 'id', None) \
            or getattr(getattr(query, 'message', None), 'chat_id', None)

        if query is not None and hasattr(query, 'edit_message_text'):
            # 有真实的 CallbackQuery，走“编辑原消息”
            await self._send_msg(msg, parse_mode=ParseMode.MARKDOWN, keyboard=kb, query=query)
        else:
            # 否则新发一条（不要传 query）
            from telegram import InlineKeyboardMarkup
            await self._app.bot.send_message(
                chat_id=chat_id,
                text=msg,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=InlineKeyboardMarkup(kb),
            )

    @authorized_only
    async def _monitor_edit_inline(self, update: Update, context: CallbackContext) -> None:
        query = update.callback_query
        _, pair, idx_s = query.data.split('__', 2)
        idx = int(idx_s)

        key = (query.message.chat_id, query.from_user.id)
        self._pending_monitor_edit[key] = {'pair': pair, 'idx': idx, 'ts': time.time()}

        tip = (
            '请回复新的参数：\n'
            '• 五值（单入场）：`entry tp1 tp2 tp3 sl`\n'
            '• 或键值对：`entries=...`（或 `ep=` 支持多值） `tp=...`/`tp1=`/`tp2=`/`tp3=` `sl=` `auto=`\n'
            '示例：\n'
            '  `entries=[1.47,1.468] tp=1.49,1.507,1.526 sl=1.446 auto=true`\n'
            '  `71000 72000 73500 75000 69500`'
        )
        await query.message.reply_text(tip, parse_mode=ParseMode.MARKDOWN, reply_markup=ForceReply(selective=True))

    @authorized_only
    async def _monitor_recalculate_inline(self, update: Update, context: CallbackContext) -> None:
        query = update.callback_query
        _, pair, idx_s = query.data.split('__', 2)
        idx = int(idx_s)
        for config in self._rpc._freqtrade.strategy.coin_monitoring[pair]:
            config['auto_initialized'] = False
        self._rpc._freqtrade.strategy.reload_coin_monitoring(pair)
        await self._send_msg('已更新 ✅')


    @authorized_only
    async def _monitor_list(self, update: Update, context: CallbackContext) -> None:
        state_file = 'user_data/strategy_state_production.json'
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                st = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            st = {}

        cm = st.get('coin_monitoring', {})
        if not cm:
            await self._send_msg('当前没有 coin_monitoring 配置。', query=update.callback_query)
            return

        buttons = []
        for pair, items in sorted(cm.items()):
            # items 是列表，逐条列出
            for idx, it in enumerate(items):
                side = it.get('direction', 'long')
                auto = 'A' if it.get('auto') else 'M'
                eps = it.get('entry_points') or []
                ep_txt = ','.join(f'{x:g}' for x in eps[:3]) + ('...' if len(eps) > 3 else '') if eps else '—'
                btn_text = f"{pair} [{side}/{auto}] EP:{ep_txt} #{idx}"
                buttons.append(InlineKeyboardButton(text=btn_text, callback_data=f"monitor_select__{pair}__{idx}"))

        rows = self._layout_inline_keyboard(buttons, cols=1)
        rows.append([InlineKeyboardButton('刷新', callback_data='update_monitor_list')])

        await self._send_msg('请选择一个监控项查看 / 编辑：', keyboard=rows, query=update.callback_query)

    @authorized_only
    async def on_force_enter_reply(self, update: Update, context: CallbackContext) -> None:
        msg = update.message
        key = (msg.chat_id, msg.from_user.id)
        pending = self._pending_force.get(key)
        if not pending:
            return  # 不是我们的收集流程，忽略

        # （可选）超时 2 分钟
        if time.time() - pending['ts'] > 120:
            del self._pending_force[key]
            await msg.reply_text('已超时，请重新点击按钮发起。')
            return

        parts = msg.text.strip().split()
        if len(parts) < 6:
            await msg.reply_text('参数不足，请按格式：<size> <leverage> <tp1> <tp2> <tp3> <sl> [price]')
            return

        try:
            size = float(parts[0])
            leverage = float(parts[1])
            tp1 = float(parts[2])
            tp2 = float(parts[3])
            tp3 = float(parts[4])
            sl = float(parts[5])
            price = float(parts[6]) if len(parts) > 6 else None

            pair = _normalize_pair(pending['pair'])
            order_side = SignalDirection(pending['side'])

            # 写入 manual_open 配置
            await self._update_manual_trade_config(pair, size, leverage, [tp1, tp2, tp3], sl, order_side, entry_price=price, lock=True)

            # 真正强制进场
            await self._force_enter_action(
                pair, price, order_side,
                stake_amount=size, leverage=leverage,
                enter_tag=f'manual_{order_side.value}'
            )

            await self._update_manual_trade_config(pair, size, leverage, [tp1, tp2, tp3], sl, order_side, entry_price=price, is_update=True)

            await msg.reply_text('手动开单已提交 ✅')
        except Exception as e:
            await msg.reply_text(f"解析或下单失败：{e}")
            logger.exception('Force enter reply error')
        finally:
            # 清理状态
            self._pending_force.pop(key, None)

    @authorized_only
    async def on_monitor_edit_reply(self, update: Update, context: CallbackContext) -> None:
        msg = update.message
        key = (msg.chat_id, msg.from_user.id)
        pending = self._pending_monitor_edit.get(key)
        if not pending:
            return  # 不是 monitor 编辑流程

        # 超时保护（3分钟）
        if time.time() - pending['ts'] > 180:
            self._pending_monitor_edit.pop(key, None)
            await msg.reply_text('编辑已超时，请重新点击“✏️ 修改参数”。')
            return

        pair = pending['pair']
        idx = pending['idx']

        # 读取当前 coin_monitoring
        state_file = 'user_data/strategy_state_production.json'
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                st = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            st = {}

        items = (st.get('coin_monitoring', {}) or {}).get(pair) or []
        if idx < 0 or idx >= len(items):
            await msg.reply_text(f'{pair} 的索引 #{idx} 不存在。')
            self._pending_monitor_edit.pop(key, None)
            return
        current = items[idx] or {}

        try:
            entry_points, tps, sl, auto = self._parse_monitor_edit_input(msg.text, current)

            # 根据方向做 TP 排序（如果只改 TP）
            if tps is not None:
                side = current.get('direction', 'long')
                tps_clean = [x for x in tps if x is not None]
                if len(tps_clean) != 3:
                    old = current.get('exit_points', [])
                    while len(tps_clean) < 3 and len(old) > len(tps_clean):
                        tps_clean.append(old[len(tps_clean)])
                    if len(tps_clean) != 3:
                        raise ValueError('TP 数量不足（需要 3 个）。')
                if side == 'short':
                    tps_clean = sorted(tps_clean, reverse=True)
                else:
                    tps_clean = sorted(tps_clean)
            else:
                tps_clean = None

            await self._update_coin_monitoring_config(
                pair=pair,
                idx=int(idx),
                entry_points=entry_points,
                exit_points=tps_clean,
                sl=sl,
                auto=auto,  # <== 传入
            )

            await msg.reply_text('已更新 ✅')

            # 刷新详情页
            class _Q:
                pass
            q = _Q()
            q.data = f"monitor_select__{pair}__{idx}"
            q.message = msg
            fake_cb = Update(update.update_id, callback_query=q)
            await self._monitor_view(update=fake_cb, context=context)

        except Exception as e:
            await msg.reply_text(
                f'解析或保存失败：{e}\n'
                '示例：`71000 72000 73500 75000 69500` 或 `entries=[71000,70800] tp=72000,73500,75000 sl=69500 auto=false`',
                parse_mode=ParseMode.MARKDOWN
            )
        finally:
            self._pending_monitor_edit.pop(key, None)

    async def _force_enter_action(self, pair, price: float | None, order_side: SignalDirection, stake_amount: float, leverage: float, enter_tag: str):
        if pair != 'cancel':
            try:

                @safe_async_db
                def _force_enter():
                    kwargs = dict(
                        pair=pair,
                        price=price,
                        order_side=order_side.value,      # 确保传字符串
                        stake_amount=stake_amount,
                        leverage=leverage,
                        enter_tag=enter_tag,
                    )
                    logger.info(f'Force enter called with args: {kwargs}')
                    if price is not None:
                        # 只对“提供了 price 的手动单”用限价
                        with _temp_entry_type(self._rpc._freqtrade.strategy, 'limit'):
                            self._rpc._rpc_force_entry(**kwargs)
                    else:
                        # 没给 price -> 仍走市价
                        self._rpc._rpc_force_entry(**kwargs)

                loop = asyncio.get_running_loop()
                # Workaround to avoid nested loops
                await loop.run_in_executor(None, _force_enter)
            except RPCException as e:
                logger.exception('Forcebuy error!')
                await self._send_msg(str(e), ParseMode.HTML)

    async def _force_enter_inline(self, update: Update, _: CallbackContext) -> None:
        if update.callback_query:
            query = update.callback_query
            payload = query.data.split('__')[1]  # "force_enter__<pair|cancel>_||_<side>"
            if payload == 'cancel':
                await query.answer()
                await query.edit_message_text(text='Force enter canceled.')
                return

            if '_||_' in payload:
                pair, side = payload.split('_||_')
                order_side = SignalDirection(side)
                await query.answer()
                await query.edit_message_text(text=f"准备手动开单 {order_side.value} / {pair}")

                # 记录待填写状态（按 chat_id + user_id）
                key = (query.message.chat_id, query.from_user.id)
                self._pending_force[key] = {'pair': pair, 'side': order_side.value, 'ts': time.time()}

                # 让用户在下一条消息里输入参数
                await query.message.reply_text(
                    '请按格式回复：<size> <leverage> <tp1> <tp2> <tp3> <sl> [price]\n'
                    '例如：100 3 71000 72000 73500 69500',
                    reply_markup=ForceReply(selective=True)
                )

    def _has_open_trade(self, pair: str) -> bool:
        """
        判断是否存在指定交易对的未平仓单。
        优先使用 RPC 的 open_trades；若不可用则回退到本地 Trade 模型。
        """
        # 1) RPC 路径（如果可用）
        try:
            if hasattr(self, '_rpc') and hasattr(self._rpc, '_rpc_open_trades'):
                open_trades = self._rpc._rpc_open_trades() or []
                for t in open_trades:
                    p = t['pair'] if isinstance(t, dict) else getattr(t, 'pair', None)
                    if p == pair:
                        return True
        except Exception:
            # 不影响后续回退
            logger.exception('RPC open_trades check failed.')

        # 2) 本地 ORM 回退
        try:
            from freqtrade.persistence.models import Trade  # 若不可用会抛异常
            # get_trades_proxy(is_open=True) 常见返回迭代器/列表
            for t in Trade.get_trades_proxy(is_open=True):
                if getattr(t, 'pair', None) == pair:
                    return True
        except Exception:
            logger.exception('DB open_trades check failed.')

        return False


    @staticmethod
    def _layout_inline_keyboard(
        buttons: list[InlineKeyboardButton], cols=3
    ) -> list[list[InlineKeyboardButton]]:
        return [buttons[i : i + cols] for i in range(0, len(buttons), cols)]

    async def _update_manual_trade_config(
        self,
        pair: str,
        size: float,
        leverage: float,
        tps: list[float],
        sl: float | None,
        order_side: SignalDirection,
        entry_price: float | None = None,   # 既有：新增记录/或继承旧值
        raw_submission: dict | None = None, # 既有
        is_update: bool = False,            # 既有
        # ==== 新增：保持默认不动老逻辑 ====
        is_hedge: bool = False,                     # 新增：对冲写入 hedge_open，默认 False
        entries: list[float | None] | None = None,  # 新增：极简多点位数组（可含 None 表示市价）
        lock: bool = False,                        # 预留：未来可加锁定功能
    ):
        """
        Updates strategy_state.json with manual/hedge trade configuration.
        - 若当前有未平仓单：在 <root_key>[pair] 下追加 'scale_in'（不改主字段）。
        - 若无未平仓单：写/更新主字段（保留 entry_price 继承）。
        - root_key = 'manual_open'（默认）或 'hedge_open'（is_hedge=True）。
        - 完全保留原有行为；仅新增 entries/is_hedge 能力。
        """
        state_file = 'user_data/strategy_state_production.json'

        try:
            with open(state_file, 'r') as f:
                strategy_state = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            strategy_state = {}

        root_key = 'hedge_open' if is_hedge else 'manual_open'
        if root_key not in strategy_state:
            strategy_state[root_key] = {}

        # 旧值（用于继承 entry_price / entries / scale_in）
        old = strategy_state[root_key].get(pair, {})

        if is_update and pair not in strategy_state[root_key]:
            raise ValueError(f"No existing {root_key} entry for {pair} to update.")
        if is_update and pair in strategy_state[root_key]:
            # ======= 保持你原来的“更新主字段”逻辑 =======
            logger.info(f"Updating existing {root_key} entry for {pair}.")
            strategy_state[root_key][pair] = {
                'size': size,
                'leverage': leverage,
                'entry_price': entry_price if entry_price is not None else old.get('entry_price'),
                'exit_points': sorted(tps, reverse=(order_side == SignalDirection.SHORT)) if tps else (old.get('exit_points') or []),
                'stop_loss': sl,  # 可以为 None
                'direction': order_side.value,
                'timestamp': datetime.now().timestamp(),  # 避免陈旧
                # 兼容：若传了 entries 就更新；否则保留旧 entries
                **({'entries': entries} if entries else ({'entries': old.get('entries')} if 'entries' in old else {})),
                # 保留旧的 scale_in（如果有）
                **({'scale_in': old.get('scale_in')} if 'scale_in' in old else {}),
            }
        else:
            # ======= 新增/覆盖主字段 或 仅追加 scale_in（保留老逻辑） =======
            try:
                has_open = self._has_open_trade(pair)
            except Exception:
                has_open = False

            if has_open and not is_hedge and pair in strategy_state.get('manual_open', {}):
                # ---- A) 有未平仓单：只追加 scale_in，不动主字段（保留你的老逻辑） ----
                payload = {
                    'ts': datetime.now().timestamp(),
                    'request': {
                        'size': size,
                        'leverage': leverage,
                        'entry_price': entry_price,  # 仅记录本次请求；真实成交后你有回写
                        'tps': tps or [],
                        'sl': sl,
                        'direction': order_side.value,
                        # 新增：把 entries 也带入 scale_in（可选，便于多点位补仓）
                        'entries': entries or [],
                    }
                }
                if raw_submission:
                    payload['raw'] = raw_submission

                mo = strategy_state[root_key].setdefault(pair, {})
                mo['scale_in'] = payload  # ★ 保持你原来“单条 scale_in”语义
                # 不覆盖任何主字段
                strategy_state[root_key][pair] = mo

            else:
                # ---- B) 无未平仓单：写主字段（完全保留你既有逻辑） ----
                base = {
                    'size': size,
                    'leverage': leverage,
                    'entry_price': entry_price if entry_price is not None else old.get('entry_price'),
                    'exit_points': sorted(tps, reverse=(order_side == SignalDirection.SHORT)) if tps else (old.get('exit_points') or []),
                    'stop_loss': sl,  # 可以为 None
                    'direction': order_side.value,
                    'timestamp': datetime.now().timestamp(),  # 避免陈旧
                    'lock': lock,  # 预留字段
                }
                # 新增：如有 entries，用极简数组写入；否则保留旧 entries
                if entries is not None and len(entries) > 0:
                    base['entries'] = entries
                elif 'entries' in old:
                    base['entries'] = old['entries']

                # 保留旧的 scale_in（若之前写过）
                if 'scale_in' in old:
                    base['scale_in'] = old['scale_in']

                strategy_state[root_key][pair] = base

        # ---- 落盘 ----
        with open(state_file, 'w') as f:
            json.dump(strategy_state, f, indent=4)

        # ---- 同步内存（保持你的老逻辑）----
        if hasattr(self._rpc._freqtrade, 'strategy'):
            setattr(self._rpc._freqtrade.strategy, root_key, strategy_state[root_key])

        logger.info(f"Updated {root_key} for {pair} in {state_file} and in-memory.")
        await self._send_msg(
            f"{'HEDGE' if is_hedge else 'Manual'} trade config for {pair} has been set. The strategy will pick it up on the next tick."
        )

        # ---- 白名单（保留你的老逻辑）----
        current_whitelist = self._rpc._rpc_whitelist()['whitelist']
        if pair in current_whitelist:
            await self._send_msg(f'交易对 {pair} 已在当前白名单中')
        else:
            # with open('/freqtrade/config_production.json', 'r') as f:
            #     config = json.load(f)
            # config['exchange']['pair_whitelist'].append(pair)
            # with open('/freqtrade/config_production.json', 'w') as f:
            #     json.dump(config, f, indent=4)
            # self._rpc._rpc_reload_config()
            await self._send_msg(f'请手动将 {pair} 加入白名单， <code>/addpair {pair}</code>', parse_mode=ParseMode.HTML)


    @authorized_only
    async def _force_enter(self, update: Update, context: CallbackContext, order_side: SignalDirection, is_hedge: bool=False) -> None:
        args = context.args or []

        if len(args) >= 8:
            try:
                pair = _normalize_pair(args[0])
                size = float(args[1])
                leverage = float(args[2])
                tp1, tp2, tp3 = float(args[3]), float(args[4]), float(args[5])
                sl = float(args[6])
                entries = _parse_price_list(args[7])

                # 写配置（注意：entry_price 参数不传 -> 继承旧值）
                await self._update_manual_trade_config(
                    pair=pair, size=size, leverage=leverage,
                    tps=[tp1, tp2, tp3], sl=sl, order_side=order_side,
                    entry_price=None,
                    raw_submission={'mode': 'full_args_entries', 'args': args},
                    is_update=False, is_hedge=is_hedge, entries=entries, lock=True
                )

                if is_hedge:
                    await self._send_msg(f"{pair} 对冲计划已写入（entries={len(entries)}）。由策略自动判断是否开启。")
                    return

                # === 主方向首段执行 ===
                allocs = _allocations_for(len(entries))
                if not entries:
                    await self._send_msg(f"{pair} entries 为空，已仅写入配置，不执行下单。")
                    return

                price0 = entries[0]
                stake0 = round(size * allocs[0] / 100.0, 8)

                if price0 is None:
                    # 市价首段
                    await self._force_enter_action(
                        pair, None, order_side,
                        stake_amount=stake0, leverage=leverage,
                        enter_tag=('manual_' + order_side.value)
                    )
                    await self._update_manual_trade_config(
                        pair=pair, size=size, leverage=leverage,
                        tps=[tp1, tp2, tp3], sl=sl, order_side=order_side,
                        entry_price=None,
                        raw_submission={'mode': 'full_args_entries', 'args': args},
                        is_update=True, is_hedge=is_hedge, entries=entries
                    )
                    # entry_price 由成交回调 set_entry_price() 更新
                    await self._send_msg(f"{pair} 第一段【市价】已执行（{allocs[0]}%），其余段位仅写配置等待触发。")
                else:
                    # 限价首段（价格必须为数值）
                    p0 = float(price0)
                    await self._force_enter_action(
                        pair, p0, order_side,
                        stake_amount=stake0, leverage=leverage,
                        enter_tag=('manual_' + order_side.value)
                    )
                    await self._update_manual_trade_config(
                        pair=pair, size=size, leverage=leverage,
                        tps=[tp1, tp2, tp3], sl=sl, order_side=order_side,
                        entry_price=None,
                        raw_submission={'mode': 'full_args_entries', 'args': args},
                        is_update=True, is_hedge=is_hedge, entries=entries
                    )
                    await self._send_msg(f"{pair} 第一段【限价 {p0}】已挂单（{allocs[0]}%），其余段位仅写配置等待触发。")

                return
            except Exception as e:
                await self._send_msg(f"参数或下单错误（entries）：{e}")
                logger.exception('Error in _force_enter (entries)')
                return

        # ===== A) 全参：<pair> <size> <lev> <tp1> <tp2> <tp3> <sl> [price] -> 保持原逻辑 =====
        if len(args) >= 7:
            try:
                pair = _normalize_pair(args[0])
                size = float(args[1])
                leverage = float(args[2])
                tp1 = float(args[3])
                tp2 = float(args[4])
                tp3 = float(args[5])
                sl = float(args[6])
                price = float(args[7]) if len(args) > 7 else None

                await self._update_manual_trade_config(
                    pair, size, leverage, [tp1, tp2, tp3], sl, order_side, entry_price=price,
                    raw_submission={'mode': 'full_args', 'args': args}, lock=True
                )

                await self._force_enter_action(
                    pair, price, order_side,
                    stake_amount=size, leverage=leverage,
                    enter_tag=f'manual_{order_side.value}'
                )

                await self._update_manual_trade_config(
                    pair, size, leverage, [tp1, tp2, tp3], sl, order_side, entry_price=price,
                    raw_submission={'mode': 'full_args', 'args': args}, is_update=True
                )

                return
            except Exception as e:
                await self._send_msg(f"参数或下单错误：{e}")
                logger.exception('Error in _force_enter')
                return

        # ===== 新增：KV 模式（部分参数可缺省）：<pair> <size> <lev> key=value ... =====
        # 例：/forcelong eth 10 10 price=2700 sl=2500
        if len(args) >= 3 and any(('=' in a) for a in args[3:]):
            try:
                pair = _normalize_pair(args[0])
                size = float(args[1])
                leverage = float(args[2])

                kv = _parse_kv_tokens(args[3:])
                price = _maybe_float(kv.get('price'))
                sl = _maybe_float(kv.get('sl'))
                tps = _extract_tps_from_kv(kv)  # 可能为空 []

                # 写入 manual_open：只把提供的内容写入（未提供的就留空/None/[]）
                await self._update_manual_trade_config(
                    pair, size, leverage, tps, sl, order_side, entry_price=price,
                    raw_submission={'mode': 'kv_partial', 'args': args, 'kv': kv}, lock=True
                )

                # 下单：若提供 price 即挂限价单；否则按你 _force_enter_action 的默认行为
                await self._force_enter_action(
                    pair, price, order_side,
                    stake_amount=size, leverage=leverage,
                    enter_tag=f'manual_{order_side.value}'
                )

                await self._update_manual_trade_config(
                    pair, size, leverage, tps, sl, order_side, entry_price=price,
                    raw_submission={'mode': 'kv_partial', 'args': args, 'kv': kv}, is_update=True
                )

                # 友好回执
                tip_lines = [
                    f"已接收 KV 模式下单：{pair} {order_side.value}",
                    f"- size={size}, lev={leverage}",
                    f"- price={'未提供' if price is None else price}",
                    f"- sl={'未提供' if sl is None else sl}",
                    f"- tps={'未提供' if not tps else ','.join(map(str, tps))}",
                ]
                await self._send_msg('\n'.join(tip_lines))
                return
            except Exception as e:
                await self._send_msg(f"参数或下单错误(KV)：{e}")
                logger.exception('Error in _force_enter (kv mode)')
                return

        # ===== 保持你现有的“3 参快速进场”分支 =====
        if len(args) == 3:
            try:
                pair = _normalize_pair(args[0])
                size = float(args[1])
                leverage = float(args[2])

                await self._update_manual_trade_config(
                    pair, size, leverage, [], None, order_side, entry_price=None,
                    raw_submission={'mode': 'quick3', 'args': args}, lock=True
                )

                await self._force_enter_action(
                    pair, None, order_side,
                    stake_amount=size, leverage=leverage,
                    enter_tag=f'manual_{order_side.value}'
                )

                await self._update_manual_trade_config(
                    pair, size, leverage, [], None, order_side, entry_price=None,
                    raw_submission={'mode': 'quick3', 'args': args}, is_update=True
                )

                return
            except Exception as e:
                await self._send_msg(f"参数或下单错误：{e}")
                logger.exception('Error in _force_enter')
                return

        # ===== B) 只有 pair：保持原逻辑 =====
        if len(args) == 1:
            pair = _normalize_pair(args[0])
            key = (update.effective_chat.id, update.effective_user.id)
            self._pending_force[key] = {'pair': pair, 'side': order_side.value, 'ts': time.time()}
            tip = ('请按格式回复以下参数：\n'
                '<size> <leverage> <tp1> <tp2> <tp3> <sl> [price]\n'
                '示例：100 3 71000 72000 73500 69500')
            await context.bot.send_message(chat_id=update.effective_chat.id, text=tip, reply_markup=ForceReply(selective=True))
            return

        # ===== C) 无参数：保持原逻辑 =====
        whitelist = self._rpc._rpc_whitelist()['whitelist']
        pair_buttons = [
            InlineKeyboardButton(text=p, callback_data=f"force_enter__{p}_||_{order_side.value}")
            for p in sorted(whitelist)
        ]
        buttons_aligned = self._layout_inline_keyboard(pair_buttons)
        buttons_aligned.append([InlineKeyboardButton(text='Cancel', callback_data='force_enter__cancel')])

        await self._send_msg(msg='Which pair?', keyboard=buttons_aligned, query=update.callback_query)

    @authorized_only
    async def _hedge_force_open(self, update: Update, context: CallbackContext) -> None:
        """
        /fehg <pair> <long|short> <lev> <size> [entry] [sl] [tp1,tp2,tp3]
        例：/fehg BTC/USDT:USDT short 2 100 103150 104769 100350,98950,95550
        """
        try:
            args = context.args or []
            if len(args) < 4:
                raise RPCException('Usage: /fehg <pair> <long|short> <lev> <size> [sl] [tp1,tp2,...] [entry]')

            pair = _normalize_pair(args[0])
            direction = args[1].lower()
            lev = float(args[2]); size = float(args[3])
            sl = float(args[4]) if len(args) >= 5 and args[4].lower() != 'none' else None
            tps = []
            if len(args) >= 6 and args[5].lower() != 'none':
                tps = [float(x) for x in str(args[5]).split(',') if x]
            entry = float(args[6]) if len(args) >= 7 else self._rpc._freqtrade.strategy.hedge.fetch_last_price(pair)

            meta = {'stop_loss': sl, 'exit_points': sorted(tps, reverse=(direction=='short'))}
            self._rpc._freqtrade.strategy.hedge.open(pair, direction, lev, size, entry, meta)
            await self._update_manual_trade_config(
                pair=pair, size=size, leverage=lev,
                tps=tps, sl=sl, order_side=direction,
                entry_price=None,
                raw_submission={'mode': 'full_args_entries', 'args': args},
                is_update=False, is_hedge=True, entries=[entry],
            )
            await self._send_msg(f'✅ HEDGE OPEN {pair} {direction} lev={lev} size={size} @ {entry}\nSL={sl} TP={tps}')
        except Exception as e:
            await self._send_msg(f'❌ fehg error: {e}')

    @authorized_only
    async def _hedge_force_close(self, update: Update, context: CallbackContext) -> None:
        """
        /fxhg <pair> [percent]  —— 不填 percent 则 100%
        例：/fxhg BTC/USDT:USDT 50     # 关一半
            /fxhg BTC/USDT:USDT       # 全关
        direction 自动按活动单方向反向 reduceOnly
        """
        try:
            args = context.args or []
            if len(args) < 1:
                raise RPCException('Usage: /fxhg <pair> [percent]')
            pair = _normalize_pair(args[0])
            pct  = float(args[1]) if len(args) >= 2 else 100.0

            st = self._rpc._freqtrade.strategy
            act = st.hedge.get_active(pair)
            if not act:
                await self._send_msg('⚠️ no active hedge for this pair.')
                return
            direction = act['direction'].lower()
            lev = float(act['leverage'] or 1.0)
            px  = st.hedge.fetch_last_price(pair)

            if pct >= 100.0:
                st.hedge.close_all(pair, direction, lev, 100, px)
                await self._send_msg(f'✅ HEDGE CLOSE-ALL {pair} ({direction}) @ {px}')
            else:
                st.hedge.close_partial(pair, direction, lev, pct, px)
                await self._send_msg(f'✅ HEDGE CLOSE-PART {pair} {pct:.2f}% ({direction}) @ {px}')
        except Exception as e:
            await self._send_msg(f'❌ fxhg error: {e}')

    @authorized_only
    async def _hedge_status(self, update: Update, context: CallbackContext) -> None:
        """
        /statushg
        不带参数：列出所有 open hedge 的盈亏快照
        """
        try:
            st = self._rpc._freqtrade.strategy
            rows = st.hedge.list_open_with_pnl()
            if not rows:
                await self._send_msg('ℹ️ No active hedge orders.')
                return
            lines = []
            for r in rows:
                pnl_pct = r['pnl_pct'] * 100.0
                tps = r['meta'].get('exit_points') or []
                sl  = r['meta'].get('stop_loss', None)
                lines.append(
                    f"#{r['id']} {r['pair']} {r['direction']}/lev{r['leverage']} "
                    f"rem={r['remaining_notional']:.2f} USDT "
                    f"EP={r['entry_price']} MP={r['mark_price']} "
                    f"PnL={r['pnl']:.2f} ({pnl_pct:+.2f}%) stage={r['hedge_exit_stage']} "
                    f"SL={sl} TP={tps}"
                )
            await self._send_msg('📊 HEDGE STATUS\n' + '\n'.join(lines))
        except Exception as e:
            await self._send_msg(f'❌ statushg error: {e}')

    @authorized_only
    async def _set_manual(self, update: Update, context: CallbackContext):
        """
        /setmanual <pair|trade_id> <long|short> <tp1> <tp2> <tp3> <sl> [entry_price]

        新增：
        也支持 KV 模式，例如：
        /setmanual BTC/USDT side=long tp=71000,72000 sl=69500 entry=70550
        /setmanual BTC/USDT side=short tp1=71000 tp2=72000 sl=69500
        """
        args = context.args or []
        if not args:
            await self._send_msg(
                '用法：\n'
                '/setmanual BTC/USDT side=long tp=71000,72000 sl=69500 entries=70500,70600\n'
            )
            return

        ident = args[0]
        trade = self._get_open_trade_by_pair_or_id(ident)
        if not trade:
            await self._send_msg(f"未找到未平仓的订单：{ident}")
            return

        # A) 提供了后续参数：尝试解析（自动支持位置参数模式 + KV 模式）
        if len(args) >= 2:
            try:
                side, tps, sl, price, entries = self._parse_setmanual_params(args[1:])

                if side not in ('long', 'short'):
                    raise ValueError('方向必须是 long 或 short')

                if not tps:
                    raise ValueError('至少需要一个 TP')

                # 从 trade 中取 size / leverage
                size = float(getattr(trade, 'stake_amount', 0) or 0)
                leverage = float(getattr(trade, 'leverage', 1) or 1)

                # 1) 修改 enter_tag 并保存
                trade.enter_tag = f"manual_{side}"
                trade.commit()

                # 2) 写入/更新配置（同时会更新内存 strategy.manual_open）
                from freqtrade.enums import SignalDirection
                order_side = SignalDirection.LONG if side == 'long' else SignalDirection.SHORT

                await self._update_manual_trade_config(
                    pair=trade.pair,
                    size=size,
                    leverage=leverage,
                    tps=tps,   # 这里 tps 是 list[float]，长度可变（1~3 或更多，看你策略怎么用）
                    sl=sl,
                    order_side=order_side,
                    entry_price=(
                        float(price)
                        if price is not None
                        else float(getattr(trade, 'open_rate', 0) or 0)
                    ),
                    entries=entries,
                    is_update=True,
                )

                # 友好提示，把 TP/SL 列一下
                tps_str = ', '.join(f"{x:g}" for x in tps)
                msg = (
                    f"✅ 已将 {trade.pair} 标记为 manual_{side}。\n"
                    f"TP: {tps_str}\n"
                    f"SL: {sl:g}\n"
                    f"入场价: {price if price is not None else getattr(trade, 'open_rate', '—')}"
                )
                await self._send_msg(msg)
                return

            except Exception as e:
                logger.exception('Error in /setmanual')
                await self._send_msg(f"参数或写入错误：{e}")
                return

        # B) 只给了 ident：进入引导输入
        key = (update.effective_chat.id, update.effective_user.id)
        self._pending_force[key] = {'pair': trade.pair, 'side': 'manual_set', 'ts': time.time()}

        tip = (
            '请按格式回复以下参数（支持位置参数 或 KV 模式）：\n'
            '1) 位置参数：\n'
            '   <long|short> <tp1> <tp2> <tp3> <sl> [entry_price]\n'
            '   示例：long 71000 72000 73500 69500 70550\n\n'
            '2) KV 模式（TP 数量可变）：\n'
            '   side=<long|short> tp=71000,72000 sl=69500 [entry=70550]\n'
            '   或：side=short tp1=71000 tp2=72000 sl=69500'
        )

        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=tip,
            reply_markup=ForceReply(selective=True),
        )

    def _parse_setmanual_params(self, tokens):
        """
        统一解析 /setmanual 的参数部分（去掉第一个 ident 后的那一段）：

        返回：
            side: str ('long' / 'short')
            tps: List[float]
            sl: float
            price: Optional[float]      # 主入场价（如果有）
            entries: Optional[List[float]]  # 如果有逗号分隔的多入场价，则这里返回列表
        """

        if not tokens:
            raise ValueError('缺少参数')

        # ---------- KV 模式 ----------
        # 例如：
        #   side=long tp=71000,72000 sl=69500 entry=70550,70600
        #   side=short tp1=71000 tp2=72000 sl=69500 entries=70550,70600
        if any('=' in t for t in tokens):
            kv = {}
            for t in tokens:
                if '=' not in t:
                    raise ValueError(f'KV 模式下参数必须是 key=value 形式，收到：{t}')
                k, v = t.split('=', 1)
                k = k.strip().lower()
                v = v.strip()
                if not k:
                    raise ValueError(f'非法参数：{t}')
                kv[k] = v

            side = kv.get('side') or kv.get('s')
            if not side:
                raise ValueError('缺少 side=long|short')

            def _parse_float_list(s: str):
                return [float(x) for x in s.split(',') if x.strip()]

            # ---- TP 解析 ----
            tps: list[float] = []
            if 'tp' in kv:
                tps.extend(_parse_float_list(kv['tp']))

            for i in range(1, 10):  # 预留 tp1~tp9
                key = f'tp{i}'
                if key in kv:
                    tps.append(float(kv[key]))

            if not tps:
                raise ValueError('缺少 TP 参数（tp=... 或 tp1=...）')

            # ---- SL ----
            if 'sl' not in kv:
                raise ValueError('缺少 sl= 止损参数')
            sl = float(kv['sl'])

            # ---- entries & 主 price ----
            entries = None
            price = None

            # 优先解析 entries=...
            if 'entries' in kv:
                entries = _parse_float_list(kv['entries'])
                if not entries:
                    raise ValueError('entries= 中没有有效价格')
                price = entries[0]
            else:
                # 兼容 entry / entry_price / price
                for ek in ('entry', 'entry_price', 'price'):
                    if ek in kv:
                        raw = kv[ek]
                        if ',' in raw:
                            entries = _parse_float_list(raw)
                            if not entries:
                                raise ValueError(f'{ek}= 中没有有效价格')
                            price = entries[0]
                        else:
                            price = float(raw)
                            entries = [price]
                        break

            return side.lower(), tps, sl, price, entries

        # ---------- 位置参数模式（兼容旧用法） ----------
        # <long|short> <tp1> <tp2> <tp3> <sl> [entry_price 或 entry1,entry2...]
        if len(tokens) < 5:
            raise ValueError(
                '位置参数模式下至少需要：<long|short> <tp1> <tp2> <tp3> <sl> [entry_price]'
            )

        side = tokens[0].lower()
        if side not in ('long', 'short'):
            raise ValueError('第一个参数必须是 long 或 short')

        tp1 = float(tokens[1])
        tp2 = float(tokens[2])
        tp3 = float(tokens[3])
        sl = float(tokens[4])

        price = None
        entries = None

        if len(tokens) > 5:
            raw = tokens[5]
            if ',' in raw:
                # 多个 entry，用逗号分隔
                entries = [float(x) for x in raw.split(',') if x.strip()]
                if not entries:
                    raise ValueError('入场价列表中没有有效价格')
                price = entries[0]
            else:
                price = float(raw)
                entries = [price]

        return side, [tp1, tp2, tp3], sl, price, entries

    @authorized_only
    async def _restore_manual(self, update: Update, context: CallbackContext):
        """
        /restoremanual <pair|trade_id> <auto|monitor>
        1) auto:  enter_tag -> buy / short  (依据 trade.is_short)
        2) monitor: enter_tag -> fixed_{long|short}_entry_{<entry_points[0]>}
        其中 entry_points[0] 来自内存 strategy.coin_monitoring[pair] 中
        “第一个 direction 匹配的配置”的第一个入场点
        同时会清除 strategy_state_production.json 的 manual_open[pair]，并同步内存。
        """
        args = context.args or []
        if len(args) != 2:
            await self._send_msg('用法：/restoremanual <pair|trade_id> <auto|monitor>')
            return

        ident, mode = args[0], args[1].lower()
        if mode not in ('auto', 'monitor'):
            await self._send_msg('第二个参数必须是 auto 或 monitor')
            return

        trade = self._get_open_trade_by_pair_or_id(ident)
        if not trade:
            await self._send_msg(f"未找到未平仓的订单：{ident}")
            return

        try:
            is_short = bool(getattr(trade, 'is_short', False))
            side = 'short' if is_short else 'long'

            if mode == 'auto':
                # 恢复成自动量化买卖
                new_tag = 'short' if is_short else 'buy'
                trade.enter_tag = new_tag
                trade.update()
                cleared = self._clear_manual_open_for_pair(trade.pair)
                msg = f"已将 {trade.pair} 恢复为自动量化（enter_tag='{new_tag}'）。"
                if cleared:
                    msg += ' 已清除手动托管配置。'
                await self._send_msg(msg)
                return

            # mode == 'monitor'
            # 恢复成自动点位监控：fixed_{long|short}_entry_{price}
            first_entry = self._find_monitor_first_entry(trade.pair, side)
            if first_entry is None:
                await self._send_msg(f"未在内存 coin_monitoring 中找到 {trade.pair} 方向 {side} 的 entry_points。")
                return

            price_str = self._fmt_price_str(first_entry)
            new_tag = f"fixed_{side}_entry_{price_str}"
            trade.enter_tag = new_tag
            trade.update()
            cleared = self._clear_manual_open_for_pair(trade.pair)

            msg = f"已将 {trade.pair} 恢复为自动点位监控（enter_tag='{new_tag}'）。"
            if cleared:
                msg += ' 已清除手动托管配置。'
            await self._send_msg(msg)

        except Exception as e:
            logger.exception('Error in /restoremanual')
            await self._send_msg(f"恢复失败：{e}")


    @authorized_only
    async def _trades(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /trades <n>
        Returns last n recent trades.
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        stake_cur = self._config['stake_currency']
        try:
            nrecent = int(context.args[0]) if context.args else 10
        except (TypeError, ValueError, IndexError):
            nrecent = 10
        nonspot = self._config.get('trading_mode', TradingMode.SPOT) != TradingMode.SPOT
        trades = self._rpc._rpc_trade_history(nrecent)
        trades_tab = tabulate(
            [
                [
                    dt_humanize_delta(dt_from_ts(trade['close_timestamp'])),
                    f"{trade['pair']} (#{trade['trade_id']}"
                    f"{(' ' + ('S' if trade['is_short'] else 'L')) if nonspot else ''})",
                    f"{(trade['close_profit']):.2%} ({trade['close_profit_abs']})",
                ]
                for trade in trades['trades']
            ],
            headers=[
                'Close Date',
                'Pair (ID L/S)' if nonspot else 'Pair (ID)',
                f"Profit ({stake_cur})",
            ],
            tablefmt='simple',
        )
        message = f"<b>{min(trades['trades_count'], nrecent)} recent trades</b>:\n" + (
            f"<pre>{trades_tab}</pre>" if trades['trades_count'] > 0 else ''
        )
        await self._send_msg(message, parse_mode=ParseMode.HTML)

    @authorized_only
    async def _delete_trade(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /delete <id>.
        Delete the given trade
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        if not context.args or len(context.args) == 0:
            raise RPCException('Trade-id not set.')
        trade_id = int(context.args[0])
        msg = self._rpc._rpc_delete(trade_id)
        await self._send_msg(
            f"`{msg['result_msg']}`\n"
            'Please make sure to take care of this asset on the exchange manually.'
        )

    @authorized_only
    async def _sync_trade(self, update: Update, context: CallbackContext) -> None:
        """
        /sync_trade <trade_id|pair> [open_rate=..] [amount=..] [stake_amount=..]
        - 若为 trade_id：按该单的 pair 同步
        - 若为 pair：同步该 pair 最近一条未平仓 Trade
        - 自动映射并允许用户覆盖
        """
        if not context.args or len(context.args) == 0:
            raise RPCException('Usage: /sync_trade <trade_id|pair> [open_rate=..] [amount=..] [stake_amount=..]')

        head, *rest = context.args
        overrides = _parse_overrides(rest)

        if self._rpc._freqtrade.state != State.RUNNING:
            raise RPCException('trader is not running')

        with self._rpc._freqtrade._exit_lock:
            # 1) trade_id or pair
            trade = None
            if head.isdigit():
                trade_id = int(head)
                trade = Trade.get_trades(
                    trade_filter=[Trade.id == trade_id, Trade.is_open.is_(True)]
                ).first()
                if not trade:
                    raise RPCException(f"Invalid trade_id or trade closed: {trade_id}")
                pair = trade.pair
            else:
                pair = head
            pair = _normalize_pair(pair)
            # 2) 交易所仓位
            exchange = _build_exchange_from_config(self._rpc._freqtrade.config)
            pos = _fetch_position(exchange, pair)
            if not pos:
                # 若是 pair 模式，可能本地也没有未平仓，这里统一提示
                if trade is None:
                    await self._send_msg(f"⚠️ No exchange position for {pair}.")
                    return
                raise RPCException(f"No exchange position for {pair}.")

            # 3) 选择本地 Trade（pair 模式下找最近一条）
            if trade is None:
                trade = Trade.get_trades(
                    trade_filter=[Trade.pair == pair, Trade.is_open.is_(True)]
                ).first()
                if not trade:
                    snap = f"{pos['side']} size={pos['contracts']} entry={pos.get('entryPrice')} lev={pos.get('leverage')}"
                    await self._send_msg(f"⚠️ No local open trade for {pair}.\nExchange position: {snap}")
                    return

            # 4) 计算写回值（支持覆盖）
            entry = overrides.get('open_rate', pos.get('entryPrice'))
            amount = overrides.get('amount', pos.get('contracts'))
            lev = trade.leverage
            stake_calc = (entry * amount / lev) if (entry is not None and amount is not None) else None
            stake_amount = overrides.get('stake_amount', stake_calc)
            max_stake_amount = stake_amount * lev

            if entry is None or amount is None:
                raise RPCException('Unable to resolve open_rate/amount (exchange value missing and no override given).')

            # 5) 写回 + 提交
            before = {
                'open_rate': trade.open_rate,
                'amount': trade.amount,
                'stake_amount': getattr(trade, 'stake_amount', None),
            }
            trade.open_rate = float(entry)
            trade.amount = float(amount)
            trade.amount_requested = float(amount)
            if hasattr(trade, 'stake_amount') and stake_amount is not None:
                trade.stake_amount = float(stake_amount)
                trade.max_stake_amount = float(max_stake_amount)
                trade.open_trade_value = float(max_stake_amount)
            Trade.commit()

            after = {
                'open_rate': trade.open_rate,
                'amount': trade.amount,
                'stake_amount': getattr(trade, 'stake_amount', None),
            }

        # 6) 汇报
        msg = (
            '✅ 同步完成\n'
            f"Pair         : {pair}\n"
            f"Side         : {pos['side']}\n"
            f"open_rate    : {before['open_rate']} -> {after['open_rate']}\n"
            f"amount       : {before['amount']} -> {after['amount']}\n"
            f"stake_amount : {before['stake_amount']} -> {after['stake_amount']}"
        )
        await self._send_msg(msg)

    @authorized_only
    async def _cancel_open_order(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /cancel_open_order <id>.
        Cancel open order for tradeid
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        if not context.args or len(context.args) == 0:
            raise RPCException('Trade-id not set.')
        trade_id = int(context.args[0])
        self._rpc._rpc_cancel_open_order(trade_id)
        await self._send_msg('Open order canceled.')

    def _reset_trade_data_core(
        self,
        trade_id: int,
        exit_stage: int,
        initial_stake: float | None = None,
        use_current_stake: bool = False,
    ) -> None:
        """
        在本类中直接重置 trade 的自定义数据（不通过 RPC）：
        - 总是设置 exit_stage（若外部未传则按 Handler 默认 0）
        - initial_stake：
            * use_current_stake=True -> 取 trade.stake_amount
            * initial_stake 显式传入 -> 用传入值
            * 否则不修改
        """
        if self._rpc._freqtrade.state != State.RUNNING:
            raise RPCException('trader is not running')

        # 尽量沿用你在 cancel_open_order 中的加锁方式
        with self._rpc._freqtrade._exit_lock:
            trade = Trade.get_trades(
                trade_filter=[
                    Trade.id == trade_id,
                    Trade.is_open.is_(True),
                ]
            ).first()

            if not trade:
                logger.warning('reset_trade_data: Invalid trade_id received.')
                raise RPCException('Invalid trade_id.')

            # 1) exit_stage（存自定义数据区）
            try:
                trade.set_custom_data('exit_stage', int(exit_stage))
            except Exception as e:
                logger.exception('Failed to set custom exit_stage', exc_info=True)
                raise RPCException(f"Failed to set exit_stage: {e}")

            # 2) initial_stake（按优先级修改或保持不变）
            try:
                if use_current_stake:
                    trade.initial_stake = trade.stake_amount
                elif initial_stake is not None:
                    trade.initial_stake = float(initial_stake)
                # 未显式要求则不修改
            except Exception as e:
                logger.exception('Failed to set initial_stake', exc_info=True)
                raise RPCException(f"Failed to set initial_stake: {e}")

            Trade.commit()
            logger.info(
                'Trade %s reset done: exit_stage=%s, initial_stake=%s',
                trade_id,
                exit_stage,
                'stake_amount' if use_current_stake else ('unchanged' if initial_stake is None else initial_stake),
            )


    @authorized_only
    async def _reset_trade_data(self, update: Update, context: CallbackContext) -> None:
        """
        /reset_trade_data <trade_id> [exit_stage] [initial_stake]
        /reset_trade_data <trade_id> [exit_stage=<int>] [initial_stake[=<float>]]

        支持示例：
        /reset_trade_data 123               -> exit_stage=0；initial_stake不变
        /reset_trade_data 123 2             -> exit_stage=2；initial_stake不变
        /reset_trade_data 123 initial_stake -> exit_stage=0；initial_stake=stake_amount
        /reset_trade_data 123 initial_stake=150
                                            -> exit_stage=0；initial_stake=150
        /reset_trade_data 123 3 initial_stake
                                            -> exit_stage=3；initial_stake=stake_amount
        /reset_trade_data 123 exit_stage=2 initial_stake=100
                                            -> exit_stage=2；initial_stake=100
        """
        if not context.args:
            await self._send_msg('用法：/reset_trade_data 123')
            raise RPCException('Trade-id not set.')

        # 解析 trade_id
        try:
            trade_id = int(context.args[0])
        except ValueError:
            await self._send_msg('用法：/reset_trade_data 123 exit_stage=2 initial_stake=100')
            raise RPCException('Invalid trade_id.')

        # 解析其余参数
        exit_stage = None                # 未提供则默认 0
        initial_stake = None             # 显式提供数值时使用
        use_current_stake = False        # 用户写了 'initial_stake' 但未给值时 = True
        pos = []

        for raw in context.args[1:]:
            if '=' in raw:
                k, v = raw.split('=', 1)
                k, v = k.strip().lower(), v.strip()
                if k == 'exit_stage':
                    try:
                        exit_stage = int(v)
                    except ValueError:
                        raise RPCException('Invalid exit_stage value.')
                elif k == 'initial_stake':
                    if v == '':
                        use_current_stake = True
                    else:
                        try:
                            initial_stake = float(v)
                        except ValueError:
                            raise RPCException('Invalid initial_stake value.')
                else:
                    raise RPCException(f"Unknown parameter '{k}'.")
            else:
                pos.append(raw)

        # 位置参数： [exit_stage] [initial_stake]
        if len(pos) >= 1 and exit_stage is None:
            if pos[0].lower() == 'initial_stake':
                use_current_stake = True
            else:
                try:
                    exit_stage = int(pos[0])
                except ValueError:
                    raise RPCException('Invalid exit_stage value.')

        if len(pos) >= 2:
            if pos[1].lower() == 'initial_stake':
                use_current_stake = True
            else:
                try:
                    initial_stake = float(pos[1])
                except ValueError:
                    raise RPCException('Invalid initial_stake value.')

        if len(pos) > 2:
            raise RPCException('Too many positional arguments.')

        # 未传 exit_stage -> 默认 0
        if exit_stage is None:
            exit_stage = 0

        # 直接在本类内执行业务逻辑（不走 RPC）
        self._reset_trade_data_core(
            trade_id=trade_id,
            exit_stage=exit_stage,
            initial_stake=initial_stake,
            use_current_stake=use_current_stake,
        )

        await self._send_msg(
            f"Trade {trade_id} 已重置：exit_stage={exit_stage}，"
            f"initial_stake={'使用当前 stake_amount' if use_current_stake else ('未修改' if initial_stake is None else initial_stake)}"
        )


    @authorized_only
    async def _performance(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /performance.
        Shows a performance statistic from finished trades
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        trades = self._rpc._rpc_performance()
        output = '<b>Performance:</b>\n'
        for i, trade in enumerate(trades):
            stat_line = (
                f"{i + 1}.\t <code>{trade['pair']}\t"
                f"{fmt_coin(trade['profit_abs'], self._config['stake_currency'])} "
                f"({trade['profit_ratio']:.2%}) "
                f"({trade['count']})</code>\n"
            )

            if len(output + stat_line) >= MAX_MESSAGE_LENGTH:
                await self._send_msg(output, parse_mode=ParseMode.HTML)
                output = stat_line
            else:
                output += stat_line

        await self._send_msg(
            output,
            parse_mode=ParseMode.HTML,
            reload_able=True,
            callback_path='update_performance',
            query=update.callback_query,
        )

    @authorized_only
    async def _enter_tag_performance(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /entries PAIR .
        Shows a performance statistic from finished trades
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        pair = None
        if context.args and isinstance(context.args[0], str):
            pair = context.args[0]

        trades = self._rpc._rpc_enter_tag_performance(pair)
        output = '*Entry Tag Performance:*\n'
        for i, trade in enumerate(trades):
            stat_line = (
                f"{i + 1}.\t `{trade['enter_tag']}\t"
                f"{fmt_coin(trade['profit_abs'], self._config['stake_currency'])} "
                f"({trade['profit_ratio']:.2%}) "
                f"({trade['count']})`\n"
            )

            if len(output + stat_line) >= MAX_MESSAGE_LENGTH:
                await self._send_msg(output, parse_mode=ParseMode.MARKDOWN)
                output = stat_line
            else:
                output += stat_line

        await self._send_msg(
            output,
            parse_mode=ParseMode.MARKDOWN,
            reload_able=True,
            callback_path='update_enter_tag_performance',
            query=update.callback_query,
        )

    @authorized_only
    async def _exit_reason_performance(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /exits.
        Shows a performance statistic from finished trades
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        pair = None
        if context.args and isinstance(context.args[0], str):
            pair = context.args[0]

        trades = self._rpc._rpc_exit_reason_performance(pair)
        output = '*Exit Reason Performance:*\n'
        for i, trade in enumerate(trades):
            stat_line = (
                f"{i + 1}.\t `{trade['exit_reason']}\t"
                f"{fmt_coin(trade['profit_abs'], self._config['stake_currency'])} "
                f"({trade['profit_ratio']:.2%}) "
                f"({trade['count']})`\n"
            )

            if len(output + stat_line) >= MAX_MESSAGE_LENGTH:
                await self._send_msg(output, parse_mode=ParseMode.MARKDOWN)
                output = stat_line
            else:
                output += stat_line

        await self._send_msg(
            output,
            parse_mode=ParseMode.MARKDOWN,
            reload_able=True,
            callback_path='update_exit_reason_performance',
            query=update.callback_query,
        )

    @authorized_only
    async def _mix_tag_performance(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /mix_tags.
        Shows a performance statistic from finished trades
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        pair = None
        if context.args and isinstance(context.args[0], str):
            pair = context.args[0]

        trades = self._rpc._rpc_mix_tag_performance(pair)
        output = '*Mix Tag Performance:*\n'
        for i, trade in enumerate(trades):
            stat_line = (
                f"{i + 1}.\t `{trade['mix_tag']}\t"
                f"{fmt_coin(trade['profit_abs'], self._config['stake_currency'])} "
                f"({trade['profit_ratio']:.2%}) "
                f"({trade['count']})`\n"
            )

            if len(output + stat_line) >= MAX_MESSAGE_LENGTH:
                await self._send_msg(output, parse_mode=ParseMode.MARKDOWN)
                output = stat_line
            else:
                output += stat_line

        await self._send_msg(
            output,
            parse_mode=ParseMode.MARKDOWN,
            reload_able=True,
            callback_path='update_mix_tag_performance',
            query=update.callback_query,
        )

    @authorized_only
    async def _count(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /count.
        Returns the number of trades running
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        counts = self._rpc._rpc_count()
        message = tabulate(
            {k: [v] for k, v in counts.items()},
            headers=['current', 'max', 'total stake'],
            tablefmt='simple',
        )
        message = f"<pre>{message}</pre>"
        logger.debug(message)
        await self._send_msg(
            message,
            parse_mode=ParseMode.HTML,
            reload_able=True,
            callback_path='update_count',
            query=update.callback_query,
        )

    @authorized_only
    async def _locks(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /locks.
        Returns the currently active locks
        """
        rpc_locks = self._rpc._rpc_locks()
        if not rpc_locks['locks']:
            await self._send_msg('No active locks.', parse_mode=ParseMode.HTML)

        for locks in chunks(rpc_locks['locks'], 25):
            message = tabulate(
                [
                    [lock['id'], lock['pair'], lock['lock_end_time'], lock['reason']]
                    for lock in locks
                ],
                headers=['ID', 'Pair', 'Until', 'Reason'],
                tablefmt='simple',
            )
            message = f"<pre>{escape(message)}</pre>"
            logger.debug(message)
            await self._send_msg(message, parse_mode=ParseMode.HTML)

    @authorized_only
    async def _delete_locks(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /delete_locks.
        Returns the currently active locks
        """
        arg = context.args[0] if context.args and len(context.args) > 0 else None
        lockid = None
        pair = None
        if arg:
            try:
                lockid = int(arg)
            except ValueError:
                pair = arg

        self._rpc._rpc_delete_lock(lockid=lockid, pair=pair)
        await self._locks(update, context)

    @authorized_only
    async def _whitelist(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /whitelist
        Shows the currently active whitelist
        """
        whitelist = self._rpc._rpc_whitelist()

        if context.args:
            if 'sorted' in context.args:
                whitelist['whitelist'] = sorted(whitelist['whitelist'])
            if 'baseonly' in context.args:
                whitelist['whitelist'] = [pair.split('/')[0] for pair in whitelist['whitelist']]

        message = f"Using whitelist `{whitelist['method']}` with {whitelist['length']} pairs\n"
        message += f"`{', '.join(whitelist['whitelist'])}`"

        logger.debug(message)
        await self._send_msg(message)

    @authorized_only
    async def _blacklist(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /blacklist
        Shows the currently active blacklist
        """
        await self.send_blacklist_msg(self._rpc._rpc_blacklist(context.args))

    async def send_blacklist_msg(self, blacklist: dict):
        errmsgs = []
        for _, error in blacklist['errors'].items():
            errmsgs.append(f"Error: {error['error_msg']}")
        if errmsgs:
            await self._send_msg('\n'.join(errmsgs))

        message = f"Blacklist contains {blacklist['length']} pairs\n"
        message += f"`{', '.join(blacklist['blacklist'])}`"

        logger.debug(message)
        await self._send_msg(message)

    @authorized_only
    async def _blacklist_delete(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /bl_delete
        Deletes pair(s) from current blacklist
        """
        await self.send_blacklist_msg(self._rpc._rpc_blacklist_delete(context.args or []))

    @authorized_only
    async def _logs(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /logs
        Shows the latest logs
        """
        try:
            limit = int(context.args[0]) if context.args else 10
        except (TypeError, ValueError, IndexError):
            limit = 10
        logs = RPC._rpc_get_logs(limit)['logs']
        msgs = ''
        msg_template = '*{}* {}: {} \\- `{}`'
        for logrec in logs:
            msg = msg_template.format(
                escape_markdown(logrec[0], version=2),
                escape_markdown(logrec[2], version=2),
                escape_markdown(logrec[3], version=2),
                escape_markdown(logrec[4], version=2),
            )
            if len(msgs + msg) + 10 >= MAX_MESSAGE_LENGTH:
                # Send message immediately if it would become too long
                await self._send_msg(msgs, parse_mode=ParseMode.MARKDOWN_V2)
                msgs = msg + '\n'
            else:
                # Append message to messages to send
                msgs += msg + '\n'

        if msgs:
            await self._send_msg(msgs, parse_mode=ParseMode.MARKDOWN_V2)

    @authorized_only
    async def _edge(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /edge
        Shows information related to Edge
        """
        edge_pairs = self._rpc._rpc_edge()
        if not edge_pairs:
            message = '<b>Edge only validated following pairs:</b>'
            await self._send_msg(message, parse_mode=ParseMode.HTML)

        for chunk in chunks(edge_pairs, 25):
            edge_pairs_tab = tabulate(chunk, headers='keys', tablefmt='simple')
            message = f"<b>Edge only validated following pairs:</b>\n<pre>{edge_pairs_tab}</pre>"

            await self._send_msg(message, parse_mode=ParseMode.HTML)

    @authorized_only
    async def _help(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /help.
        Show commands of the bot
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        force_enter_text = (
            '*/forcelong <pair> [<rate>]:* `Instantly buys the given pair. '
            'Optionally takes a rate at which to buy '
            '(only applies to limit orders).` \n'
        )
        if self._rpc._freqtrade.trading_mode != TradingMode.SPOT:
            force_enter_text += (
                '*/forceshort <pair> [<rate>]:* `Instantly shorts the given pair. '
                'Optionally takes a rate at which to sell '
                '(only applies to limit orders).` \n'
            )
        message = (
            '_Bot Control_\n'
            '------------\n'
            '*/start:* `Starts the trader`\n'
            '*/stop:* `Stops the trader`\n'
            '*/stopentry:* `Stops entering, but handles open trades gracefully` \n'
            '*/forceexit <trade_id>|all:* `Instantly exits the given trade or all trades, '
            'regardless of profit`\n'
            '*/fx <trade_id>|all:* `Alias to /forceexit`\n'
            f"{force_enter_text if self._config.get('force_entry_enable', False) else ''}"
            '*/delete <trade_id>:* `Instantly delete the given trade in the database`\n'
            '*/reload_trade <trade_id>:* `Reload trade from exchange Orders`\n'
            '*/cancel_open_order <trade_id>:* `Cancels open orders for trade. '
            'Only valid when the trade has open orders.`\n'
            '*/coo <trade_id>|all:* `Alias to /cancel_open_order`\n'
            '*/whitelist [sorted] [baseonly]:* `Show current whitelist. Optionally in '
            'order and/or only displaying the base currency of each pairing.`\n'
            '*/blacklist [pair]:* `Show current blacklist, or adds one or more pairs '
            'to the blacklist.` \n'
            '*/blacklist_delete [pairs]| /bl_delete [pairs]:* '
            '`Delete pair / pattern from blacklist. Will reset on reload_conf.` \n'
            '*/reload_config:* `Reload configuration file` \n'
            "*/unlock <pair|id>:* `Unlock this Pair (or this lock id if it's numeric)`\n"
            '_Current state_\n'
            '------------\n'
            '*/show_config:* `Show running configuration` \n'
            '*/locks:* `Show currently locked pairs`\n'
            '*/balance:* `Show bot managed balance per currency`\n'
            '*/balance total:* `Show account balance per currency`\n'
            '*/logs [limit]:* `Show latest logs - defaults to 10` \n'
            '*/count:* `Show number of active trades compared to allowed number of trades`\n'
            '*/edge:* `Shows validated pairs by Edge if it is enabled` \n'
            '*/health* `Show latest process timestamp - defaults to 1970-01-01 00:00:00` \n'
            '*/marketdir [long | short | even | none]:* `Updates the user managed variable '
            'that represents the current market direction. If no direction is provided `'
            '`the currently set market direction will be output.` \n'
            '*/list_custom_data <trade_id> <key>:* `List custom_data for Trade ID & Key combo.`\n'
            '`If no Key is supplied it will list all key-value pairs found for that Trade ID.`\n'
            '_Statistics_\n'
            '------------\n'
            '*/status <trade_id>|[table]:* `Lists all open trades`\n'
            '         *<trade_id> :* `Lists one or more specific trades.`\n'
            '                        `Separate multiple <trade_id> with a blank space.`\n'
            '         *table :* `will display trades in a table`\n'
            '                `pending buy orders are marked with an asterisk (*)`\n'
            '                `pending sell orders are marked with a double asterisk (**)`\n'
            '*/entries <pair|none>:* `Shows the enter_tag performance`\n'
            '*/exits <pair|none>:* `Shows the exit reason performance`\n'
            '*/mix_tags <pair|none>:* `Shows combined entry tag + exit reason performance`\n'
            '*/trades [limit]:* `Lists last closed trades (limited to 10 by default)`\n'
            '*/profit [<n>]:* `Lists cumulative profit from all finished trades, '
            'over the last n days`\n'
            '*/performance:* `Show performance of each finished trade grouped by pair`\n'
            '*/daily <n>:* `Shows profit or loss per day, over the last n days`\n'
            '*/weekly <n>:* `Shows statistics per week, over the last n weeks`\n'
            '*/monthly <n>:* `Shows statistics per month, over the last n months`\n'
            '*/stats:* `Shows Wins / losses by Sell reason as well as '
            'Avg. holding durations for buys and sells.`\n'
            '*/help:* `This help message`\n'
            '*/version:* `Show version`\n'
        )

        await self._send_msg(message, parse_mode=ParseMode.MARKDOWN)

    @authorized_only
    async def _health(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /health
        Shows the last process timestamp
        """
        health = self._rpc.health()
        message = f"Last process: `{health['last_process_loc']}`\n"
        message += f"Initial bot start: `{health['bot_start_loc']}`\n"
        message += f"Last bot restart: `{health['bot_startup_loc']}`"
        await self._send_msg(message)

    @authorized_only
    async def _version(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /version.
        Show version information
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        strategy_version = self._rpc._freqtrade.strategy.version()
        version_string = f"*Version:* `{__version__}`"
        if strategy_version is not None:
            version_string += f"\n*Strategy version: * `{strategy_version}`"

        await self._send_msg(version_string)

    @authorized_only
    async def _show_config(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /show_config.
        Show config information information
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        val = RPC._rpc_show_config(self._config, self._rpc._freqtrade.state)

        if val['trailing_stop']:
            sl_info = (
                f"*Initial Stoploss:* `{val['stoploss']}`\n"
                f"*Trailing stop positive:* `{val['trailing_stop_positive']}`\n"
                f"*Trailing stop offset:* `{val['trailing_stop_positive_offset']}`\n"
                f"*Only trail above offset:* `{val['trailing_only_offset_is_reached']}`\n"
            )

        else:
            sl_info = f"*Stoploss:* `{val['stoploss']}`\n"

        if val['position_adjustment_enable']:
            pa_info = (
                f"*Position adjustment:* On\n"
                f"*Max enter position adjustment:* `{val['max_entry_position_adjustment']}`\n"
            )
        else:
            pa_info = '*Position adjustment:* Off\n'

        await self._send_msg(
            f"*Mode:* `{'Dry-run' if val['dry_run'] else 'Live'}`\n"
            f"*Exchange:* `{val['exchange']}`\n"
            f"*Market: * `{val['trading_mode']}`\n"
            f"*Stake per trade:* `{val['stake_amount']} {val['stake_currency']}`\n"
            f"*Max open Trades:* `{val['max_open_trades']}`\n"
            f"*Minimum ROI:* `{val['minimal_roi']}`\n"
            f"*Entry strategy:* ```\n{json.dumps(val['entry_pricing'])}```\n"
            f"*Exit strategy:* ```\n{json.dumps(val['exit_pricing'])}```\n"
            f"{sl_info}"
            f"{pa_info}"
            f"*Timeframe:* `{val['timeframe']}`\n"
            f"*Strategy:* `{val['strategy']}`\n"
            f"*Current state:* `{val['state']}`"
        )

    @authorized_only
    async def _list_custom_data(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /list_custom_data <id> <key>.
        List custom_data for specified trade (and key if supplied).
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        try:
            if not context.args or len(context.args) == 0:
                raise RPCException('Trade-id not set.')
            trade_id = int(context.args[0])
            key = None if len(context.args) < 2 else str(context.args[1])

            results = self._rpc._rpc_list_custom_data(trade_id, key)
            messages = []
            if len(results) > 0:
                messages.append('Found custom-data entr' + ('ies: ' if len(results) > 1 else 'y: '))
                for result in results:
                    lines = [
                        f"*Key:* `{result['cd_key']}`",
                        f"*ID:* `{result['id']}`",
                        f"*Trade ID:* `{result['ft_trade_id']}`",
                        f"*Type:* `{result['cd_type']}`",
                        f"*Value:* `{result['cd_value']}`",
                        f"*Create Date:* `{format_date(result['created_at'])}`",
                        f"*Update Date:* `{format_date(result['updated_at'])}`",
                    ]
                    # Filter empty lines using list-comprehension
                    messages.append('\n'.join([line for line in lines if line]))
                for msg in messages:
                    if len(msg) > MAX_MESSAGE_LENGTH:
                        msg = 'Message dropped because length exceeds '
                        msg += f"maximum allowed characters: {MAX_MESSAGE_LENGTH}"
                        logger.warning(msg)
                    await self._send_msg(msg)
            else:
                message = f"Didn't find any custom-data entries for Trade ID: `{trade_id}`"
                message += f" and Key: `{key}`." if key is not None else ''
                await self._send_msg(message)

        except RPCException as e:
            await self._send_msg(str(e))

    async def _update_msg(
        self,
        query: CallbackQuery,
        msg: str,
        callback_path: str = '',
        reload_able: bool = False,
        parse_mode: str = ParseMode.MARKDOWN,
        photo: BytesIO | None = None,
        document: io.BufferedReader | None = None,
        filename: str | None = None,
        keyboard: list[list[InlineKeyboardButton]] | None = None,   # <== 新增
    ) -> None:
        # 组合 reply_markup：优先合并你自定义的 keyboard，再附加 Refresh（如果需要）
        if reload_able:
            refresh_row = [InlineKeyboardButton('Refresh', callback_data=callback_path)]
            if keyboard:
                reply_markup = InlineKeyboardMarkup(keyboard + [refresh_row])
            else:
                reply_markup = InlineKeyboardMarkup([refresh_row])
        else:
            if keyboard:
                reply_markup = InlineKeyboardMarkup(keyboard)
            else:
                reply_markup = InlineKeyboardMarkup([[]])

        msg += f"\nUpdated: {datetime.now().ctime()}"
        if not query.message:
            return

        try:
            if photo:
                await query.edit_message_media(
                    InputMediaPhoto(photo, caption=msg, parse_mode=parse_mode),
                    reply_markup=reply_markup,
                )
            elif document and filename:
                from telegram import InputFile
                await query.edit_message_media(
                    InputMediaDocument(InputFile(document, filename=filename), caption=msg, parse_mode=parse_mode), reply_markup=reply_markup
                )
            else:
                await query.edit_message_text(text=msg, parse_mode=parse_mode, reply_markup=reply_markup)
        except BadRequest as e:
            if 'not modified' in e.message.lower():
                pass
            else:
                logger.warning('TelegramError: %s', e.message)
        except TelegramError as telegram_err:
            logger.warning('TelegramError: %s! Giving up on that message.', telegram_err.message)

    async def _send_msg(
        self,
        msg: str,
        parse_mode: str = ParseMode.MARKDOWN,
        disable_notification: bool = False,
        keyboard: list[list[InlineKeyboardButton]] | None = None,
        callback_path: str = '',
        reload_able: bool = False,
        query: CallbackQuery | None = None,
        photo: BytesIO | None = None,
        document: io.BufferedReader | None = None,
        filename: str | None = None,
    ) -> None:
        """
        Send given markdown message
        :param msg: message
        :param bot: alternative bot
        :param parse_mode: telegram parse mode
        :return: None
        """
        reply_markup: InlineKeyboardMarkup | ReplyKeyboardMarkup
        if query:
            await self._update_msg(
                query=query,
                msg=msg,
                parse_mode=parse_mode,
                callback_path=callback_path,
                reload_able=reload_able,
                photo=photo,
                document=document,
                filename=filename,
                keyboard=keyboard,
            )
            return
        if reload_able and self._config['telegram'].get('reload', True):
            reply_markup = InlineKeyboardMarkup(
                [[InlineKeyboardButton('Refresh', callback_data=callback_path)]]
            )
        else:
            if keyboard is not None:
                reply_markup = InlineKeyboardMarkup(keyboard)
            else:
                reply_markup = ReplyKeyboardMarkup(self._keyboard, resize_keyboard=True)
        try:
            try:
                if photo:
                    await self._app.bot.send_photo(
                        self._config['telegram']['chat_id'],
                        photo=photo,
                        caption=msg,
                        parse_mode=parse_mode,
                        reply_markup=reply_markup,
                        disable_notification=disable_notification,
                        message_thread_id=self._config['telegram'].get('topic_id'),
                    )
                elif document and filename:
                    await self._app.bot.send_document(
                        self._config['telegram']['chat_id'],
                        document=document,
                        filename=filename,
                        parse_mode=parse_mode,
                        reply_markup=reply_markup,
                        disable_notification=disable_notification,
                        message_thread_id=self._config['telegram'].get('topic_id'),
                    )
                else:
                    await self._app.bot.send_message(
                        self._config['telegram']['chat_id'],
                        text=msg,
                        parse_mode=parse_mode,
                        reply_markup=reply_markup,
                        disable_notification=disable_notification,
                        message_thread_id=self._config['telegram'].get('topic_id'),
                    )
            except NetworkError as network_err:
                # Sometimes the telegram server resets the current connection,
                # if this is the case we send the message again.
                logger.warning(
                    'Telegram NetworkError: %s! Trying one more time.', network_err.message
                )
                if photo:
                    await self._app.bot.send_photo(
                        self._config['telegram']['chat_id'],
                        photo=photo,
                        caption=msg,
                        parse_mode=parse_mode,
                        reply_markup=reply_markup,
                        disable_notification=disable_notification,
                        message_thread_id=self._config['telegram'].get('topic_id'),
                    )
                elif document and filename:
                    await self._app.bot.send_message(
                        self._config['telegram']['chat_id'],
                        document=document,
                        filename=filename,
                        parse_mode=parse_mode,
                        reply_markup=reply_markup,
                        disable_notification=disable_notification,
                        message_thread_id=self._config['telegram'].get('topic_id'),
                    )
                else:
                    await self._app.bot.send_message(
                        self._config['telegram']['chat_id'],
                        text=msg,
                        parse_mode=parse_mode,
                        reply_markup=reply_markup,
                        disable_notification=disable_notification,
                        message_thread_id=self._config['telegram'].get('topic_id'),
                    )
        except TelegramError as telegram_err:
            logger.warning('TelegramError: %s! Giving up on that message.', telegram_err.message)

    @authorized_only
    async def _changemarketdir(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /marketdir.
        Updates the bot's market_direction
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        if context.args and len(context.args) == 1:
            new_market_dir_arg = context.args[0]
            old_market_dir = self._rpc._get_market_direction()
            new_market_dir = None
            if new_market_dir_arg == 'long':
                new_market_dir = MarketDirection.LONG
            elif new_market_dir_arg == 'short':
                new_market_dir = MarketDirection.SHORT
            elif new_market_dir_arg == 'even':
                new_market_dir = MarketDirection.EVEN
            elif new_market_dir_arg == 'none':
                new_market_dir = MarketDirection.NONE

            if new_market_dir is not None:
                self._rpc._update_market_direction(new_market_dir)
                await self._send_msg(
                    'Successfully updated market direction'
                    f" from *{old_market_dir}* to *{new_market_dir}*."
                )
            else:
                raise RPCException(
                    'Invalid market direction provided. \n'
                    'Valid market directions: *long, short, even, none*'
                )
        elif context.args is not None and len(context.args) == 0:
            old_market_dir = self._rpc._get_market_direction()
            await self._send_msg(f"Currently set market direction: *{old_market_dir}*")
        else:
            raise RPCException(
                'Invalid usage of command /marketdir. \n'
                'Usage: */marketdir [short |  long | even | none]*'
            )

    async def _tg_info(self, update: Update, context: CallbackContext) -> None:
        """
        Intentionally unauthenticated Handler for /tg_info.
        Returns information about the current telegram chat - even if chat_id does not
        correspond to this chat.

        :param update: message update
        :return: None
        """
        if not update.message:
            return
        chat_id = update.message.chat_id
        topic_id = update.message.message_thread_id

        msg = f"""Freqtrade Bot Info:
        ```json
            {{
                "enabled": true,
                "token": "********",
                "chat_id": "{chat_id}",
                {f'"topic_id": "{topic_id}"' if topic_id else ''}
            }}
        ```
        """
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=msg,
                parse_mode=ParseMode.MARKDOWN_V2,
                message_thread_id=topic_id,
            )
        except TelegramError as telegram_err:
            logger.warning('TelegramError: %s! Giving up on that message.', telegram_err.message)

    def _page_kb(self, key: str, i: int, n: int) -> InlineKeyboardMarkup:
        prev_i = max(0, i - 1)
        next_i = min(n - 1, i + 1)
        rows = []
        if n > 1:
            rows.append([
                InlineKeyboardButton('⬅️ Prev', callback_data=f"ai_page:{key}:{prev_i}"),
                InlineKeyboardButton('Next ➡️', callback_data=f"ai_page:{key}:{next_i}"),
            ])
        rows.append([InlineKeyboardButton('📋 Copy as text', callback_data=f"ai_copy:{key}:{i}")])
        return InlineKeyboardMarkup(rows)

    async def _send_html_paginated(self, chat_id: int, html_text: str, key: str):
        # 1) 先清洗为 Telegram-safe
        safe_html = sanitize_telegram_html(html_text)
        # 2) 再标签感知分页
        pages = split_html_pages(safe_html)
        save_pages(key, pages)
        page, i, n = get_page(key, 0)
        # 3) 安全发送（失败自动 .txt 回退）
        await self._safe_send_html(
            chat_id=chat_id,
            html_text=page,
            reply_markup=self._page_kb(key, i, n),
            page_hint=f"page {i + 1}/{n}"
        )

    @authorized_only
    async def _ai_pagination_handler(self, update, context):
        q = update.callback_query
        data = q.data or ''
        try:
            # 先取 action
            action, rest = data.split(':', 1)          # e.g. "ta_page:1758100545:1"
            # 再从右侧取 idx
            key, idx_s = rest.rsplit(':', 1)           # key = "1758100545", idx_s = "1" ；若 key 中还有冒号也安全
            idx = int(idx_s)
        except Exception:
            await q.answer('Invalid action')
            return

        if action == 'ai_page':
            page, i, n = get_page(key, idx)
            await self._safe_edit_html(
                q=q,
                html_text=page,
                reply_markup=self._page_kb(key, i, n),
                page_hint=f"page {i + 1}/{n}",
            )
            try:
                await q.answer(f"Page {i + 1}/{n}")
            except Exception:
                pass

        elif action == 'ai_copy':
            page, i, n = get_page(key, idx)
            plain = re.sub(r'<[^>]+>', '', page or '')
            if len(plain) > 3900:
                plain = plain[:3900]
            try:
                await q.answer('Copied as text', show_alert=False)
            except Exception:
                pass
            await self._bot.send_message(chat_id=q.message.chat_id, text=plain)

    async def _send_txt_document(self, chat_id: int, content: str, filename: str, caption: str | None = None):
        bio = BytesIO(content.encode('utf-8'))
        bio.name = filename
        await self._app.bot.send_document(
            chat_id=chat_id,
            document=bio,
            caption=caption or 'Rendered as .txt (HTML parsing error). Please forward this file back for debugging.'
        )

    async def _safe_send_html(self, chat_id: int, html_text: str, reply_markup=None, page_hint: str | None = None):
        try:
            pretty = _beautify_inequalities(html_text or '')
            await self._app.bot.send_message(
                chat_id=chat_id,
                text=pretty,
                parse_mode='HTML',
                disable_web_page_preview=True,
                reply_markup=reply_markup,
            )
        except Exception:
            plain = _to_clean_plain(html_text or '')
            await self._app.bot.send_message(
                chat_id=chat_id,
                text=plain,
                reply_markup=reply_markup,
            )

    async def _safe_edit_html(self, q, html_text: str, reply_markup=None, page_hint: str | None = None):
        try:
            pretty = _beautify_inequalities(html_text or '')
            await q.edit_message_text(
                text=pretty,
                parse_mode='HTML',
                disable_web_page_preview=True,
                reply_markup=reply_markup,
            )
        except Exception:
            plain = _to_clean_plain(html_text or '')
            await q.edit_message_text(
                text=plain,
                reply_markup=reply_markup,
            )
            try:
                await q.answer('Rendered as plain text due to HTML error', show_alert=False)
            except Exception:
                pass

    @authorized_only
    async def _cmd_two_phase(self, update, context):
        msg = update.message or update.effective_message
        raw = (msg.text or '').strip()
        parts = raw.split(' ', 1)
        if len(parts) < 2 or not parts[1].strip():
            await msg.reply_text('Usage: /ai <prompt>\nExamples:\n'
                                '/ai [0] BTC short-term trend plan\n'
                                '/ai 1:: ETH reversal setup\n'
                                '/ai ONDO/USDT short-term plan ::2\n'
                                '0: normal, 1: reversal, 2: trend 3: market 4: postion dac\n')
            return

        # 原始自由 prompt（包含可能的模式标记）
        user_input = parts[1].strip()

        # 只在开头或结尾解析模式；默认给 0
        mode, cleaned_prompt = extract_mode_and_prompt(user_input, default_mode=0)

        await msg.reply_text(
            f"Got it. Analyzing (mode={mode}):\n{cleaned_prompt}\n\nI'll send the result here."
        )

        async def _worker(chat_id: int, prompt: str, mode: int):
            try:
                html_result = await run_two_phase(prompt, mode)  # 返回 HTML 片段
                key = f"{chat_id}:{int(time.time())}"
                await self._send_html_paginated(chat_id, html_result, key)
            except Exception as e:
                await self._app.bot.send_message(chat_id=chat_id, text=f"Analysis failed: {e}")

        asyncio.create_task(_worker(msg.chat_id, cleaned_prompt, mode))

    @authorized_only
    async def _chart(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /chart <pair> [timeframe].
        Generate and send back a chart for the pair using CCXT directly

        :param update: message update
        :param context: callback context
        :return: None
        """
        query = update.callback_query

        # Initialize context.args if needed
        if not hasattr(context, 'args'):
            context.args = []

        # Check if we have parameters
        if query and ':' in query.data:
            try:
                # Parse parameters
                base, params_str = query.data.split(':', 1)
                params = {}

                # Simple parsing: param1=value1,param2=value2
                for item in params_str.split(','):
                    if '=' in item:
                        key, value = item.split('=', 1)
                        params[key] = value

                # Extract parameters
                if 'pair' in params:
                    # Make sure context.args is a list
                    context.args = [params['pair']]
                    # Add timeframe if available
                    if 'timeframe' in params:
                        context.args.append(params['timeframe'])
            except Exception as e:
                logger.error(f"Error parsing callback parameters: {e}")

        if not context.args or len(context.args) == 0:
            raise RPCException('Usage: /chart <pair> [timeframe]')

        pair = context.args[0].upper()  # 确保大写一致性
        if not pair.endswith('/USDT:USDT'):
            pair += '/USDT:USDT'
        timeframe = context.args[1] if len(context.args) > 1 else self._config['timeframe']

        try:
            # 获取当前使用的交易所名称
            exchange_name = self._rpc._freqtrade.exchange.name.lower()
            logger.info(
                f"Fetching chart data for {pair} ({timeframe}) using CCXT with {exchange_name}"
            )

            # 确定交易对格式
            # 某些交易所使用/分隔符，有些不使用
            if '/' not in pair and exchange_name not in ['ftx', 'kucoin']:
                # 为Binance, OKX等交易所，尝试添加/
                base_quote_pairs = []
                # 常见分隔位置，例如 BTC/USDT, ETH/USDT
                for pos in [3, 4]:
                    if len(pair) > pos:
                        base_quote_pairs.append(f"{pair[:pos]}/{pair[pos:]}")

                # 尝试其他可能的格式
                for symbol_format in [pair, f"{pair[:-4]}/{pair[-4:]}", f"{pair[:-3]}/{pair[-3:]}"]:
                    if symbol_format not in base_quote_pairs and '/' in symbol_format:
                        base_quote_pairs.append(symbol_format)
            else:
                # 已经有/或使用不需要/的交易所
                base_quote_pairs = [pair]

            # 创建CCXT交易所实例
            exchange_class = getattr(ccxt, exchange_name)

            # 获取交易所API凭证
            exchange_config = self._rpc._freqtrade.config.get('exchange', {})
            ccxt_config = {
                'apiKey': exchange_config.get('key', ''),
                'secret': exchange_config.get('secret', ''),
                'password': exchange_config.get('password', ''),
                'enableRateLimit': True,
            }

            # 对于某些交易所的特殊设置
            if 'ccxt_config' in exchange_config:
                ccxt_config.update(exchange_config['ccxt_config'])

            # 只保留非空值
            ccxt_config = {k: v for k, v in ccxt_config.items() if v}

            # 实例化交易所
            exchange = exchange_class(ccxt_config)

            # 设置市场类型 (如果需要)
            if hasattr(exchange, 'options'):
                if exchange_name == 'binance':
                    exchange.options['defaultType'] = 'spot'
                elif exchange_name == 'okx':
                    exchange.options['defaultType'] = 'spot'

            # 根据时间框架计算合适的数据范围
            try:
                # 解析时间框架并设置合适的回溯期
                if timeframe.endswith('m'):
                    minutes = int(timeframe[:-1])
                    if minutes <= 5:
                        lookback_days = 0.5  # 12小时
                    elif minutes <= 15:
                        lookback_days = 1  # 1天
                    elif minutes <= 30:
                        lookback_days = 2  # 2天
                    else:
                        lookback_days = 3  # 3天
                elif timeframe.endswith('h'):
                    hours = int(timeframe[:-1])
                    if hours <= 1:
                        lookback_days = 3  # 3天
                    elif hours <= 4:
                        lookback_days = 7  # 7天
                    elif hours <= 8:
                        lookback_days = 14  # 14天
                    else:
                        lookback_days = 21  # 21天
                elif timeframe.endswith('d'):
                    lookback_days = 90  # 90天
                elif timeframe.endswith('w'):
                    lookback_days = 180  # 半年
                else:
                    lookback_days = 7  # 默认一周
            except (ValueError, AttributeError):
                lookback_days = 7

            logger.info(f"Using {lookback_days} days of data for {timeframe} timeframe")
            since_ms = int((datetime.now() - timedelta(days=lookback_days)).timestamp() * 1000)

            # 尝试所有可能的交易对格式
            ohlcv_data = None
            used_pair = None
            error_messages = []

            for try_pair in base_quote_pairs:
                try:
                    logger.info(f"Trying to fetch OHLCV for {try_pair}")

                    # 加载交易所市场，以确保支持的交易对
                    exchange.load_markets()

                    # 获取OHLCV数据
                    ohlcv_data = exchange.fetch_ohlcv(
                        symbol=try_pair, timeframe=timeframe, since=since_ms, limit=500  # 请求足够的数据点
                    )

                    if ohlcv_data and len(ohlcv_data) > 0:
                        used_pair = try_pair
                        logger.info(f"Successfully fetched data for {try_pair}")
                        break
                    else:
                        error_messages.append(f"No data returned for {try_pair}")

                except Exception as e:
                    error_msg = f"Error fetching {try_pair}: {str(e)}"
                    logger.info(error_msg)
                    error_messages.append(error_msg)
                    continue

            # 检查是否成功获取数据
            if ohlcv_data is None or len(ohlcv_data) == 0:
                error_summary = '\n'.join(error_messages)
                await self._send_msg(
                    f"无法获取数据。尝试了以下交易对: {', '.join(base_quote_pairs)}\n错误信息:\n{error_summary}"
                )
                return

            # 将数据转换为DataFrame
            candles = pd.DataFrame(
                ohlcv_data, columns=['date', 'open', 'high', 'low', 'close', 'volume']
            )

            # 时间戳转换为日期时间
            candles['date'] = pd.to_datetime(candles['date'], unit='ms')

            # 确保索引是日期时间类型
            candles = candles.set_index(pd.DatetimeIndex(candles['date']))

            # 计算技术指标和市场统计
            # 1. 计算波动率 (ATR - 平均真实范围)
            candles['tr1'] = abs(candles['high'] - candles['low'])
            candles['tr2'] = abs(candles['high'] - candles['close'].shift())
            candles['tr3'] = abs(candles['low'] - candles['close'].shift())
            candles['tr'] = candles[['tr1', 'tr2', 'tr3']].max(axis=1)
            candles['atr'] = candles['tr'].rolling(14).mean()

            # 2. 计算RSI
            delta = candles['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            candles['rsi'] = 100 - (100 / (1 + rs))

            # 3. 计算移动平均线
            candles['sma20'] = candles['close'].rolling(window=20).mean()
            candles['sma50'] = candles['close'].rolling(window=50).mean()
            candles['sma200'] = candles['close'].rolling(window=200).mean()

            # 4. 计算MACD
            candles['ema12'] = candles['close'].ewm(span=12, adjust=False).mean()
            candles['ema26'] = candles['close'].ewm(span=26, adjust=False).mean()
            candles['macd'] = candles['ema12'] - candles['ema26']
            candles['signal'] = candles['macd'].ewm(span=9, adjust=False).mean()
            candles['histogram'] = candles['macd'] - candles['signal']

            # 5. 计算布林带
            candles['bb_middle'] = candles['close'].rolling(window=20).mean()
            candles['bb_std'] = candles['close'].rolling(window=20).std()
            candles['bb_upper'] = candles['bb_middle'] + (candles['bb_std'] * 2)
            candles['bb_lower'] = candles['bb_middle'] - (candles['bb_std'] * 2)

            # 获取基本市场统计数据
            latest_candle = candles.iloc[-1]
            prev_candle = candles.iloc[-2]

            # 计算重要价格水平
            last_close = latest_candle['close']
            last_open = latest_candle['open']
            day_high = latest_candle['high']
            day_low = latest_candle['low']
            sma20 = latest_candle['sma20']
            sma50 = latest_candle['sma50']

            # 计算24小时价格变化
            price_change_24h = last_close - candles['close'].iloc[-24 if timeframe == '1h' else -1]
            price_change_percent_24h = (
                price_change_24h / candles['close'].iloc[-24 if timeframe == '1h' else -1]
            ) * 100

            # 计算波动性指标
            atr = latest_candle['atr']
            atr_percent = (atr / last_close) * 100

            # 计算交易量变化
            volume = latest_candle['volume']
            avg_volume = candles['volume'].rolling(20).mean().iloc[-1]
            volume_change = ((volume / avg_volume) - 1) * 100

            # 计算关键技术指标
            rsi = latest_candle['rsi']
            # macd = latest_candle['macd']
            # signal = latest_candle['signal']
            histogram = latest_candle['histogram']

            # 准备市场状态描述
            market_status = '看涨' if last_close > last_open else '看跌'
            price_position = ''
            if last_close > sma20 and last_close > sma50:
                price_position = '多头趋势'
            elif last_close < sma20 and last_close < sma50:
                price_position = '空头趋势'
            else:
                price_position = '震荡区间'

            # RSI状态
            rsi_status = ''
            if rsi > 70:
                rsi_status = '超买'
            elif rsi < 30:
                rsi_status = '超卖'
            else:
                rsi_status = '中性'

            # 布林带位置
            bb_upper = latest_candle['bb_upper']
            bb_lower = latest_candle['bb_lower']
            bb_position = ''
            if last_close > bb_upper:
                bb_position = '突破上轨(可能超买)'
            elif last_close < bb_lower:
                bb_position = '突破下轨(可能超卖)'
            else:
                width = (bb_upper - bb_lower) / latest_candle['bb_middle'] * 100
                if width < 10:
                    bb_position = '带宽收窄(可能突破)'
                else:
                    bb_position = '正常波动区间'

            # MACD信号
            macd_signal = ''
            if histogram > 0 and histogram > prev_candle['histogram']:
                macd_signal = '看涨(加速)'
            elif histogram > 0 and histogram < prev_candle['histogram']:
                macd_signal = '看涨(减速)'
            elif histogram < 0 and histogram < prev_candle['histogram']:
                macd_signal = '看跌(加速)'
            elif histogram < 0 and histogram > prev_candle['histogram']:
                macd_signal = '看跌(减速)'
            else:
                macd_signal = '横盘整理'

            # 根据时间框架设置显示的蜡烛数量
            try:
                if timeframe.endswith('m'):
                    minutes = int(timeframe[:-1])
                    if minutes <= 5:
                        target_candles = 50  # 更少的蜡烛，更清晰
                    elif minutes <= 15:
                        target_candles = 60
                    elif minutes <= 30:
                        target_candles = 70
                    else:
                        target_candles = 75
                elif timeframe.endswith('h'):
                    hours = int(timeframe[:-1])
                    if hours <= 1:
                        target_candles = 70
                    elif hours <= 4:
                        target_candles = 80
                    elif hours <= 8:
                        target_candles = 90
                    else:
                        target_candles = 100
                elif timeframe.endswith('d'):
                    target_candles = 90
                elif timeframe.endswith('w'):
                    target_candles = 80
                else:
                    target_candles = 80
            except (ValueError, AttributeError):
                target_candles = 80

            # 限制要显示的蜡烛数量
            max_candles = min(target_candles, len(candles))

            # 配置图表样式
            mc = mpf.make_marketcolors(
                up='green',
                down='red',
                edge='black',
                wick={'up': 'green', 'down': 'red'},
                volume='blue',
            )

            # 添加自定义样式
            s = mpf.make_mpf_style(
                marketcolors=mc, gridstyle='--', figcolor='white', facecolor='white'
            )

            # 创建BytesIO对象保存图表
            buf = BytesIO()

            # 设置适合的日期格式
            if timeframe.endswith('m'):
                date_format = '%m-%d %H:%M'
            elif timeframe.endswith('h') and int(timeframe[:-1]) < 24:
                date_format = '%m-%d %H:%M'
            else:
                date_format = '%Y-%m-%d'

            # 添加指标到图表上
            apds = [
                mpf.make_addplot(candles['sma20'].tail(max_candles), color='blue', width=0.7),
                mpf.make_addplot(candles['sma50'].tail(max_candles), color='orange', width=0.7),
                mpf.make_addplot(
                    candles['bb_upper'].tail(max_candles), color='gray', width=0.5, linestyle='--'
                ),
                mpf.make_addplot(
                    candles['bb_lower'].tail(max_candles), color='gray', width=0.5, linestyle='--'
                ),
            ]

            # 绘制K线图
            fig, axes = mpf.plot(
                candles.tail(max_candles),
                type='candle',
                volume=True,
                title=f'\n{used_pair} {timeframe}',
                style=s,
                returnfig=True,
                datetime_format=date_format,
                figsize=(12, 8),
                addplot=apds,
            )

            # 保存图表到BytesIO对象
            fig.savefig(buf, format='png', dpi=150)
            buf.seek(0)

            # 生成市场分析文本
            # 处理浮点数显示
            def fmt_num(num):
                if abs(num) < 0.001:
                    return f"{num:.8f}"
                elif abs(num) < 1:
                    return f"{num:.4f}"
                elif abs(num) < 10:
                    return f"{num:.2f}"
                else:
                    return f"{num:.1f}"

            # 生成市场分析文本
            currency = used_pair.split('/')[1] if '/' in used_pair else 'USD'
            analysis_text = (
                f"{used_pair} ({timeframe}) - 最近 {max_candles} 根K线\n"
                f"——————市场概况——————\n"
                f"价格: {fmt_num(last_close)} {currency} ({'+' if price_change_percent_24h > 0 else ''}{price_change_percent_24h:.2f}%)\n"
                f"成交量: {fmt_num(volume)} ({'+' if volume_change > 0 else ''}{volume_change:.1f}%)\n"
                f"日内: 高 {fmt_num(day_high)} / 低 {fmt_num(day_low)}\n"
                f"波动性: ATR {fmt_num(atr)} ({atr_percent:.2f}%)\n"
                f"——————技术指标——————\n"
                f"RSI(14): {rsi:.1f} ({rsi_status})\n"
                f"MACD: {macd_signal}\n"
                f"趋势: {price_position}\n"
                f"布林带: {bb_position}\n"
                f"当前状态: {market_status}\n"
                f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )

            # # 发送图表图像
            # await self._app.bot.send_photo(
            #     chat_id=update.effective_chat.id,
            #     photo=buf,
            #     caption=analysis_text,
            #     message_thread_id=self._config['telegram'].get('topic_id'),
            # )
            # Example for chart with parameters
            def create_callback_data(base_pattern, **params):
                """Create compact callback data"""
                if not params:
                    return base_pattern

                # Create a compact representation
                # Format: base_pattern:param1=value1,param2=value2
                params_str = ','.join(f"{k}={v}" for k, v in params.items())

                # Make sure we're under the 64 byte limit
                if len(f"{base_pattern}:{params_str}") > 60:  # Leave some margin
                    # If too long, just include the most important params
                    if 'pair' in params:
                        return f"{base_pattern}:pair={params['pair']}"
                    # Or use the first param only
                    first_key = list(params.keys())[0]
                    return f"{base_pattern}:{first_key}={params[first_key]}"

                return f"{base_pattern}:{params_str}"

            # Usage:
            callback_data = create_callback_data('update_chart', pair=pair, timeframe=timeframe)

            await self._send_msg(
                analysis_text,
                photo=buf,
                reload_able=True,
                callback_path=callback_data,
                query=update.callback_query,
            )

        except Exception as e:
            logger.exception('生成图表时出错: %s', str(e))
            await self._send_msg(f"生成图表时出错: {str(e)}")

    @authorized_only
    async def _prompt(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /analysis <pair> [timeframe].
        Uses LLM to analyze the given trading pair
        :param update: message update
        :return: None
        """

        query = update.callback_query

        # Initialize context.args if needed
        if not hasattr(context, 'args'):
            context.args = []

        # Check if we have parameters
        if query and ':' in query.data:
            try:
                # Parse parameters
                base, params_str = query.data.split(':', 1)
                params = {}

                # Simple parsing: param1=value1,param2=value2
                for item in params_str.split(','):
                    if '=' in item:
                        key, value = item.split('=', 1)
                        params[key] = value

                # Extract parameters
                if 'pair' in params:
                    # Make sure context.args is a list
                    context.args = [params['pair']]
            except Exception as e:
                logger.error(f"Error parsing callback parameters: {e}")

        if not context.args or len(context.args) == 0:
            raise RPCException('Usage: /prompt <pair>')

        pair = context.args[0].upper()
        if not pair.endswith('/USDT:USDT'):
            pair += '/USDT:USDT'

        try:
            # Perform new analysis
            exchange_config = self._rpc._freqtrade.config.get('exchange', {})

            # Initialize analyzer
            analyst = CryptoTechnicalAnalyst(
                api_key=exchange_config.get('key', ''),
                api_secret=exchange_config.get('secret', ''),
            )
            prompt = analyst.gen_llm_prompt(pair)

            if len(prompt) > 3000:
                # 创建临时文本文件
                import tempfile
                import os

                with tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='w') as tmp_file:
                    tmp_file.write(prompt)
                    tmp_file_path = tmp_file.name

                # 发送文件
                with open(tmp_file_path, 'rb') as document:
                    def create_callback_data(base_pattern, **params):
                        """Create compact callback data"""
                        if not params:
                            return base_pattern

                        # Create a compact representation
                        # Format: base_pattern:param1=value1,param2=value2
                        params_str = ','.join(f"{k}={v}" for k, v in params.items())

                        # Make sure we're under the 64 byte limit
                        if len(f"{base_pattern}:{params_str}") > 60:  # Leave some margin
                            # If too long, just include the most important params
                            if 'pair' in params:
                                return f"{base_pattern}:pair={params['pair']}"
                            # Or use the first param only
                            first_key = list(params.keys())[0]
                            return f"{base_pattern}:{first_key}={params[first_key]}"

                        return f"{base_pattern}:{params_str}"

                    # Usage:
                    callback_data = create_callback_data('update_prompt', pair=pair)
                    await self._send_msg(
                        f"Generated prompt for {pair}",
                        document=document,
                        filename=f"{pair.split('/')[0]}_prompt.txt",
                        reload_able=True,
                        callback_path=callback_data,
                        query=update.callback_query,
                    )

                # 删除临时文件
                os.unlink(tmp_file_path)
            else:
                # 如果内容不超过限制，直接发送文本消息
                await self._send_msg(f"```{prompt}```", reload_able=True, callback_path=callback_data, query=update.callback_query)

        except Exception as e:
            logger.exception('Error during prompt gen: %s', str(e))
            await self._send_msg(f"❌ Error during prompt gen: {str(e)}")

    @authorized_only
    async def _prompt_json(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /analysis <pair> [timeframe].
        Uses LLM to analyze the given trading pair
        :param update: message update
        :return: None
        """
        if not context.args or len(context.args) == 0:
            raise RPCException('Usage: /promptjson <text>')

        message_text = update.message.text
        if message_text.startswith('/promptjson'):
            message_text = message_text[len('/promptjson') :].strip()

        try:

            # Perform new analysis
            extractor = TradingSignalExtractor()
            prompt = extractor.gen_json_prompt(message_text)

            # Get deep analysis
            await self._send_msg(f"```{prompt}```")

        except Exception as e:
            logger.exception('Error during prompt gen: %s', str(e))
            await self._send_msg(f"❌ Error during prompt gen: {str(e)}")

    @authorized_only
    async def _analysis(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /analysis <pair> [timeframe].
        Uses LLM to analyze the given trading pair
        :param update: message update
        :return: None
        """
        if not context.args or len(context.args) == 0:
            raise RPCException('Usage: /analysis <pair> [timeframe]')

        pair = context.args[0].upper()
        if not pair.endswith('/USDT:USDT'):
            pair += '/USDT:USDT'

        try:
            # Show analysis in progress message
            await self._send_msg(f"📊 Analyzing {pair}... Please wait.")

            # Connect to database
            conn = connect_to_db()
            if not conn:
                logger.error('Failed to connect to database.')
                await self._send_msg(
                    '❌ Database connection failed. Proceeding with direct analysis.'
                )
                conn = None

            # Check if we already have today's analysis
            existing_analysis = None
            if conn:
                existing_analysis = get_todays_analysis(conn, pair)
            # Perform new analysis
            exchange_config = self._rpc._freqtrade.config.get('exchange', {})

            # Initialize analyzer
            analyst = CryptoTechnicalAnalyst(
                api_key=exchange_config.get('key', ''),
                api_secret=exchange_config.get('secret', ''),
            )
            if existing_analysis:
                # Use existing analysis from today
                logger.info(f"Using existing analysis from today for {pair}")
                table_output, analysis_text, raw_json_db, processed_json_db = existing_analysis

                # Convert from database JSONB to Python objects if needed
                if isinstance(raw_json_db, str):
                    raw_json = json.loads(raw_json_db)
                else:
                    raw_json = raw_json_db

                if isinstance(processed_json_db, str):
                    processed_json = json.loads(processed_json_db)
                else:
                    processed_json = processed_json_db

                # Send the cached results
                await self._send_msg(table_output, parse_mode=ParseMode.HTML)

                await self._send_msg(analysis_text, parse_mode=ParseMode.MARKDOWN)

                # Send the processed JSON data
                await self._send_msg(f"📋 Processed JSON data for {pair}:")
                await self._send_msg(f"```json\n{json.dumps(processed_json, indent=2)}\n```")
            else:
                # Analyze trading pair
                analyst.analyze_crypto(pair)

                # Generate formatted table
                table_output = analyst.generate_formatted_table(pair)

                await self._send_msg(table_output, parse_mode=ParseMode.HTML)

                # Get deep analysis
                await self._send_msg(f"🤖 Generating in-depth analysis for {pair}...")
                analysis_chunks, analysis_text = analyst.get_formatted_llm_analysis(pair)

                # Extract trading signals if needed
                extractor = TradingSignalExtractor()
                raw_json, processed_json = extractor.extract_to_json_string(
                    analysis_text, consolidate=True
                )

                # Store results in database if connection exists
                if conn:
                    insert_analysis_result(
                        conn, pair, table_output, analysis_text, raw_json, processed_json
                    )
                    logger.info(f"Analysis results stored in database for {pair}")

                await self._send_msg(analysis_text, parse_mode=ParseMode.MARKDOWN)

                # Send the processed JSON data
                await self._send_msg(f"📋 Processed JSON data for {pair}:")
                await self._send_msg(f"```json\n{json.dumps(processed_json, indent=2)}\n```")

            # Close database connection if it exists
            if conn:
                conn.close()

        except Exception as e:
            logger.exception('Error during analysis: %s', str(e))
            await self._send_msg(f"❌ Error during analysis: {str(e)}")

    @authorized_only
    async def _add_pair(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /addpair <pair>.
        添加交易对到白名单
        """
        if not context.args or len(context.args) == 0:
            raise RPCException('使用方法: /addpair <币种/USDT>')

        pairs = context.args[0].upper()

        # 获取当前白名单
        current_whitelist = self._rpc._rpc_whitelist()['whitelist']

        pairs_to_add = []
        for pair in pairs.split(','):
            if not pair.endswith('/USDT:USDT'):
                pair += '/USDT:USDT'

            if pair in current_whitelist:
                await self._send_msg(f'交易对 {pair} 已在当前白名单中')
            else:
                pairs_to_add.append(pair)

        if pairs_to_add:
            with open('/freqtrade/config_production.json', 'r') as f:
                config = json.load(f)

            config['exchange']['pair_whitelist'].extend(pairs_to_add)

            with open('/freqtrade/config_production.json', 'w') as f:
                json.dump(config, f, indent=4)

            self._rpc._rpc_reload_config()
            await self._send_msg(f'交易对 {', '.join(pairs_to_add)} 已加入白名单')

        # 获取最新分析作为参考
        try:
            conn = connect_to_db()
            if conn:
                analysis_result = get_todays_analysis(conn, pair)
                conn.close()

                if analysis_result:
                    _, _, _, processed_json = analysis_result

                    if isinstance(processed_json, str):
                        processed_json = json.loads(processed_json)

                    # 发送JSON供用户参考和修改
                    await self._send_msg(f"📋 {pair} 的最新策略参数参考：")
                    await self._send_msg(f"```json\n{json.dumps(processed_json, indent=2)}\n```")
                    await self._send_msg('您可以复制上面的JSON，根据需要进行修改，然后通过 /setpairstrategy 命令设置。')
        except Exception as e:
            logger.error(f"获取参考数据时出错: {str(e)}")

        await self._send_msg(f"👉 请使用 /setpairstrategy 命令，并发送 {pair} 的策略 JSON。")

    @authorized_only
    async def _del_pair(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /delpair <pair>.
        从白名单移除交易对
        """
        if not context.args or len(context.args) == 0:
            raise RPCException('使用方法: /delpair 币种/USDT')

        pairs = context.args[0].upper()

        pairs_to_del = []

        # 获取当前白名单
        current_whitelist = self._rpc._rpc_whitelist()['whitelist']

        for pair in pairs.split(','):
            if not pair.endswith('/USDT:USDT'):
                pair += '/USDT:USDT'

            if pair not in current_whitelist:
                await self._send_msg(f'交易对 {pair} 不在当前白名单中')
            else:
                pairs_to_del.append(pair)

        if pairs_to_del:

            with open('/freqtrade/config_production.json', 'r') as f:
                config = json.load(f)

            for pair in pairs_to_del:
                config['exchange']['pair_whitelist'].remove(pair)

            with open('/freqtrade/config_production.json', 'w') as f:
                json.dump(config, f, indent=4)

            self._rpc._rpc_reload_config()
            await self._send_msg(f'交易对 {', '.join(pair)} 已从白名单移除')

    @authorized_only
    async def _set_pair_strategy_auto(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /setpairstrategyauto.
        设置交易对的策略参数 - 从消息文本中获取JSON而不是命令参数
        """
        if not context.args or len(context.args) == 0:
            raise RPCException('使用方法: /setpairstrategyauto 币种1/USDT,long;币种2/USDT,short')

        # 获取消息文本，删除命令本身
        message_text = update.message.text
        if message_text.startswith('/setpairstrategyauto'):
            message_text = message_text[len('/setpairstrategyauto') :].strip()

        # 检查是否有JSON内容
        if not message_text:
            await self._send_msg('❌ 未提供策略交易对。请在命令后发送交易对数据。')
            return

        # 尝试解析JSON内容
        try:
            with open('/freqtrade/user_data/strategy_state_production.json', 'r') as f:
                strategy_state = json.load(f)

            for pair_str in message_text.split(';'):
                parts = pair_str.split(',')
                pair = parts[0].upper()
                if not pair.endswith('/USDT:USDT'):
                    pair += '/USDT:USDT'

                new_direction = parts[1].lower()  # 获取新的 direction

                config = self._rpc._freqtrade.strategy.calculate_coin_points(pair, new_direction)
                if not config:
                    self._send_msg(f'❌ 无法为 {pair} 计算策略参数。')
                    continue

                if pair in strategy_state['coin_monitoring']:
                    # 如果 pair 存在，过滤掉与新 direction 相同的策略
                    strategy_state['coin_monitoring'][pair] = [
                        strategy for strategy in strategy_state['coin_monitoring'][pair]
                        if strategy['direction'] != new_direction
                    ]
                    # 如果列表变空，则删除该 pair 的键（可选）
                    if not strategy_state['coin_monitoring'][pair]:
                        del strategy_state['coin_monitoring'][pair]

                # 如果 pair 已删除或原本不存在，直接 append 新策略
                if pair not in strategy_state['coin_monitoring']:
                    strategy_state['coin_monitoring'][pair] = []

                strategy_state['coin_monitoring'][pair].append(
                    {
                        'direction': new_direction,
                        'auto': True,
                        **config
                    }
                )

            self._rpc._freqtrade.strategy.coin_monitoring = strategy_state['coin_monitoring']

            # strategy_state['coin_monitoring'][pair] = strategy_json

            with open('/freqtrade/user_data/strategy_state_production.json', 'w') as f:
                json.dump(strategy_state, f, indent=4)

            await self._send_msg(f"✅ 成功添加 {pair} 到交易对白名单，并设置了相应的策略参数。")

        except json.JSONDecodeError as e:
            await self._send_msg(f"❌ JSON格式无效: {str(e)}。请提供有效的JSON字符串。")
        except Exception as e:
            logger.exception('设置交易对策略时出错: %s', str(e))
            await self._send_msg(f"❌ 设置交易对策略时出错: {str(e)}")

    @authorized_only
    async def _set_pair_strategy(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /setpairstrategy.
        设置交易对的策略参数 - 从消息文本中获取JSON而不是命令参数
        """
        if not context.args or len(context.args) < 2:
            raise RPCException('使用方法: /setpairstrategy 币种/USDT 交易策略')

        pair = context.args[0].upper()
        if not pair.endswith('/USDT:USDT'):
            pair += '/USDT:USDT'

        # 获取消息文本，删除命令本身
        message_text = update.message.text
        if message_text.startswith('/setpairstrategy'):
            message_text = message_text[len(f'/setpairstrategy {context.args[0]}') :].strip()

        # 检查是否有JSON内容
        if not message_text:
            await self._send_msg('❌ 未提供策略JSON。请在命令后发送JSON数据。')
            return

        # 尝试解析JSON内容
        try:
            # 尝试提取可能被包装在代码块中的JSON
            if message_text.startswith('```json') and message_text.endswith('```'):
                message_text = message_text[7:-3].strip()
            elif message_text.startswith('```') and message_text.endswith('```'):
                message_text = message_text[3:-3].strip()

            try:
                strategy_json = json.loads(message_text)
            except:
                strategy_json = []
                for i in message_text.split(';'):
                    parts = i.split(',')  # 先将字符串按逗号分割成列表
                    direction = parts[0]  # 第一个元素是 direction
                    entry_point = float(parts[1])  # 第二个元素是 entry_point
                    stop_loss = float(parts[-1])  # 最后一个元素是 stop_loss
                    # 中间的部分（除了 direction, entry_point 和 stop_loss）作为 exit_points
                    exit_points = [float(x) for x in parts[2:-1]]  # 动态提取退出点

                    strategy_json.append(
                        {
                            'direction': direction,
                            'entry_points': [entry_point],
                            'exit_points': exit_points,  # 动态数量的退出点
                            'stop_loss': stop_loss,
                        }
                    )

            with open('/freqtrade/user_data/strategy_state_production.json', 'r') as f:
                strategy_state = json.load(f)

            strategy_state['coin_monitoring'][pair] = strategy_json

            self._rpc._freqtrade.strategy.coin_monitoring = strategy_state['coin_monitoring']
            with open('/freqtrade/user_data/strategy_state_production.json', 'w') as f:
                json.dump(strategy_state, f, indent=4)

            await self._send_msg(f"✅ 成功添加 {pair} 到交易对白名单，并设置了相应的策略参数。")

        except json.JSONDecodeError as e:
            await self._send_msg(f"❌ JSON格式无效: {str(e)}。请提供有效的JSON字符串。")
        except Exception as e:
            logger.exception('设置交易对策略时出错: %s', str(e))
            await self._send_msg(f"❌ 设置交易对策略时出错: {str(e)}")

    @authorized_only
    async def _del_pair_strategy(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /delpairstrategy <pair>.
        删除交易对的策略参数
        """
        if not context.args or len(context.args) < 1:
            raise RPCException('使用方法: /delpairstrategy <币种/USDT>')

        pair = context.args[0].upper()
        if not pair.endswith('/USDT:USDT'):
            pair += '/USDT:USDT'

        try:
            strategy_file = '/freqtrade/user_data/strategy_state_production.json'

            # 读取当前策略状态
            try:
                with open(strategy_file, 'r') as f:
                    strategy_state = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                await self._send_msg('❌ 未找到策略状态文件或文件格式无效。')
                return

            # 检查交易对是否存在
            if (
                'coin_monitoring' not in strategy_state
                or pair not in strategy_state['coin_monitoring']
            ):
                await self._send_msg(f"❌ 未找到 {pair} 的策略参数。")
                return

            # 删除策略参数
            removed_strategy = strategy_state['coin_monitoring'].pop(pair)  # noqa

            # 保存更新后的策略状态
            with open(strategy_file, 'w') as f:
                json.dump(strategy_state, f, indent=4)

            self._rpc._freqtrade.strategy.coin_monitoring.pop(pair)

            await self._send_msg(f"✅ 成功删除 {pair} 的策略参数。")

        except Exception as e:
            logger.exception('删除策略参数时出错: %s', str(e))
            await self._send_msg(f"❌ 删除策略参数时出错: {str(e)}")

    @authorized_only
    async def _show_pair_strategy(self, update: Update, context: CallbackContext) -> None:
        """
        Handler for /showpairstrategy [pair or all].
        显示交易对的策略参数和自动量化策略配置
        必须传入参数, 要么是 'all' 或者特定交易对
        """
        try:
            # 检查是否有参数
            if not context.args or len(context.args) == 0:
                usage_msg = (
                    '⚠️ 使用方法:\n/showpairstrategy all - 显示所有交易对策略\n/showpairstrategy <币种> - 显示特定交易对策略'
                )
                await self._send_msg(usage_msg)
                return

            # 检查固定点位策略
            has_coin_monitoring = len(self._rpc._freqtrade.strategy.coin_monitoring) > 0

            # 检查自动量化策略
            has_pair_strategy_mode = len(self._rpc._freqtrade.strategy.pair_strategy_mode) > 0

            if not has_coin_monitoring and not has_pair_strategy_mode:
                await self._send_msg('⚠️ 当前没有任何交易对的策略参数。')
                return

            # 处理特定参数
            param = context.args[0].upper()

            coin_monitoring = self._rpc._freqtrade.strategy.coin_monitoring
            pair_strategy_mode = self._rpc._freqtrade.strategy.pair_strategy_mode

            # 处理 'all' 参数 - 显示所有交易对
            if param == 'ALL':
                # 显示所有交易对的策略参数摘要
                summary = '📋 当前配置的交易对策略：\n\n'

                # 合并两种策略的所有交易对
                all_pairs = set()
                if has_coin_monitoring:
                    all_pairs.update(coin_monitoring.keys())
                if has_pair_strategy_mode:
                    all_pairs.update(pair_strategy_mode.keys())

                # 按字母顺序排序
                sorted_pairs = sorted(all_pairs)

                # 创建摘要
                for i, pair in enumerate(sorted_pairs, 1):
                    summary_line = f"{i}. {pair} - "

                    strategy_types = []

                    # 检查是否有固定点位策略
                    if has_coin_monitoring and pair in coin_monitoring:
                        strategy_types.append('固定点位')

                    # 检查是否有自动量化策略
                    if has_pair_strategy_mode and pair in pair_strategy_mode:
                        mode = pair_strategy_mode[pair]
                        strategy_types.append(f"自动量化({mode})")

                    summary_line += ', '.join(strategy_types)
                    summary += summary_line + '\n'

                await self._send_msg(summary)
                await self._send_msg('使用 /showpairstrategy <币种> 查看特定交易对的详细策略参数。')

                # 如果有自动量化策略，单独显示一次
                if has_pair_strategy_mode:
                    auto_strategy_summary = '🤖 自动量化策略配置：\n```json\n'
                    auto_strategy_summary += json.dumps(
                        pair_strategy_mode, indent=2
                    )
                    auto_strategy_summary += '\n```'
                    await self._send_msg(auto_strategy_summary)

                # 如果有固定点位策略，单独显示一次
                if has_coin_monitoring:
                    fixed_strategy_summary = '📊 固定点位策略配置：\n```json\n'
                    # fixed_strategy_summary += json.dumps(
                    #     strategy_state['coin_monitoring'], indent=2
                    # )
                    for key, strategies in coin_monitoring.items():
                        strategy_summaries = []
                        for strategy in strategies:
                            strategy_summaries.append(f"{strategy['direction']},{strategy['entry_points'][0]},{','.join([str(i) for i in strategy['exit_points']])},{strategy['stop_loss']}")
                        fixed_strategy_summary += f'/setpairstrategy {key} {';'.join(strategy_summaries)}\n'

                    fixed_strategy_summary += '\n```'

                    # await self._send_msg(fixed_strategy_summary)

                    if len(fixed_strategy_summary) > 3000:
                        # 创建临时文本文件
                        import tempfile
                        import os

                        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='w') as tmp_file:
                            tmp_file.write(fixed_strategy_summary)
                            tmp_file_path = tmp_file.name

                        # 发送文件
                        with open(tmp_file_path, 'rb') as document:
                            await context.bot.send_document(
                                chat_id=update.effective_chat.id,
                                document=document,
                                filename=f"{pair}_strategies.txt",
                                caption=f"Generated prompt for {pair}",
                            )

                        # 删除临时文件
                        os.unlink(tmp_file_path)
                    else:
                        # 如果内容不超过限制，直接发送文本消息
                        await self._send_msg(fixed_strategy_summary)

            # 处理特定交易对参数
            else:
                pair = param
                if not pair.endswith('/USDT:USDT'):
                    pair += '/USDT:USDT'

                # 检查固定点位策略
                has_fixed_strategy = (
                    has_coin_monitoring and pair in coin_monitoring
                )

                # 检查自动量化策略
                has_auto_strategy = (
                    has_pair_strategy_mode and pair in pair_strategy_mode
                )

                if not has_fixed_strategy and not has_auto_strategy:
                    await self._send_msg(f"❌ 未找到 {pair} 的任何策略参数。")
                    return

                # 发送固定点位策略信息 - 以便于复制修改的格式
                if has_fixed_strategy:
                    strategy_json = coin_monitoring[pair]
                    await self._send_msg(f"📋 {pair} 的固定点位策略参数：")

                    # 从JSON提取币种名称(移除'/USDT:USDT')
                    coin_name = pair.split('/')[0]

                    # 检查策略格式类型并提取相应参数
                    if isinstance(strategy_json, list) and len(strategy_json) > 0:
                        # 新格式: 列表格式
                        strs = []
                        for strategy_data in strategy_json:  # 取第一个元素

                            # 提取方向和价格点位
                            direction = strategy_data.get('direction', 'long').lower()

                            # 收集所有价格点位
                            price_points = []

                            # 添加入场点位
                            if 'entry_points' in strategy_data:
                                for price in strategy_data['entry_points']:
                                    price_points.append(str(price))

                            # 添加出场点位
                            if 'exit_points' in strategy_data:
                                for price in strategy_data['exit_points']:
                                    price_points.append(str(price))

                            # 添加止损
                            if 'stop_loss' in strategy_data and strategy_data['stop_loss'] is not None:
                                price_points.append(str(strategy_data['stop_loss']))

                            strs.append(f"{direction},{','.join(price_points)}")

                    # 格式化成便于复制的命令格式
                    formatted_command = (
                        f"/setpairstrategy {coin_name} {';'.join(strs)}"
                    )
                    await self._send_msg(f"```\n{formatted_command}\n```")

                    # 同时仍然提供完整信息以供参考
                    await self._send_msg(
                        f"完整参数：```json\n{json.dumps(strategy_json, indent=2)}\n```"
                    )

                # 发送自动量化策略信息
                if has_auto_strategy:
                    mode = pair_strategy_mode[pair]
                    await self._send_msg(f"🤖 {pair} 的自动量化策略：{mode}")

        except Exception as e:
            logger.exception('显示策略参数时出错: %s', str(e))
            await self._send_msg(f"❌ 显示策略参数时出错: {str(e)}")
