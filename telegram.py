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
)
from telegram.constants import MessageLimit, ParseMode
from telegram.error import BadRequest, NetworkError, TelegramError
from telegram.ext import Application, CallbackContext, CallbackQueryHandler, CommandHandler
from telegram.helpers import escape_markdown

from freqtrade.__init__ import __version__
from freqtrade.constants import DUST_PER_COIN, Config
from freqtrade.enums import (
    MarketDirection,
    RPCMessageType,
    SignalDirection,
    TradingMode,
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
from freqtrade.freqllm.key_level_agent import TradingSignalExtractor
from freqtrade.freqllm.db_manager import connect_to_db, get_todays_analysis, insert_analysis_result


MAX_MESSAGE_LENGTH = MessageLimit.MAX_TEXT_LENGTH


logger = logging.getLogger(__name__)

logger.debug('Included module rpc.telegram ...')


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
            ['/chart', '/analysis', '/prompt', '/promptjson'],
            ['/addpair', '/delpair'],
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
            r'/chart$',  # chart命令格式
            r'/analysis$',  # analysis命令格式
            r'/prompt$',  # analysis命令格式
            r'/promptjson$',  # analysis命令格式
            r'/addpair$',
            r'/delpair$',
            r'/setpairstrategy$',
            r'/setpairstrategyauto$',
            r'/delpairstrategy$',
            r'/showpairstrategy$',
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
                partial(self._force_enter, order_side=SignalDirection.LONG),
            ),
            CommandHandler(
                'forceshort', partial(self._force_enter, order_side=SignalDirection.SHORT)
            ),
            CommandHandler('reload_trade', self._reload_trade_from_exchange),
            CommandHandler('trades', self._trades),
            CommandHandler('delete', self._delete_trade),
            CommandHandler(['coo', 'cancel_open_order'], self._cancel_open_order),
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
            CommandHandler('analysis', self._analysis),
            CommandHandler('prompt', self._prompt),
            CommandHandler('promptjson', self._prompt_json),
            CommandHandler('addpair', self._add_pair),
            CommandHandler('delpair', self._del_pair),
            CommandHandler('setpairstrategy', self._set_pair_strategy),
            CommandHandler('setpairstrategyauto', self._set_pair_strategy_auto),
            CommandHandler('delpairstrategy', self._del_pair_strategy),
            CommandHandler('showpairstrategy', self._show_pair_strategy),
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
        best_pair_profit_abs = fmt_coin(stats['best_pair_profit_abs'], stake_cur)
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
                    f"*Best Performing:* `{best_pair}: {best_pair_profit_abs} "
                    f"({best_pair_profit_ratio:.2%})`\n"
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

    async def _force_enter_action(self, pair, price: float | None, order_side: SignalDirection):
        if pair != 'cancel':
            try:

                @safe_async_db
                def _force_enter():
                    self._rpc._rpc_force_entry(pair, price, order_side=order_side)

                loop = asyncio.get_running_loop()
                # Workaround to avoid nested loops
                await loop.run_in_executor(None, _force_enter)
            except RPCException as e:
                logger.exception('Forcebuy error!')
                await self._send_msg(str(e), ParseMode.HTML)

    async def _force_enter_inline(self, update: Update, _: CallbackContext) -> None:
        if update.callback_query:
            query = update.callback_query
            if query.data and '__' in query.data:
                # Input data is "force_enter__<pair|cancel>_<side>"
                payload = query.data.split('__')[1]
                if payload == 'cancel':
                    await query.answer()
                    await query.edit_message_text(text='Force enter canceled.')
                    return
                if payload and '_||_' in payload:
                    pair, side = payload.split('_||_')
                    order_side = SignalDirection(side)
                    await query.answer()
                    await query.edit_message_text(text=f"Manually entering {order_side} for {pair}")
                    await self._force_enter_action(pair, None, order_side)

    @staticmethod
    def _layout_inline_keyboard(
        buttons: list[InlineKeyboardButton], cols=3
    ) -> list[list[InlineKeyboardButton]]:
        return [buttons[i : i + cols] for i in range(0, len(buttons), cols)]

    @authorized_only
    async def _force_enter(
        self, update: Update, context: CallbackContext, order_side: SignalDirection
    ) -> None:
        """
        Handler for /forcelong <asset> <price> and `/forceshort <asset> <price>
        Buys a pair trade at the given or current price
        :param bot: telegram bot
        :param update: message update
        :return: None
        """
        if context.args:
            pair = context.args[0]
            price = float(context.args[1]) if len(context.args) > 1 else None
            await self._force_enter_action(pair, price, order_side)
        else:
            whitelist = self._rpc._rpc_whitelist()['whitelist']
            pair_buttons = [
                InlineKeyboardButton(
                    text=pair, callback_data=f"force_enter__{pair}_||_{order_side}"
                )
                for pair in sorted(whitelist)
            ]
            buttons_aligned = self._layout_inline_keyboard(pair_buttons)

            buttons_aligned.append(
                [InlineKeyboardButton(text='Cancel', callback_data='force_enter__cancel')]
            )
            await self._send_msg(
                msg='Which pair?', keyboard=buttons_aligned, query=update.callback_query
            )

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
    ) -> None:
        if reload_able:
            reply_markup = InlineKeyboardMarkup(
                [
                    [InlineKeyboardButton('Refresh', callback_data=callback_path)],
                ]
            )
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
                await query.edit_message_media(
                    InputMediaDocument(document, filename=filename, caption=msg, parse_mode=parse_mode),
                    reply_markup=reply_markup,
                )
            else:
                await query.edit_message_text(
                    text=msg, parse_mode=parse_mode, reply_markup=reply_markup
                )
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
                await self._send_msg(f"```{prompt}```",reload_able=True,
                        callback_path=callback_data,
                        query=update.callback_query,)

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
            message_text = message_text[len(f'/setpairstrategyauto') :].strip()

        # 检查是否有JSON内容
        if not message_text:
            await self._send_msg('❌ 未提供策略交易对。请在命令后发送交易对数据。')
            return

        # 尝试解析JSON内容
        try:
            with open('/freqtrade/user_data/strategy_state.json', 'r') as f:
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

            with open('/freqtrade/user_data/strategy_state.json', 'w') as f:
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

            with open('/freqtrade/user_data/strategy_state.json', 'r') as f:
                strategy_state = json.load(f)

            strategy_state['coin_monitoring'][pair] = strategy_json

            self._rpc._freqtrade.strategy.coin_monitoring = strategy_state['coin_monitoring']
            with open('/freqtrade/user_data/strategy_state.json', 'w') as f:
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
            strategy_file = '/freqtrade/user_data/strategy_state.json'

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
