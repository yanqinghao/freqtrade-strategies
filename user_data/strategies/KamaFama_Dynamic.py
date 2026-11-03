"""Telegram initiated positions, deterministic exits, and advisory market reports.

The historical class name is kept for compatibility with existing configurations.
There are deliberately no signal entries or unattended position increases.
"""
import logging
import os
from copy import deepcopy
from math import isfinite, nextafter, inf
from pathlib import Path
from threading import RLock
from concurrent.futures import Future
from queue import Queue, Empty

from pandas import DataFrame
from freqtrade.strategy import IStrategy, stoploss_from_absolute
from freqtrade.persistence import Trade
from freqtrade.util import FtPrecise

from assistant_state import read_state, update_state
from market_monitor import HourlyMarketMonitor
from market_alerts import MarketAlerts
from exit_policy import plan_exit

logger = logging.getLogger(__name__)


class KamaFama_Dynamic(IStrategy):
    INTERFACE_VERSION = 3
    can_short = True
    timeframe = '5m'  # Freqtrade execution heartbeat; reports use only 1d and 4h.
    startup_candle_count = 0
    process_only_new_candles = True
    minimal_roi = {}
    stoploss = -0.99  # Explicit per-plan SL replaces this framework fallback after entry.
    trailing_stop = False
    use_custom_stoploss = True
    position_adjustment_enable = True  # Partial exits and explicit RPC additions.
    max_entry_position_adjustment = -1
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'force_entry': 'market',
        'force_exit': 'market',
        'emergency_exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': True,
        'stoploss_on_exchange_interval': 60,
        'stoploss_on_exchange_limit_ratio': 0.99,
    }
    order_time_in_force = {'entry': 'gtc', 'exit': 'gtc'}

    def __init__(self, config):
        super().__init__(config)
        self.state_file = str(
            os.environ.get('ASSISTANT_STATE_FILE')
            or config.get('assistant_state_file')
            or Path(config.get('user_data_dir', 'user_data')) / 'strategy_state_production.json'
        )
        self.manual_open = read_state(self.state_file).get('manual_open', {})
        self._state_lock = RLock()
        self.market_monitor = None
        self._commands = Queue()
        self.execution_bot = None

    def bot_start(self, **kwargs):
        if str(getattr(self.config.get('runmode'), 'value', self.config.get('runmode'))) in (
            'live',
            'dry_run',
        ):
            self.market_monitor = HourlyMarketMonitor(
                exchange_name=self.config.get('exchange', {}).get('name', 'binance'),
                futures=self.config.get('trading_mode', 'futures') == 'futures',
                alerts=MarketAlerts(Path(self.state_file).with_name('market_alert_state.json')),
            )

    def submit_command(self, function, *args):
        future = Future()
        self._commands.put((future, function, args))
        return future

    def bot_loop_start(self, current_time, **kwargs):
        # All Telegram mutations run on the trading thread, before native exit
        # processing. Native RPC can acquire its own lock without reentrancy.
        for _ in range(10):
            try:
                future, function, args = self._commands.get_nowait()
            except Empty:
                break
            if future.set_running_or_notify_cancel():
                try:
                    future.set_result(function(*args))
                except Exception as exc:
                    if hasattr(Trade, 'session'):
                        Trade.rollback()
                    future.set_exception(exc)
        if self.execution_bot is not None:
            for trade in Trade.get_open_trades():
                self.check_full_exit(trade, current_time)
        if self.market_monitor:
            for message in self.market_monitor.tick(current_time):
                self.dp.send_msg(message, always_send=True)

    def get_market_reports(self):
        if self.market_monitor is None:
            raise ValueError('监控仅在运行中的 live / dry_run 机器人可用')
        return self.market_monitor.reports()

    def ft_bot_cleanup(self):
        while not self._commands.empty():
            future, _, _ = self._commands.get_nowait()
            future.cancel()
        if self.market_monitor:
            self.market_monitor.close()
        super().ft_bot_cleanup()

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['enter_tag'] = ''
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        return dataframe

    def set_manual_plan(self, pair, plan):
        """Merge one explicit command without overwriting other coins' plans."""

        def change(state):
            plans = state.setdefault('manual_open', {})
            plans[pair] = deepcopy(plan)

        with self._state_lock:
            state = update_state(self.state_file, change)
            self.manual_open = state.get('manual_open', {})

    def delete_manual_plan(self, pair, trade_id=None):
        def change(state):
            plans = state.setdefault('manual_open', {})
            if trade_id is None or plans.get(pair, {}).get('trade_id') == trade_id:
                plans.pop(pair, None)

        with self._state_lock:
            state = update_state(self.state_file, change)
            self.manual_open = state.get('manual_open', {})

    @staticmethod
    def _execution(trade):
        return deepcopy(trade.get_custom_data('assistant_execution', default={}) or {})

    @staticmethod
    def _save_execution(trade, state):
        # One custom-data commit for all fill accounting and replay markers.
        trade.set_custom_data('assistant_execution', state)

    def _plan(self, trade):
        direction = 'short' if trade.is_short else 'long'
        saved = trade.get_custom_data('assistant_plan', default={}) or {}
        plan = self.manual_open.get(trade.pair)
        if plan and (
            plan.get('direction') != direction or plan.get('trade_id') not in (None, trade.id)
        ):
            plan = None
        plan = deepcopy(plan or saved)
        if not plan:
            return {}
        plan['trade_id'] = trade.id
        if self.manual_open.get(trade.pair, {}).get('trade_id') is None:
            self.set_manual_plan(trade.pair, plan)
        if saved and saved.get('exit_points') != plan.get('exit_points'):
            state = self._execution(trade)
            state.update(
                stage=0,
                initial_amount=trade.amount,
                stage2_amount=None,
                tp_filled={},
                tp_targets={},
            )
            self._save_execution(trade, state)
        if saved != plan:
            trade.set_custom_data('assistant_plan', plan)
        return plan

    @staticmethod
    def _targets(plan, trade):
        return sorted(
            set(float(p) for p in plan.get('exit_points', []) if p), reverse=trade.is_short
        )[:3]

    def _stop_price(self, trade, plan):
        targets = self._targets(plan, trade)
        state = self._execution(trade)
        stage = state.get('stage', 0)
        candidates = []
        sl = plan.get('stop_loss')
        if sl is not None:
            candidates.append(float(sl))
        if stage >= 1:
            # Include trading fees in the cost-protection reference.
            fee = float(getattr(trade, 'fee_open', 0) or 0) + float(
                getattr(trade, 'fee_close', 0) or 0
            )
            candidates.append(trade.open_rate * (1 - fee if trade.is_short else 1 + fee))
        if stage >= 2 and targets:
            candidates.append(targets[0])
        old = state.get('protected_stop')
        if old:
            candidates.append(float(old))
        if not candidates:
            return None
        stop = min(candidates) if trade.is_short else max(candidates)
        if old != stop:
            state['protected_stop'] = stop
            self._save_execution(trade, state)
        return stop

    def custom_stoploss(
        self, pair, trade, current_time, current_rate, current_profit, after_fill=False, **kwargs
    ):
        plan = self._plan(trade)
        if not plan:
            return None
        stop = self._stop_price(trade, plan)
        if stop is None:
            return None
        return stoploss_from_absolute(
            stop, current_rate, is_short=trade.is_short, leverage=trade.leverage or 1
        )

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        if not trade.is_open or not trade.amount or not trade.open_rate:
            return None
        plan = self._plan(trade)
        if not plan:
            return None
        stop = self._stop_price(trade, plan)
        hit = stop is not None and (
            current_rate >= stop if trade.is_short else current_rate <= stop
        )
        if hit:
            stage = self._execution(trade).get('stage', 0)
            return 'manual_pullback_stop' if stage else 'manual_sl_hit'
        full = self._execution(trade).get('full_exit')
        if full and not full.get('blocked'):
            if full.get('attempt_at') and not full.get('retry_ready'):
                return None
            quantity, reason = self.exit_decision(trade, current_rate, trade.amount)
            if quantity is None:
                self.request_full_exit(trade, reason, blocked=True)
                return None
            return 'manual_minimum_full_exit'
        return None

    def exit_decision(self, trade, rate, quantity, *, manual=False):
        exchange = self.dp._exchange
        market = exchange.markets.get(trade.pair, {})
        name = self.config.get('exchange', {}).get('name') or getattr(trade, 'exchange', None)
        exempt = (
            name in ('binance', 'binanceusdm')
            and self.config.get('trading_mode', 'futures') == 'futures'
            and market.get('contract')
            and market.get('linear')
        )
        # Match native checks exactly: these methods return the value that the
        # pinned bot/RPC compares directly with quote-value of the remainder.
        if manual:
            floor = exchange.get_min_pair_stake_amount(
                trade.pair, rate, getattr(trade, 'stop_loss_pct', None) or 0.0
            )
        else:
            floor = exchange.get_min_pair_stake_amount(
                trade.pair, rate, self.stoploss, trade.leverage
            )
        return plan_exit(
            exchange,
            trade.pair,
            trade.amount,
            quantity,
            rate,
            min_notional_exempt=exempt,
            min_remaining_value=floor or 0.0,
        )

    def _exit_notice(self, message):
        logger.warning(message)
        try:
            self.dp.send_msg(message, always_send=True)
        except Exception:
            logger.exception('Unable to enqueue exit alert')

    def request_full_exit(self, trade, reason, *, blocked=False):
        state = self._execution(trade)
        full = state.get('full_exit') or {}
        changed = not full or full.get('blocked') != blocked or full.get('reason') != reason
        full.update(reason=reason, blocked=blocked)
        if blocked:
            full['needs_protection_check'] = True
        state['full_exit'] = full
        self._save_execution(trade, state)
        if changed:
            self._exit_notice(
                f'{trade.pair}：整仓暂不满足平仓规则（{reason}），未平记录已保留，需核实交易所仓位与止损。'
                if blocked
                else f'{trade.pair}：分批退出受限（{reason}），已切换为全平计划，实际成交前不会清理记录。'
            )

    def mark_full_exit_attempt(self, trade, current_time):
        state = self._execution(trade)
        full = state.get('full_exit')
        if not full:
            return
        full.update(
            attempt_at=current_time.timestamp(),
            attempt_done=False,
            blocked=False,
            retry_ready=False,
            orders_before=[str(o.order_id) for o in trade.orders],
        )
        self._save_execution(trade, state)

    def check_full_exit(self, trade, current_time):
        state = self._execution(trade)
        full = state.get('full_exit')
        if not full or not trade.is_open or not trade.amount:
            return
        attempt = full.get('attempt_at')
        check_protection = full.get('needs_protection_check', False)
        retry = False
        if not check_protection:
            if not attempt or full.get('attempt_done') or current_time.timestamp() <= attempt:
                return
            if trade.has_open_orders and current_time.timestamp() - attempt < 30:
                return
            previous = set(full.get('orders_before', []))
            retry = not trade.has_open_orders and any(
                str(o.order_id) not in previous
                and o.ft_order_side == trade.exit_side
                and float(o.filled or 0) > 0
                and o.status in ('closed', 'canceled', 'cancelled', 'expired')
                for o in trade.orders
            )
        full.update(
            attempt_done=True, needs_protection_check=False, blocked=not retry, retry_ready=retry
        )
        self._save_execution(trade, state)
        protection = '尚无法核验止损'
        bot = self.execution_bot
        if bot is not None and self.order_types.get('stoploss_on_exchange'):
            try:
                with bot._exit_lock:
                    bot.handle_stoploss_on_exchange(trade)
                protection = (
                    '框架已核验/尝试恢复止损，当前有止损记录'
                    if getattr(trade, 'open_sl_orders', [])
                    else '框架已尝试恢复止损，但仍无有效止损记录，请立即检查交易所'
                )
            except Exception:
                protection = '止损核验失败，请立即检查交易所'
                logger.exception('Unable to verify stop protection for %s', trade.pair)
        if not trade.is_open or not trade.amount:
            return
        self._exit_notice(
            f'{trade.pair}：全平仅部分成交，仍有 {trade.amount:g}，将继续退出余量。{protection}。'
            if retry
            else f'{trade.pair}：全平失败或未确认完成，仍有 {trade.amount:g}。'
            f'已暂停自动重复提交，请通过 /order、交易所核实后用 /forceexit 重试。{protection}。'
        )

    def _tranche_target(self, trade, state, stage, target_count):
        key = str(stage)
        targets = state.setdefault('tp_targets', {})
        if key not in targets:
            if stage == 1:
                baseline = state.setdefault('initial_amount', trade.amount)
                fraction = 0.5 if target_count == 2 else 0.3
            else:
                baseline = state.get('stage2_amount') or trade.amount
                fraction = 0.5
            quantity = baseline * fraction
            exchange = getattr(getattr(self, 'dp', None), '_exchange', None)
            if exchange:
                quantity = float(exchange.amount_to_contract_precision(trade.pair, quantity))
            targets[key] = quantity
        return targets[key]

    def adjust_trade_position(
        self,
        trade,
        current_time,
        current_rate,
        current_profit,
        min_stake=None,
        max_stake=None,
        current_exit_rate=None,
        **kwargs,
    ):
        """Only reduce positions. Automatic entries and all legacy DCA are disabled."""
        if not trade.is_open or not trade.amount or not trade.open_rate or trade.has_open_orders:
            return None
        rate = current_exit_rate if current_exit_rate is not None else current_rate
        exit_reason = self.custom_exit(trade.pair, trade, current_time, rate, current_profit)
        full = self._execution(trade).get('full_exit')
        if full:
            if full.get('blocked') or (full.get('attempt_at') and not full.get('retry_ready')):
                return None
            self.mark_full_exit_attempt(trade, current_time)
            return -trade.stake_amount, 'manual_minimum_full_exit'
        if exit_reason:
            return -trade.stake_amount, 'manual_protective_exit'
        plan = self._plan(trade)
        targets = self._targets(plan, trade)
        state = self._execution(trade)
        stage = int(state.get('stage', 0))
        if stage >= len(targets):
            return None
        if not (rate <= targets[stage] if trade.is_short else rate >= targets[stage]):
            return None
        if stage == len(targets) - 1:
            amount = trade.amount
        else:
            target = self._tranche_target(trade, state, stage + 1, len(targets))
            already = state.get('tp_filled', {}).get(str(stage + 1), 0)
            amount = min(trade.amount, max(0, target - already))
            self._save_execution(trade, state)
        quantity, reason = self.exit_decision(trade, rate, amount)
        if reason:
            self.request_full_exit(trade, reason, blocked=quantity is None)
            if quantity is None:
                return None
            self.mark_full_exit_attempt(trade, current_time)
            return -trade.stake_amount, 'manual_minimum_full_exit'
        amount = quantity
        # Use the native core's string arithmetic for the inverse, then verify
        # its actual roundtrip before returning a stake-based reduction.
        stake = min(
            trade.stake_amount,
            nextafter(
                float(FtPrecise(trade.stake_amount) * FtPrecise(amount) / FtPrecise(trade.amount)),
                inf,
            ),
        )
        roundtrip = self.dp._exchange.amount_to_contract_precision(
            trade.pair,
            abs(float(FtPrecise(-stake) * FtPrecise(trade.amount) / FtPrecise(trade.stake_amount))),
        )
        if roundtrip != amount:
            quantity, full_reason = self.exit_decision(trade, rate, trade.amount)
            self.request_full_exit(
                trade, full_reason or 'quantity_roundtrip', blocked=quantity is None
            )
            if quantity is None:
                return None
            self.mark_full_exit_attempt(trade, current_time)
            return -trade.stake_amount, 'manual_minimum_full_exit'
        return -stake, f"manual_{'short' if trade.is_short else 'long'}_tp{stage + 1}"

    def order_filled(self, pair, trade, order, current_time, **kwargs):
        filled = float(order.filled or 0)
        if filled <= 0:
            return
        if order.ft_order_side in (trade.exit_side, 'stoploss') and (
            not trade.is_open or trade.amount <= 0
        ):
            # Idempotent and ownership-checked, even when an old fill is replayed.
            self.delete_manual_plan(pair, trade_id=trade.id)
            return
        state = self._execution(trade)
        seen = state.get('seen', {})
        delta = max(0, filled - seen.get(str(order.order_id), 0))
        if delta == 0:
            return
        if order.ft_order_side == trade.entry_side:
            self._plan(trade)
            state = self._execution(trade)
            state.update(
                initial_amount=trade.amount,
                stage2_amount=None,
                tp_filled={},
                tp_targets={},
                stage=0,
            )
        elif order.ft_order_side == trade.exit_side:
            tag = order.ft_order_tag or ''
            if not tag.startswith(('manual_long_tp', 'manual_short_tp')):
                return
            stage = int(tag.rsplit('tp', 1)[1].split('_', 1)[0])
            plan = self._plan(trade)
            targets = self._targets(plan, trade)
            if not 1 <= stage < len(targets):
                # The final TP remains active until the actual position closes.
                return
            state = self._execution(trade)
            required = self._tranche_target(trade, state, stage, len(targets))
            amounts = state.setdefault('tp_filled', {})
            amounts[str(stage)] = amounts.get(str(stage), 0) + delta
            if amounts[str(stage)] >= required * (1 - 1e-8):
                state['stage'] = max(stage, state.get('stage', 0))
                if stage == 1:
                    state['stage2_amount'] = trade.amount
        else:
            return
        state.setdefault('seen', {})[str(order.order_id)] = filled
        self._save_execution(trade, state)

    def confirm_trade_entry(
        self,
        pair,
        order_type,
        amount,
        rate,
        time_in_force,
        current_time,
        entry_tag,
        side='long',
        **kwargs,
    ):
        # Signal generation is disabled; explicit Telegram/RPC uses a manual tag.
        plan = self.manual_open.get(pair, {})
        if not (entry_tag or '').startswith('manual_') or plan.get('direction') != side:
            return False
        try:
            sl = float(plan['stop_loss'])
            return isfinite(sl) and sl > 0 and (sl > rate if side == 'short' else sl < rate)
        except (KeyError, ValueError, TypeError):
            return False

    def confirm_trade_exit(
        self,
        pair,
        trade,
        order_type,
        amount,
        rate,
        time_in_force,
        exit_reason,
        current_time,
        **kwargs,
    ):
        self.mark_full_exit_attempt(trade, current_time)
        return True  # Never remove a plan or reject a protective exit before a fill.

    def _cancel_for_stop(self, trade):
        if not getattr(self, 'dp', None):
            return False
        rate = self.dp._exchange.get_rate(
            trade.pair, side='exit', is_short=trade.is_short, refresh=False
        )
        plan = self._plan(trade)
        stop = self._stop_price(trade, plan) if plan else None
        return stop is not None and (rate >= stop if trade.is_short else rate <= stop)

    def check_entry_timeout(self, pair, trade, order, current_time, **kwargs):
        return self._cancel_for_stop(trade)

    def check_exit_timeout(self, pair, trade, order, current_time, **kwargs):
        # Release a pending TP limit order so a protective exit can run this loop.
        if (order.ft_order_tag or '').startswith(('manual_long_tp', 'manual_short_tp')):
            return self._cancel_for_stop(trade)
        return False
