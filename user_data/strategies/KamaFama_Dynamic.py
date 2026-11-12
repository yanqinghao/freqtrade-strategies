"""Telegram initiated positions, deterministic exits, and advisory market reports.

The historical class name is kept for compatibility with existing configurations.
There are deliberately no signal entries or unattended position increases.
"""

import logging
import os
from notification_format import notice
from datetime import datetime, timezone
from copy import deepcopy
from contextlib import contextmanager
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
from assistant_alerts import AlertQueue
from entry_requests import EntryRequests
from entry_pause import EntryPause
from entry_targets import (
    UNIT_FIELDS,
    TargetError,
    has_units,
    convert_targets,
    describe_targets,
    reconcile_basis,
)
from freqtrade.exchange import ROUND_UP, ROUND_DOWN
from entry_pause_monitor import EntryPauseMonitor
from entry_risk import (
    EntryRiskError,
    plan_for_entry,
    assess_entry,
    load_limits,
    enforce,
    install_risk_guard,
)
from market_monitor import HourlyMarketMonitor, SharedVolumeCandles
from position_monitor import PositionMonitor
from market_alerts import MarketAlerts
from extreme_volume_alerts import ExtremeVolumeMonitor
from condition_watch import WatchStore, ConditionMonitor
from exit_policy import plan_exit
from entry_guard import ensure_position_matches, ensure_no_entry_orders
from execution_health import HeartbeatPublisher
from entry_lifecycle import process_entry_orders, install_entry_receipt_guard
from stop_protection import check_protection
from trade_overview import overview_messages
from candle_exits import RULE_GROUPS, validate_rules, closed_candle, describe_rules

logger = logging.getLogger(__name__)


class KamaFama_Dynamic(IStrategy):
    risk_error = EntryRiskError
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
        self.entry_requests = EntryRequests(
            Path(config.get('user_data_dir', 'user_data')) / 'entry_requests.sqlite'
        )
        self.alert_queue = AlertQueue(
            Path(config.get('user_data_dir', 'user_data')) / 'assistant_alerts.sqlite'
        )
        self.watches = WatchStore(self.alert_queue)
        self._unsaved_alerts = {}
        self.risk_limits_file = Path(self.state_file).with_name('assistant_risk.json')
        self._risk_request_ref = None
        self._state_lock = RLock()
        self.market_monitor = None
        self.position_monitor = None
        self.extreme_volume_monitor = None
        self.entry_pause = None
        self.condition_monitor = None
        self._commands = Queue()
        self.execution_bot = None
        self._health_publisher = None

    def attach_execution_bot(self, bot, submit_entry):
        self.execution_bot = bot
        self.entry_pause = EntryPause(self, bot, submit_entry)

    def bot_start(self, **kwargs):
        if self.execution_bot is not None:
            install_entry_receipt_guard(self, self.execution_bot)
            install_risk_guard(self, self.execution_bot)
        if str(getattr(self.config.get('runmode'), 'value', self.config.get('runmode'))) in (
            'live',
            'dry_run',
        ):
            self.market_monitor = HourlyMarketMonitor(
                exchange_name=self.config.get('exchange', {}).get('name', 'binance'),
                futures=self.config.get('trading_mode', 'futures') == 'futures',
                alerts=MarketAlerts(Path(self.state_file).with_name('market_alert_state.json')),
            )
            self.position_monitor = PositionMonitor(
                exchange_name=self.config.get('exchange', {}).get('name', 'binance'),
                futures=self.config.get('trading_mode', 'futures') == 'futures',
            )
            volume_candles = SharedVolumeCandles()
            self.extreme_volume_monitor = ExtremeVolumeMonitor(
                self.alert_queue,
                volume_candles=volume_candles,
                exchange_name=self.config.get('exchange', {}).get('name', 'binance'),
                futures=self.config.get('trading_mode', 'futures') == 'futures',
            )
            if self.entry_pause:
                self.entry_pause.monitor = EntryPauseMonitor(
                    volume_candles=volume_candles,
                    exchange_name=self.config.get('exchange', {}).get('name', 'binance'),
                    futures=self.config.get('trading_mode', 'futures') == 'futures',
                )
            self.condition_monitor = ConditionMonitor(
                self.watches,
                exchange_name=self.config.get('exchange', {}).get('name', 'binance'),
                futures=self.config.get('trading_mode', 'futures') == 'futures',
            )
            if self.execution_bot is not None:
                self._health_publisher = HeartbeatPublisher(
                    self.execution_bot,
                    Path(self.config.get('user_data_dir', 'user_data'))
                    / 'health'
                    / 'assistant.json',
                ).start()

    def submit_command(self, function, *args):
        future = Future()
        self._commands.put((future, function, args))
        return future

    def overview(self, bot_state, current_time):
        return overview_messages(self, Trade.get_open_trades(), bot_state, current_time)

    def preview_entry(self, request, trade):
        plan = plan_for_entry(self, request, trade)
        basis = plan.get('target_basis')
        result = assess_entry(
            self,
            request,
            trade,
            plan,
            rate=basis['rate'] if basis else None,
            amount=basis['quantity'] if basis else None,
        )
        if has_units(plan):
            result.update(
                target_description=describe_targets(plan),
                resolved_targets=plan.get('exit_points', []),
            )
        result['candle_rules'] = describe_rules(plan)
        if trade and trade.has_open_orders:
            result['reasons'].append('该币种有未完成订单，实际入场前须先核实或撤单')
            result['allowed'] = False
        return result

    @contextmanager
    def risk_request(self, ref):
        previous = self._risk_request_ref
        self._risk_request_ref = ref
        try:
            yield
        finally:
            self._risk_request_ref = previous

    def enforce_entry_risk(self, request, trade, plan, rate):
        if load_limits(self.risk_limits_file) is None and not has_units(plan):
            return None
        return enforce(
            assess_entry(
                self, request, trade, plan, rate=rate, ignore_request=self._risk_request_ref
            )
        )

    def ensure_entry_position(self, pair, trade):
        mode = self.config.get('runmode')
        simulated = self.config.get('dry_run', getattr(mode, 'value', mode) == 'dry_run')
        ensure_position_matches(self.dp._exchange, pair, trade, dry_run=simulated)
        ensure_no_entry_orders(self.dp._exchange, pair, dry_run=simulated)

    def ensure_entry_protection(self, trade):
        if trade is not None and not check_protection(
            self, trade, datetime.now(timezone.utc), force=True
        ):
            raise ValueError('止损尚未确认，本次加仓未提交；请先核对 /order 与交易所')

    def bot_loop_start(self, current_time, **kwargs):
        for key, message in list(self._unsaved_alerts.items()):
            self._exit_notice(message, key=key)
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
                process_entry_orders(self, self.execution_bot, trade, current_time)
            for trade in Trade.get_open_trades():
                self._recover_entry_targets(trade)
                self.check_full_exit(trade, current_time)
            for trade in Trade.get_open_trades():
                check_protection(self, trade, current_time)
        if self.entry_pause:
            try:
                self.entry_pause.tick(current_time)
            except Exception as error:
                logger.warning('Entry pause unavailable (%s)', type(error).__name__)
        if self.extreme_volume_monitor:
            self.extreme_volume_monitor.tick(current_time)
        if self.condition_monitor:
            self.condition_monitor.tick(current_time)
        if self.position_monitor:
            try:
                self.position_monitor.tick(current_time, self, Trade.get_open_trades())
            except Exception as error:
                logger.warning('Position reports unavailable (%s)', type(error).__name__)
        if self.market_monitor:
            for message in self.market_monitor.tick(current_time):
                self.dp.send_msg(message, always_send=True)

    def get_market_reports(self):
        if self.market_monitor is None:
            raise ValueError('监控仅在运行中的 live / dry_run 机器人可用')
        return self.market_monitor.reports()

    def ft_bot_cleanup(self):
        if self._health_publisher:
            self._health_publisher.close()
        while not self._commands.empty():
            future, _, _ = self._commands.get_nowait()
            future.cancel()
        if self.market_monitor:
            self.market_monitor.close()
        if self.position_monitor:
            self.position_monitor.close()
        if self.entry_pause and self.entry_pause.monitor:
            self.entry_pause.monitor.close()
        if self.extreme_volume_monitor:
            self.extreme_volume_monitor.close()
        if self.condition_monitor:
            self.condition_monitor.close()
        super().ft_bot_cleanup()

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def informative_pairs(self):
        plans = dict(self.manual_open)
        # A saved trade plan also survives a missing manual-plan file after restart.
        for trade in Trade.get_open_trades():
            plans.setdefault(trade.pair, trade.get_custom_data('assistant_plan', default={}) or {})
        return sorted(
            {
                (pair, plan[group[-1]])
                for pair, plan in plans.items()
                for key, group in RULE_GROUPS.items()
                if plan.get(key) is not None and plan.get(group[-1])
            }
        )

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['enter_tag'] = ''
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        return dataframe

    def set_manual_plan(self, pair, plan, *, reset_rules=True):
        """Merge one explicit command without overwriting other coins' plans."""
        validate_rules(plan)
        plan = deepcopy(plan)
        old = self.manual_open.get(pair) or {}
        for key, group in RULE_GROUPS.items():
            if reset_rules and any(plan.get(field) != old.get(field) for field in group):
                plan['soft_sl_since' if key == 'soft_sl' else 'extend_since'] = datetime.now(
                    timezone.utc
                ).timestamp()

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

    def _target_owner(self, trade):
        current = self.manual_open.get(trade.pair)
        return trade.is_open and (
            not current
            or (
                current.get('trade_id') in (None, trade.id)
                and current.get('direction') == ('short' if trade.is_short else 'long')
            )
        )

    def _store_target_plan(self, trade, plan):
        if not self._target_owner(trade):
            return False
        accepted = False

        def change(state):
            nonlocal accepted
            plans = state.setdefault('manual_open', {})
            current = plans.get(trade.pair)
            if current and (
                current.get('trade_id') not in (None, trade.id)
                or current.get('direction') != plan['direction']
                or {k: current[k] for k in UNIT_FIELDS if k in current}
                != {k: plan[k] for k in UNIT_FIELDS if k in plan}
            ):
                return
            plans[trade.pair] = deepcopy(plan)
            accepted = True

        with self._state_lock:
            state = update_state(self.state_file, change)
            self.manual_open = state.get('manual_open', {})
        return accepted

    def _record_target_entry(self, trade, basis):
        if not basis.get('order_id'):
            return
        state = self._execution(trade)
        seen = state.setdefault('seen', {})
        ident = basis['order_id']
        if seen.get(ident, 0) >= basis['filled']:
            return
        seen[ident] = max(seen.get(ident, 0), basis['filled'])
        state.setdefault('initial_amount', basis['quantity'])
        state.setdefault('stage', 0)
        state.setdefault('stage2_amount', None)
        state.setdefault('tp_filled', {})
        state.setdefault('tp_targets', {})
        self._save_execution(trade, state)

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
        if has_units(plan):
            if not self._target_owner(trade):
                return plan
            try:
                repaired = reconcile_basis(plan, saved)
                if repaired != plan:
                    if not self._store_target_plan(trade, repaired):
                        return plan
                    plan = repaired
            except TargetError as error:
                if not self._execution(trade).get('full_exit'):
                    self.request_full_exit(
                        trade, str(error), exit_tag='manual_target_conversion_failed'
                    )
                # Keep the tighter protection if frozen peers disagree.
                if plan.get('stop_loss') and saved.get('stop_loss'):
                    plan['stop_loss'] = (min if trade.is_short else max)(
                        plan['stop_loss'], saved['stop_loss']
                    )
                return plan
        if self.manual_open.get(trade.pair, {}).get('trade_id') is None:
            self.set_manual_plan(trade.pair, plan, reset_rules=False)
        if (
            saved
            and saved.get('exit_points') != plan.get('exit_points')
            and plan.get('tp_keep_progress_from') != saved.get('exit_points')
        ):
            state = self._execution(trade)
            state.pop('tp2_completed_at', None)
            state.pop('final_tp_started', None)
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

    def _anchor_entry_targets(self, trade, order):
        if not self._target_owner(trade):
            return
        plan = self._plan(trade)
        basis = plan.get('target_basis', {})
        if (
            not has_units(plan)
            or basis.get('anchored')
            or basis.get('failed')
            or self._execution(trade).get('full_exit')
            or trade.amount <= 0
            or order.ft_order_side != trade.entry_side
            or order.ft_is_open
            or order.safe_filled <= 0
        ):
            return
        ref = basis.get('request_ref')
        if ref and not (order.ft_order_tag or '').endswith('|req=' + ref):
            return
        try:
            from entry_risk import observed_quantities

            if float(observed_quantities(trade)[0]) > trade.amount * (1 + 1e-10):
                return  # Wait for native quantity accounting before freezing a basis.
            exchange = self.dp._exchange
            opened = trade.fee_open if trade.fee_open is not None else basis['fee_open']
            closed = trade.fee_close if trade.fee_close is not None else basis['fee_close']

            def rounded(price, short):
                return exchange.price_to_precision(
                    trade.pair, float(price), rounding_mode=ROUND_DOWN if short else ROUND_UP
                )

            resolved = convert_targets(plan, trade.open_rate, trade.amount, opened, closed, rounded)
            old_stop = self._stop_price(trade, plan)
            if old_stop is not None:
                resolved['stop_loss'] = (min if trade.is_short else max)(
                    old_stop, resolved['stop_loss']
                )
            validate_rules(resolved)
            resolved['target_basis'].update(
                anchored=True,
                request_ref=ref,
                order_id=str(order.order_id),
                filled=float(order.safe_filled),
                fee_source='记录的入场费率/预检估算及平仓费率估算，后续不重算',
            )
            # JSON first: a crash before custom-data sync is repaired by _plan.
            if not self._store_target_plan(trade, resolved):
                return
            trade.set_custom_data('assistant_plan', resolved)
            self._record_target_entry(trade, resolved['target_basis'])
        except Exception as error:
            reason = str(error) if isinstance(error, TargetError) else type(error).__name__
            plan.setdefault('target_basis', {}).update(failed=True, error=reason)
            try:
                if not self._store_target_plan(trade, plan):
                    return
                trade.set_custom_data('assistant_plan', plan)
            except Exception as save_error:
                logger.warning(
                    'Unable to save target calibration failure (%s)', type(save_error).__name__
                )
            if self._target_owner(trade):
                self.request_full_exit(
                    trade, '金额/幅度目标校准失败：' + reason, exit_tag='manual_target_conversion_failed'
                )

    def _recover_entry_targets(self, trade):
        if not self._target_owner(trade):
            return
        raw = (
            self.manual_open.get(trade.pair)
            or trade.get_custom_data('assistant_plan', default={})
            or {}
        )
        if not has_units(raw) or self._execution(trade).get('full_exit'):
            return
        plan = self._plan(trade)
        basis = plan.get('target_basis', {})
        if basis.get('anchored'):
            self._record_target_entry(trade, basis)
            return
        if basis.get('failed'):
            self.request_full_exit(
                trade, '金额/幅度目标校准失败，需要核对', exit_tag='manual_target_conversion_failed'
            )
            return
        for order in trade.orders:
            if (
                order.ft_order_side == trade.entry_side
                and not order.ft_is_open
                and order.safe_filled > 0
            ):
                self._anchor_entry_targets(trade, order)
                if self._plan(trade).get('target_basis', {}).get('anchored'):
                    break

    @staticmethod
    def _targets(plan, trade):
        targets = sorted(
            set(float(p) for p in plan.get('exit_points', []) if p), reverse=trade.is_short
        )[:3]
        extension = (trade.get_custom_data('assistant_execution', default={}) or {}).get(
            'extension'
        )
        if extension and len(targets) == 3:
            targets[2] = extension['target']
        return targets

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
        if state.get('extension'):
            candidates.append(state['extension']['stop'])
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

    def _evaluate_candle_rules(self, trade, plan, current_time, current_rate):
        if not any(plan.get(key) is not None for key in RULE_GROUPS):
            return
        state = self._execution(trade)
        if state.get('full_exit'):
            return
        if 'position_opened_at' not in state:
            # Native open_date is the order submission time, possibly hours before a fill.
            filled_at = getattr(trade, 'date_entry_fill_utc', None) or current_time
            state['position_opened_at'] = filled_at.timestamp()
            self._save_execution(trade, state)
        for key, group in RULE_GROUPS.items():
            if plan.get(key) is None:
                continue
            if key == 'extend_trigger':
                if state.get('extension') or state.get('final_tp_started'):
                    continue
                if any(
                    order.ft_order_side == trade.exit_side
                    and (order.ft_order_tag or '').startswith(
                        ('manual_long_tp3', 'manual_short_tp3')
                    )
                    for order in trade.orders
                ):
                    state['final_tp_started'] = True
                    self._save_execution(trade, state)
                    continue
                targets = sorted(set(plan.get('exit_points', [])), reverse=trade.is_short)
                if len(targets) != 3 or state.get('stage') != 2 or trade.has_open_orders:
                    continue
                if current_rate <= targets[2] if trade.is_short else current_rate >= targets[2]:
                    state['final_tp_started'] = True
                    self._save_execution(trade, state)
                    continue
                if 'tp2_completed_at' not in state:
                    state['tp2_completed_at'] = current_time.timestamp()
                    self._save_execution(trade, state)
            since_key = 'soft_sl_since' if key == 'soft_sl' else 'extend_since'
            if plan.get(since_key) is None:
                # Pre-upgrade plans without an activation timestamp start from this loop.
                baselines = state.setdefault('candle_baselines', {})
                if key not in baselines:
                    baselines[key] = current_time.timestamp()
                    self._save_execution(trade, state)
                since = baselines[key]
            else:
                since = plan[since_key]
            after = max(since, state['position_opened_at'])
            if key == 'extend_trigger':
                after = max(after, state['tp2_completed_at'])
            timeframe = plan[group[-1]]
            try:
                frame = self.dp.get_pair_dataframe(pair=trade.pair, timeframe=timeframe)
                candle = closed_candle(frame, timeframe, current_time, after)
            except Exception as error:
                logger.debug(
                    'Closed-candle exit data unavailable for %s %s (%s)',
                    trade.pair,
                    timeframe,
                    type(error).__name__,
                )
                continue
            if candle is None:
                continue
            close = candle['close']
            if key == 'soft_sl':
                if close >= plan[key] if trade.is_short else close <= plan[key]:
                    self.request_full_exit(
                        trade, f'{timeframe}收盘软止损 {close:g}', exit_tag='manual_soft_sl'
                    )
                    return
            else:
                # A wick reaching the original final target before this close wins too.
                if candle['low'] <= targets[2] if trade.is_short else candle['high'] >= targets[2]:
                    state['final_tp_started'] = True
                    self._save_execution(trade, state)
                    continue
                if close <= plan[key] if trade.is_short else close >= plan[key]:
                    state['extension'] = {
                        'target': plan['extend_tp3'],
                        'stop': targets[1],
                        'closed_at': candle['closed_at'],
                        'timeframe': timeframe,
                    }
                    protected = state.get('protected_stop', targets[1])
                    state['protected_stop'] = (min if trade.is_short else max)(
                        protected, targets[1]
                    )
                    self._save_execution(trade, state)
                    self._exit_notice(
                        f'{trade.pair}：{timeframe}收盘 {close:g}，延伸止盈已生效；'
                        f"TP3 改为 {plan['extend_tp3']:g}，保护推进至 TP2 {targets[1]:g}（只收紧）。",
                        key=f'extension:{trade.id}',
                    )

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
        self._evaluate_candle_rules(trade, plan, current_time, current_rate)
        stop = self._stop_price(trade, plan)
        if stop is not None and (current_rate >= stop if trade.is_short else current_rate <= stop):
            return 'manual_pullback_stop'
        full = self._execution(trade).get('full_exit')
        if full and not full.get('blocked'):
            if full.get('attempt_at') and not full.get('retry_ready'):
                return None
            quantity, reason = self.exit_decision(trade, current_rate, trade.amount)
            if quantity is None:
                self.request_full_exit(trade, reason, blocked=True)
                return None
            return full.get('exit_tag', 'manual_minimum_full_exit')
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

    def _exit_notice(self, message, *, key=None):
        logger.warning(message)
        key = key or message
        try:
            self.alert_queue.put(
                key,
                message
                if message.startswith('【')
                else notice('交易提醒', summary=message.split('。'), action='用 /order 核对当前状态。'),
            )
            self._unsaved_alerts.pop(key, None)
            return True
        except Exception as error:
            self._unsaved_alerts[key] = message
            logger.warning('Unable to persist critical alert (%s)', type(error).__name__)
            return False

    def request_full_exit(self, trade, reason, *, blocked=False, exit_tag=None):
        state = self._execution(trade)
        full = state.get('full_exit') or {}
        changed = not full or full.get('blocked') != blocked or full.get('reason') != reason
        full.update(reason=reason, blocked=blocked)
        if exit_tag:
            full['exit_tag'] = exit_tag
        if blocked:
            full['needs_protection_check'] = True
        if changed:
            full['notice_pending'] = (
                f'{trade.pair}：整仓暂不满足平仓规则（{reason}），未平记录已保留，需核实交易所仓位与止损。'
                if blocked
                else f'{trade.pair}：分批退出受限（{reason}），已切换为全平计划，实际成交前不会清理记录。'
            )
            if (
                exit_tag
                in ('manual_soft_sl', 'manual_panel_close', 'manual_target_conversion_failed')
                and not blocked
            ):
                full['notice_pending'] = f'{trade.pair}：{reason}，已记录全平意图，等待成交确认。'
        state['full_exit'] = full
        self._save_execution(trade, state)
        self._flush_exit_notice(trade)

    def _flush_exit_notice(self, trade):
        state = self._execution(trade)
        full = state.get('full_exit') or {}
        message = full.get('notice_pending')
        if message and self._exit_notice(
            notice(
                '退出计划',
                f'{trade.pair} #{trade.id}',
                summary=message.removeprefix(trade.pair + '：').split('。'),
                action='用 /order 核对余仓、退出订单与止损保护。',
            ),
            key=f'exit:{trade.id}',
        ):
            full.pop('notice_pending', None)
            self._save_execution(trade, state)

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
        self._flush_exit_notice(trade)
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
        # Commit notification intent together with execution flags; enqueue may fail independently.
        full['notice_pending'] = (
            f'{trade.pair}：全平仅部分成交，仍有 {trade.amount:g}，将继续退出余量；请核对 /order。'
            if retry
            else f'{trade.pair}：全平失败或未确认完成，仍有 {trade.amount:g}。'
            '已暂停自动重复提交，请核对 /order、交易所仓位及止损，再用 /forceexit 重试。'
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
        state = self._execution(trade)
        full = state.get('full_exit')
        if not full:
            return
        full['notice_pending'] = (
            f'{trade.pair}：全平仅部分成交，仍有 {trade.amount:g}，将继续退出余量。{protection}。'
            if retry
            else f'{trade.pair}：全平失败或未确认完成，仍有 {trade.amount:g}。'
            f'已暂停自动重复提交，请通过 /order、交易所核实后用 /forceexit 重试。{protection}。'
        )
        self._save_execution(trade, state)
        self._flush_exit_notice(trade)

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
            return -trade.stake_amount, full.get('exit_tag', 'manual_minimum_full_exit')
        if exit_reason:
            return -trade.stake_amount, 'manual_protective_exit'
        plan = self._plan(trade)
        if has_units(plan) and not plan.get('target_basis', {}).get('anchored'):
            return None
        targets = self._targets(plan, trade)
        state = self._execution(trade)
        stage = int(state.get('stage', 0))
        if stage >= len(targets):
            return None
        if not (rate <= targets[stage] if trade.is_short else rate >= targets[stage]):
            return None
        if stage == len(targets) - 1:
            state['final_tp_started'] = True
            self._save_execution(trade, state)
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
        if order.ft_order_side == trade.entry_side and not trade.is_open:
            return
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
            plan = self._plan(trade)
            basis = plan.get('target_basis', {})
            if (
                has_units(plan)
                and basis.get('anchored')
                and basis.get('order_id') == str(order.order_id)
            ):
                self._record_target_entry(trade, basis)
                if filled > basis['filled'] + 1e-10:
                    self.request_full_exit(
                        trade, '目标冻结后初始成交量发生变化，需要核对', exit_tag='manual_target_conversion_failed'
                    )
                return
            self._anchor_entry_targets(trade, order)
            state = self._execution(trade)
            filled_at = getattr(trade, 'date_entry_fill_utc', None) or current_time
            state.setdefault('position_opened_at', filled_at.timestamp())
            state.update(
                initial_amount=trade.amount,
                stage2_amount=None,
                tp_filled={},
                tp_targets={},
                stage=0,
            )
            state.pop('tp2_completed_at', None)
            state.pop('final_tp_started', None)
            state['protection'] = {
                'status': 'pending',
                'pending_since': current_time.timestamp(),
                'checked_at': 0,
            }
        elif order.ft_order_side == trade.exit_side:
            tag = order.ft_order_tag or ''
            if not tag.startswith(('manual_long_tp', 'manual_short_tp')):
                return
            stage = int(tag.rsplit('tp', 1)[1].split('_', 1)[0])
            plan = self._plan(trade)
            targets = self._targets(plan, trade)
            if not 1 <= stage < len(targets):
                # The final TP remains active until the actual position closes.
                state['final_tp_started'] = True
                self._save_execution(trade, state)
                return
            state = self._execution(trade)
            required = self._tranche_target(trade, state, stage, len(targets))
            amounts = state.setdefault('tp_filled', {})
            amounts[str(stage)] = amounts.get(str(stage), 0) + delta
            if amounts[str(stage)] >= required * (1 - 1e-8):
                state['stage'] = max(stage, state.get('stage', 0))
                if stage == 1:
                    state['stage2_amount'] = trade.amount
                if stage == 2 and 'tp2_completed_at' not in state:
                    state['tp2_completed_at'] = current_time.timestamp()
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

    def _cancel_for_stop(self, trade, current_time=None):
        if not getattr(self, 'dp', None):
            return False
        rate = self.dp._exchange.get_rate(
            trade.pair, side='exit', is_short=trade.is_short, refresh=False
        )
        plan = self._plan(trade)
        stop = self._stop_price(trade, plan) if plan else None
        if current_time and plan and trade.amount:
            self._evaluate_candle_rules(trade, plan, current_time, rate)
        if self._execution(trade).get('full_exit'):
            return True
        return stop is not None and (rate >= stop if trade.is_short else rate <= stop)

    def check_entry_timeout(self, pair, trade, order, current_time, **kwargs):
        return self._cancel_for_stop(trade, current_time)

    def check_exit_timeout(self, pair, trade, order, current_time, **kwargs):
        # Release a pending TP limit order so a protective exit can run this loop.
        if (order.ft_order_tag or '').startswith(('manual_long_tp', 'manual_short_tp')):
            return self._cancel_for_stop(trade, current_time)
        return False
