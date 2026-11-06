"""Closed-candle rules use real strategy accounting with deterministic market data."""

import unittest
import asyncio
from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace

import pandas as pd

import test_manual_strategy as manual_tests
import test_telegram_orders as order_tests
from test_telegram_assistant import load_telegram


class CandleRuleInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.module = load_telegram()

    def test_parser_and_history_roundtrip(self):
        request = self.module.parse_entry(
            [
                'BTC',
                '10',
                '3',
                'sl=110',
                'tp=90,80,70',
                'soft_sl=105',
                'soft_tf=4h',
                'extend_trigger=75',
                'extend_tp3=60',
                'extend_tf=4h',
            ],
            'short',
        )
        self.assertEqual(request['soft_sl'], 105)
        self.assertEqual(request['extend_tp3'], 60)
        self.assertEqual(
            self.module.parse_entry(self.module.entry_command(request).split()[1:], 'short'),
            request,
        )

    def test_incomplete_nonfinite_and_unsupported_rules_rejected(self):
        for tail in (
            ['soft_sl=105'],
            ['soft_tf=4h'],
            ['soft_sl=nan', 'soft_tf=4h'],
            ['soft_sl=105', 'soft_tf=3m'],
            ['extend_trigger=75', 'extend_tf=4h'],
            ['extend_trigger=75', 'extend_tp3=60', 'extend_tf=0h'],
        ):
            with self.subTest(tail=tail), self.assertRaises(ValueError):
                self.module.parse_entry(['BTC', '10', '3', 'sl=110', *tail], 'short')

    def test_disable_rule_roundtrip(self):
        request = self.module.parse_entry(
            ['BTC', '10', '3', 'sl=110', 'soft_sl=off', 'extend_trigger=off'], 'short'
        )
        self.assertIsNone(request['soft_sl'])
        self.assertIsNone(request['extend_tp3'])
        self.assertEqual(
            self.module.parse_entry(self.module.entry_command(request).split()[1:], 'short'),
            request,
        )


class CandleExecutionTests(unittest.TestCase):
    setUp = manual_tests.ManualStrategyTests.setUp
    adjust = manual_tests.ManualStrategyTests.adjust

    def configure(self, *, short=False, soft=False):
        self.trade.is_short = short
        self.trade.entry_side, self.trade.exit_side = ('sell', 'buy') if short else ('buy', 'sell')
        self.trade.open_date_utc = self.now - timedelta(hours=12)
        plan = self.strategy.manual_open[self.trade.pair]
        plan.update(
            direction='short' if short else 'long',
            stop_loss=110 if short else 90,
            exit_points=[90, 80, 70] if short else [110, 120, 130],
        )
        if soft:
            plan.update(
                soft_sl=105 if short else 95,
                soft_tf='4h',
                soft_sl_since=(self.now - timedelta(hours=8)).timestamp(),
            )
        else:
            plan.update(
                extend_trigger=75 if short else 125,
                extend_tp3=60 if short else 140,
                extend_tf='4h',
                extend_since=(self.now - timedelta(hours=8)).timestamp(),
            )
        self.strategy._save_execution(
            self.trade,
            {
                'stage': 0 if soft else 2,
                'position_opened_at': (self.now - timedelta(hours=12)).timestamp(),
                'tp2_completed_at': (self.now - timedelta(hours=5)).timestamp(),
            },
        )
        return plan

    def candle(self, close, *, high=None, low=None, hours=4):
        self.frame = pd.DataFrame(
            [
                {
                    'date': self.now - timedelta(hours=hours),
                    'open': close,
                    'high': close if high is None else high,
                    'low': close if low is None else low,
                    'close': close,
                    'volume': 10,
                }
            ]
        )
        self.strategy.dp.get_pair_dataframe = lambda **kw: self.frame.copy()

    def test_extension_locks_target_and_tp2_protection_both_directions(self):
        for short in (False, True):
            with self.subTest(short=short):
                self.trade.data.clear()
                plan = self.configure(short=short)
                trigger, target = plan['extend_trigger'], plan['extend_tp3']
                self.candle(trigger)
                self.assertIsNone(self.adjust(trigger))
                state = self.strategy._execution(self.trade)
                self.assertEqual(state['extension']['target'], target)
                self.assertEqual(
                    self.strategy._stop_price(self.trade, plan), plan['exit_points'][1]
                )
                self.assertIsNone(self.adjust(plan['exit_points'][2]))
                result = self.adjust(target)
                self.assertEqual(
                    result,
                    (-self.trade.stake_amount, 'manual_short_tp3' if short else 'manual_long_tp3'),
                )

    def test_extension_survives_strategy_restart_and_candle_disappearing(self):
        plan = self.configure()
        self.candle(125)
        self.adjust(125)
        replacement = type(self.strategy)(self.strategy.config)
        replacement.dp = self.strategy.dp
        self.strategy = replacement
        self.frame = pd.DataFrame()
        self.assertIsNone(self.adjust(130))
        self.assertEqual(self.strategy._stop_price(self.trade, plan), 120)
        self.assertIn(
            'pullback', self.strategy.custom_exit(self.trade.pair, self.trade, self.now, 120, 0)
        )

    def test_original_tp3_before_confirmation_and_partial_exit_cannot_extend(self):
        self.configure()
        self.candle(125)
        self.assertEqual(self.adjust(130)[1], 'manual_long_tp3')
        self.assertNotIn('extension', self.strategy._execution(self.trade))
        self.assertIsNone(self.adjust(125))
        self.assertNotIn('extension', self.strategy._execution(self.trade))

    def test_candle_wick_at_original_tp3_does_not_extend(self):
        self.configure()
        self.candle(125, high=130)
        self.assertIsNone(self.adjust(125))
        self.assertNotIn('extension', self.strategy._execution(self.trade))
        self.assertEqual(self.strategy._stop_price(self.trade, self.configure()), 110)

    def test_extension_waits_for_completed_tp2_and_new_candle(self):
        self.configure()
        self.candle(125)
        for state in (
            {'stage': 1},
            {'stage': 2},
            {'stage': 2, 'tp2_completed_at': self.now.timestamp()},
        ):
            with self.subTest(state=state):
                self.strategy._save_execution(self.trade, state)
                self.adjust(125)
                self.assertNotIn('extension', self.strategy._execution(self.trade))

    def test_open_stale_invalid_and_pre_rule_candles_do_not_trigger(self):
        plan = self.configure()
        for hours in (0, 8):
            self.candle(125, hours=hours)
            self.assertIsNone(self.adjust(125))
            self.assertNotIn('extension', self.strategy._execution(self.trade))
        self.candle(float('nan'))
        self.assertIsNone(self.adjust(125))
        self.assertNotIn('extension', self.strategy._execution(self.trade))
        self.candle(125)
        plan['extend_since'] = self.now.timestamp()
        self.assertIsNone(self.adjust(125))
        self.assertNotIn('extension', self.strategy._execution(self.trade))

    def test_missing_market_data_does_not_disable_original_sl_or_tp(self):
        self.configure()
        self.assertIn(
            'pullback', self.strategy.custom_exit(self.trade.pair, self.trade, self.now, 109, 0)
        )
        self.assertEqual(self.adjust(130)[1], 'manual_long_tp3')

    def test_saved_plan_rebind_keeps_original_activation_time(self):
        plan = self.configure(soft=True)
        self.trade.set_custom_data('assistant_plan', deepcopy(plan))
        self.strategy.manual_open = {}
        rebound = self.strategy._plan(self.trade)
        self.assertEqual(
            self.strategy.manual_open[self.trade.pair]['soft_sl_since'], rebound['soft_sl_since']
        )

    def test_soft_stop_equality_is_durable_even_when_price_recovers(self):
        for short in (False, True):
            with self.subTest(short=short):
                self.trade.data.clear()
                plan = self.configure(short=short, soft=True)
                self.candle(plan['soft_sl'])
                reason = self.strategy.custom_exit(self.trade.pair, self.trade, self.now, 100, 0)
                self.assertEqual(reason, 'manual_soft_sl')
                self.frame = pd.DataFrame()
                self.assertEqual(
                    self.strategy.custom_exit(self.trade.pair, self.trade, self.now, 100, 0),
                    'manual_soft_sl',
                )
                self.assertEqual(self.adjust(100)[1], 'manual_soft_sl')

    def test_pending_entry_first_fill_cannot_use_pre_position_close(self):
        self.configure(soft=True)
        self.strategy._save_execution(self.trade, {})
        self.candle(95)
        self.now += timedelta(minutes=5)
        first = SimpleNamespace(order_id='first', ft_order_side='buy', filled=10)
        self.strategy.order_filled(self.trade.pair, self.trade, first, self.now)
        self.assertIsNone(self.strategy.custom_exit(self.trade.pair, self.trade, self.now, 100, 0))
        state = self.strategy._execution(self.trade)
        self.assertEqual(state['position_opened_at'], self.now.timestamp())
        self.now += timedelta(minutes=1)
        addition = SimpleNamespace(order_id='add', ft_order_side='buy', filled=1)
        self.strategy.order_filled(self.trade.pair, self.trade, addition, self.now)
        self.assertEqual(
            self.strategy._execution(self.trade)['position_opened_at'], state['position_opened_at']
        )

    def test_legacy_position_uses_real_fill_date_or_safe_first_observation(self):
        self.configure(soft=True)
        self.candle(95)
        self.now += timedelta(minutes=5)
        for filled_at in (None, self.now):
            with self.subTest(filled_at=filled_at):
                self.strategy._save_execution(self.trade, {})
                self.trade.date_entry_fill_utc = filled_at
                self.assertIsNone(
                    self.strategy.custom_exit(self.trade.pair, self.trade, self.now, 100, 0)
                )
        self.strategy._save_execution(self.trade, {})
        self.trade.date_entry_fill_utc = self.now - timedelta(hours=1)
        self.assertEqual(
            self.strategy.custom_exit(self.trade.pair, self.trade, self.now, 100, 0),
            'manual_soft_sl',
        )

    def test_hard_stop_has_priority_over_extension(self):
        self.configure()
        self.candle(125)
        self.assertIn(
            'pullback', self.strategy.custom_exit(self.trade.pair, self.trade, self.now, 109, 0)
        )
        self.assertNotIn('extension', self.strategy._execution(self.trade))

    def test_tp2_partial_fill_does_not_arm_extension_until_completed(self):
        self.configure()
        self.strategy._save_execution(
            self.trade, {**self.strategy._execution(self.trade), 'stage': 1, 'stage2_amount': 10}
        )
        state = self.strategy._execution(self.trade)
        state.pop('tp2_completed_at')
        self.strategy._save_execution(self.trade, state)
        order = SimpleNamespace(
            order_id='tp2',
            ft_order_side='sell',
            ft_order_tag='manual_long_tp2',
            filled=2,
            status='canceled',
        )
        self.strategy.order_filled(self.trade.pair, self.trade, order, self.now)
        self.assertNotIn('tp2_completed_at', self.strategy._execution(self.trade))
        order.filled = 5
        self.strategy.order_filled(self.trade.pair, self.trade, order, self.now)
        self.assertEqual(
            self.strategy._execution(self.trade)['tp2_completed_at'], self.now.timestamp()
        )
        self.now += timedelta(hours=4)
        self.candle(125)
        self.assertIsNone(self.adjust(125))
        self.assertIn('extension', self.strategy._execution(self.trade))

    def test_pending_entry_or_tp_is_canceled_after_soft_exit(self):
        self.configure(soft=True)
        self.candle(95)
        self.trade.has_open_orders = True
        order = SimpleNamespace(ft_order_tag='manual_long_tp1')
        self.assertTrue(
            self.strategy.check_exit_timeout(self.trade.pair, self.trade, order, self.now)
        )
        self.assertTrue(
            self.strategy.check_entry_timeout(self.trade.pair, self.trade, order, self.now)
        )

    def test_already_tighter_stop_is_not_loosened(self):
        plan = self.configure()
        self.strategy._save_execution(
            self.trade, {**self.strategy._execution(self.trade), 'protected_stop': 123}
        )
        self.candle(125)
        self.adjust(125)
        self.assertEqual(self.strategy._stop_price(self.trade, plan), 123)

    def test_old_final_tp_order_prevents_extension_after_upgrade(self):
        self.configure()
        self.trade.orders = [SimpleNamespace(ft_order_side='sell', ft_order_tag='manual_long_tp3')]
        self.candle(125)
        self.assertIsNone(self.adjust(125))
        self.assertNotIn('extension', self.strategy._execution(self.trade))


class CandlePlanValidationTests(unittest.TestCase):
    setUp = order_tests.TelegramOrderTests.setUp
    position = order_tests.TelegramOrderTests.position

    def test_invalid_merged_extension_rejected_before_plan_mutation(self):
        trade = self.position()
        self.strategy.set_manual_plan(trade.pair, self.request)
        before = self.strategy.manual_open.copy()
        for trigger, target in ((119, 140), (130, 140), (125, 129)):
            with self.subTest(trigger=trigger), self.assertRaises(ValueError):
                self.bot._update_plan(
                    str(trade.id),
                    [f'extend_trigger={trigger}', f'extend_tp3={target}', 'extend_tf=4h'],
                )
            self.assertEqual(self.strategy.manual_open, before)

    def test_existing_plan_accepts_extension_and_information_pairs(self):
        trade = self.position()
        self.strategy.set_manual_plan(trade.pair, self.request)
        self.bot._update_plan(
            str(trade.id), ['extend_trigger=125', 'extend_tp3=140', 'extend_tf=4h']
        )
        self.assertIn((trade.pair, '4h'), self.strategy.informative_pairs())
        self.assertEqual(self.strategy.manual_open[trade.pair]['extend_tp3'], 140)

    def test_active_extension_blocks_edits_and_additions_but_allows_sl(self):
        trade = self.position()
        self.strategy.set_manual_plan(
            trade.pair,
            {**self.request, 'extend_trigger': 125, 'extend_tp3': 140, 'extend_tf': '4h'},
        )
        self.strategy._save_execution(
            trade, {'stage': 2, 'extension': {'target': 140, 'stop': 120}}
        )
        for tokens in (['tp=110,120,150'], ['extend_trigger=off']):
            with self.subTest(tokens=tokens), self.assertRaisesRegex(ValueError, '延伸'):
                self.bot._update_plan(str(trade.id), tokens)
        self.bot._update_plan(str(trade.id), ['sl=121'])
        with self.assertRaisesRegex(ValueError, '延伸'):
            self.bot._submit_entry(self.request)
        with self.assertRaisesRegex(ValueError, '延伸'):
            self.strategy.preview_entry(self.request, trade)

    def test_overview_and_manual_show_live_extended_target_without_mutation(self):
        from trade_overview import overview_messages
        from datetime import datetime, timezone

        trade = self.position()
        self.strategy.set_manual_plan(trade.pair, {**self.request, 'soft_sl': 95, 'soft_tf': '4h'})
        state = {'stage': 2, 'extension': {'target': 140, 'stop': 120}}
        self.strategy._save_execution(trade, state)
        before = deepcopy(self.strategy.manual_open)
        text = '\n'.join(
            overview_messages(self.strategy, [trade], 'RUNNING', datetime.now(timezone.utc))
        )
        self.assertIn('延伸已生效', text)
        self.assertIn('140', text)
        self.assertIn('软SL', text)
        messages = []

        async def send(text, **kwargs):
            messages.append(text)

        self.bot._send_msg = send
        asyncio.run(self.bot._manual.__wrapped__(self.bot, None, None))
        self.assertIn('延伸已生效', '\n'.join(messages))
        self.assertEqual(self.strategy.manual_open, before)
        self.assertEqual(self.strategy._execution(trade), state)

    def test_disabling_pending_rule_and_changing_price_rearms_timestamp(self):
        trade = self.position()
        self.strategy.set_manual_plan(trade.pair, self.request)
        self.bot._update_plan(str(trade.id), ['soft_sl=95', 'soft_tf=4h'])
        stamp = self.strategy.manual_open[trade.pair]['soft_sl_since']
        self.bot._update_plan(str(trade.id), ['sl=92'])
        self.assertEqual(self.strategy.manual_open[trade.pair]['soft_sl_since'], stamp)
        self.bot._update_plan(str(trade.id), ['soft_sl=96', 'soft_tf=4h'])
        self.assertGreaterEqual(self.strategy.manual_open[trade.pair]['soft_sl_since'], stamp)
        self.bot._update_plan(str(trade.id), ['soft_sl=off'])
        self.assertNotIn((trade.pair, '4h'), self.strategy.informative_pairs())

    def test_native_submission_guard_blocks_activated_extension_without_risk_file(self):
        from entry_risk import install_risk_guard, EntryRiskError

        trade = self.position()
        self.strategy._save_execution(trade, {'extension': {'target': 140, 'stop': 120}})
        calls = []
        self.exchange.create_order = lambda **kw: calls.append(kw)
        install_risk_guard(self.strategy, self.rpc._freqtrade)
        with self.assertRaisesRegex(EntryRiskError, '延伸'):
            self.exchange.create_order(pair=trade.pair, side='buy', reduceOnly=False)
        self.assertEqual(calls, [])
        self.exchange.create_order(pair=trade.pair, side='sell', reduceOnly=True)
        self.assertEqual(len(calls), 1)

    def test_rejected_submission_restores_rule_activation_time(self):
        from freqtrade.rpc import RPCException

        prior = {**self.request, 'soft_sl': 95, 'soft_tf': '4h'}
        self.strategy.set_manual_plan(self.request['pair'], prior)
        before = deepcopy(self.strategy.manual_open)

        def reject(*args, **kwargs):
            raise RPCException('rejected')

        self.rpc._rpc_force_entry = reject
        with self.assertRaises(RPCException):
            self.bot._submit_entry({**self.request, 'soft_sl': 96, 'soft_tf': '4h'})
        self.assertEqual(self.strategy.manual_open, before)
