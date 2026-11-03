import importlib
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from test_exit_policy import RuleExchange

# The repo's patched telegram.py is not the python-telegram-bot package.
ROOT = Path(__file__).resolve().parents[1]
sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != ROOT]
sys.path.insert(0, str(ROOT / 'user_data' / 'strategies'))


class Position(SimpleNamespace):
    def get_custom_data(self, key, default=None):
        return self.data.get(key, default)

    def set_custom_data(self, key, value):
        self.data[key] = value


class ManualStrategyTests(unittest.TestCase):
    def setUp(self):
        try:
            module = importlib.import_module('KamaFama_Dynamic')
        except ModuleNotFoundError as exc:
            self.fail(f'strategy still depends on removed hedge/AI modules: {exc}')
        from freqtrade.enums import RunMode

        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.strategy = module.KamaFama_Dynamic(
            {
                'runmode': RunMode.DRY_RUN,
                'user_data_dir': Path(self.directory.name),
                'stake_currency': 'USDT',
                'exchange': {'name': 'binance'},
            }
        )
        self.strategy.state_file = str(Path(self.directory.name) / 'strategy_state.json')
        Path(self.strategy.state_file).write_text('{}')
        self.now = datetime(2026, 11, 3, 12, tzinfo=timezone.utc)
        self.trade = Position(
            id=1,
            pair='BTC/USDT:USDT',
            is_short=False,
            is_open=True,
            open_rate=100.0,
            amount=10.0,
            stake_amount=100.0,
            leverage=10.0,
            enter_tag='manual_long',
            entry_side='buy',
            exit_side='sell',
            has_open_orders=False,
            data={},
            orders=[],
        )
        self.strategy.manual_open = {
            self.trade.pair: {
                'direction': 'long',
                'stop_loss': 90.0,
                'exit_points': [110.0, 120.0, 130.0],
                'entries': [100.0, 95.0, 90.0],
                'size': 1000.0,
                'leverage': 10.0,
                'trade_id': 1,
            }
        }
        self.exchange = RuleExchange(minimum=0.001, step='.001', cost=0.0, pair=self.trade.pair)
        self.messages = []
        self.strategy.dp = SimpleNamespace(
            _exchange=self.exchange,
            send_msg=lambda message, **kwargs: self.messages.append(message),
        )

    def adjust(self, rate):
        return self.strategy.adjust_trade_position(
            self.trade,
            self.now,
            rate,
            (rate - 100) / 10,
            min_stake=1,
            max_stake=10000,
            current_exit_rate=rate,
            current_entry_rate=rate,
            current_entry_profit=(rate - 100) / 10,
            current_exit_profit=(rate - 100) / 10,
        )

    def test_live_monitor_enables_alerts_with_separate_state(self):
        self.strategy.bot_start()
        self.addCleanup(self.strategy.market_monitor.close)
        alerts = getattr(self.strategy.market_monitor, '_alerts', None)
        self.assertIsNotNone(alerts, 'production monitoring must enable closed-candle alerts')
        self.assertEqual(alerts.state_file.name, 'market_alert_state.json')
        self.assertNotEqual(str(alerts.state_file), self.strategy.state_file)

    def test_all_entry_signals_are_disabled_even_with_legacy_configs(self):
        import pandas as pd

        self.strategy.coin_monitoring = {
            self.trade.pair: [{'direction': 'long', 'entry_points': [200]}]
        }
        frame = pd.DataFrame(
            {'close': [1.0, 100.0, 1000.0], 'enter_long': [1, 1, 1], 'enter_short': [1, 1, 1]}
        )
        result = self.strategy.populate_entry_trend(frame, {'pair': self.trade.pair})
        self.assertEqual(result.enter_long.tolist(), [0, 0, 0])
        self.assertEqual(result.enter_short.tolist(), [0, 0, 0])

    def test_no_scale_in_at_any_price_even_with_legacy_entries(self):
        for rate in (100, 99, 95, 91):
            self.assertIsNone(self.adjust(rate))

    def test_partial_take_profit_does_not_advance_until_fill(self):
        result = self.adjust(111)
        self.assertAlmostEqual(result[0], -30.0)
        self.assertEqual(self.trade.get_custom_data('assistant_execution', {}).get('stage', 0), 0)
        # A rejected/unsubmitted request can be retried without skipping TP1.
        self.assertEqual(self.adjust(111), result)
        self.trade.amount = 7.0
        self.trade.stake_amount = 70.0
        order = SimpleNamespace(
            order_id='tp1',
            ft_order_side='sell',
            ft_order_tag=result[1],
            filled=3.0,
            amount=3.0,
            status='closed',
        )
        self.strategy.order_filled(self.trade.pair, self.trade, order, self.now)
        self.assertEqual(self.trade.get_custom_data('assistant_execution', {}).get('stage', 0), 1)
        self.assertIsNone(self.adjust(111))
        self.assertAlmostEqual(self.adjust(121)[0], -35.0)

    def test_small_partial_fill_does_not_skip_remaining_tp_quantity(self):
        self.adjust(111)
        self.trade.amount = 9.0
        self.trade.stake_amount = 90.0
        order = SimpleNamespace(
            order_id='partial',
            ft_order_side='sell',
            ft_order_tag='manual_long_tp1',
            filled=1.0,
            amount=3.0,
            status='canceled',
        )
        self.strategy.order_filled(self.trade.pair, self.trade, order, self.now)
        self.assertEqual(self.trade.get_custom_data('assistant_execution', {}).get('stage', 0), 0)
        self.assertAlmostEqual(self.adjust(111)[0], -20.0)

    def test_stop_loss_and_pullback_are_not_gated_by_profit(self):
        self.assertIn(
            'sl', self.strategy.custom_exit(self.trade.pair, self.trade, self.now, 85.0, -1.5)
        )
        self.trade.set_custom_data('assistant_execution', {'stage': 1})
        self.assertIn(
            'pullback', self.strategy.custom_exit(self.trade.pair, self.trade, self.now, 95.0, -0.5)
        )
        self.trade.set_custom_data('assistant_execution', {'stage': 2})
        self.assertIn(
            'pullback', self.strategy.custom_exit(self.trade.pair, self.trade, self.now, 109.0, 0.9)
        )

    def test_absolute_stops_account_for_short_direction_and_leverage(self):
        stop = self.strategy.custom_stoploss(self.trade.pair, self.trade, self.now, 100.0, 0.0)
        self.assertAlmostEqual(stop, 1.0)
        self.trade.is_short = True
        self.trade.entry_side, self.trade.exit_side = 'sell', 'buy'
        self.strategy.manual_open[self.trade.pair].update(
            direction='short', stop_loss=110.0, exit_points=[90, 80, 70]
        )
        self.trade.data.clear()
        stop = self.strategy.custom_stoploss(self.trade.pair, self.trade, self.now, 100.0, 0.0)
        self.assertAlmostEqual(stop, 1.0)
        self.trade.set_custom_data('assistant_execution', {'stage': 2})
        stop = self.strategy.custom_stoploss(self.trade.pair, self.trade, self.now, 85.0, 1.5)
        self.assertAlmostEqual(stop, (90 / 85 - 1) * 10)

    def test_full_close_cleans_plan_only_after_confirmed_fill(self):
        self.strategy.confirm_trade_exit(
            self.trade.pair, self.trade, 'market', 10, 85, 'gtc', 'manual_sl_hit', self.now
        )
        self.assertIn(self.trade.pair, self.strategy.manual_open)
        self.trade.is_open = False
        self.trade.amount = 0
        order = SimpleNamespace(
            order_id='full',
            ft_order_side='sell',
            ft_order_tag='manual_sl_hit',
            filled=10.0,
            amount=10.0,
            status='closed',
        )
        self.strategy.order_filled(self.trade.pair, self.trade, order, self.now)
        self.assertNotIn(self.trade.pair, self.strategy.manual_open)

    def test_mismatched_plan_cannot_manage_opposite_direction(self):
        self.strategy.manual_open[self.trade.pair]['direction'] = 'short'
        self.assertIsNone(self.adjust(111))
        self.assertIsNone(
            self.strategy.custom_stoploss(self.trade.pair, self.trade, self.now, 100, 0)
        )

    def test_exchange_stop_fill_also_removes_closed_plan(self):
        self.trade.is_open = False
        self.trade.amount = 0
        order = SimpleNamespace(
            order_id='stop',
            ft_order_side='stoploss',
            ft_order_tag=None,
            filled=10.0,
            amount=10.0,
            status='closed',
        )
        self.strategy.order_filled(self.trade.pair, self.trade, order, self.now)
        self.assertNotIn(self.trade.pair, self.strategy.manual_open)

    def test_pending_take_profit_is_canceled_when_protection_is_crossed(self):
        self.strategy.dp = SimpleNamespace(_exchange=SimpleNamespace(get_rate=lambda *a, **k: 85.0))
        order = SimpleNamespace(ft_order_tag='manual_long_tp1')
        self.assertTrue(
            self.strategy.check_exit_timeout(self.trade.pair, self.trade, order, self.now)
        )
        self.assertTrue(
            self.strategy.check_entry_timeout(self.trade.pair, self.trade, order, self.now)
        )

    def test_invalid_plan_is_rejected_without_exception(self):
        for sl in (None, 'bad', float('nan'), 110):
            self.strategy.manual_open[self.trade.pair]['stop_loss'] = sl
            self.assertFalse(
                self.strategy.confirm_trade_entry(
                    self.trade.pair, 'market', 1, 100, 'gtc', self.now, 'manual_long', side='long'
                )
            )

    def test_final_target_stays_active_after_canceled_partial_fill(self):
        for count in (1, 2, 3):
            with self.subTest(targets=count):
                self.trade.data.clear()
                self.trade.amount = 10.0
                self.trade.stake_amount = 100.0
                self.strategy.manual_open[self.trade.pair]['exit_points'] = [110, 120, 130][:count]
                self.trade.set_custom_data('assistant_execution', {'stage': count - 1})
                self.adjust(140)
                self.trade.amount = 6.0
                self.trade.stake_amount = 60.0
                order = SimpleNamespace(
                    order_id=f'final{count}',
                    ft_order_side='sell',
                    ft_order_tag=f'manual_long_tp{count}',
                    filled=4.0,
                    amount=10.0,
                    status='canceled',
                )
                self.strategy.order_filled(self.trade.pair, self.trade, order, self.now)
                self.assertEqual(
                    self.trade.get_custom_data('assistant_execution', {}).get('stage', 0), count - 1
                )
                self.assertAlmostEqual(self.adjust(140)[0], -60.0)

    def test_take_profit_uses_exchange_quantity_precision(self):
        self.trade.amount = 0.011
        self.trade.stake_amount = 11.0
        result = self.adjust(111)
        self.assertAlmostEqual(abs(result[0]) * 0.011 / 11.0, 0.003)
        self.trade.amount = 0.008
        self.trade.stake_amount = 8.0
        order = SimpleNamespace(
            order_id='rounded',
            ft_order_side='sell',
            ft_order_tag=result[1],
            filled=0.003,
            amount=0.003,
            status='closed',
        )
        self.strategy.order_filled(self.trade.pair, self.trade, order, self.now)
        self.assertEqual(self.trade.get_custom_data('assistant_execution', {}).get('stage', 0), 1)

    def test_too_small_tp_tranche_closes_position_instead_of_stalling(self):
        from decimal import Decimal

        self.exchange.step = Decimal(1)
        self.exchange.markets[self.trade.pair]['limits']['amount']['min'] = 1.0
        self.exchange.markets[self.trade.pair]['info']['filters'][0].update(
            minQty='1', stepSize='1'
        )
        self.trade.amount = 1.0
        self.assertEqual(self.adjust(111)[0], -self.trade.stake_amount)

    def test_remainder_floor_switches_tp_to_durable_full_exit(self):
        self.exchange.remaining_floor = 800.0
        result = self.adjust(111)
        self.assertEqual(result[0], -100.0)
        self.assertIn('full', result[1])
        self.assertEqual(self.strategy._execution(self.trade).get('stage', 0), 0)
        # The full-close intent remains active even if price drops below TP1.
        self.assertTrue(self.strategy._execution(self.trade)['full_exit'])
        self.assertIsNone(
            self.strategy.custom_exit(self.trade.pair, self.trade, self.now, 105.0, 0.5)
        )

    def test_partial_fill_of_fallback_never_advances_tp_stage(self):
        self.exchange.remaining_floor = 800.0
        result = self.adjust(111)
        self.trade.amount = 6.0
        self.trade.stake_amount = 60.0
        order = SimpleNamespace(
            order_id='fallback',
            ft_order_side='sell',
            ft_order_tag=result[1],
            filled=4.0,
            amount=10.0,
            status='canceled',
        )
        self.trade.orders.append(order)
        self.strategy.order_filled(self.trade.pair, self.trade, order, self.now)
        self.assertEqual(self.strategy._execution(self.trade).get('stage', 0), 0)
        self.assertIn(self.trade.pair, self.strategy.manual_open)
        self.assertTrue(self.strategy._execution(self.trade)['full_exit'])
        self.assertIsNone(
            self.strategy.custom_exit(self.trade.pair, self.trade, self.now, 105.0, 0.5)
        )

    def test_failed_fallback_alerts_and_requests_stop_protection(self):
        self.assertTrue(
            hasattr(self.strategy, 'check_full_exit'), 'fallback failure observer missing'
        )
        self.exchange.remaining_floor = 800.0
        self.adjust(111)
        restored = []

        def restore(trade):
            restored.append(trade.id)
            trade.open_sl_orders = [SimpleNamespace(order_id='protected')]

        from threading import Lock

        self.strategy.execution_bot = SimpleNamespace(
            _exit_lock=Lock(), handle_stoploss_on_exchange=restore
        )
        self.trade.open_sl_orders = []
        from datetime import timedelta

        self.strategy.check_full_exit(self.trade, self.now + timedelta(seconds=5))
        self.assertEqual(restored, [1])
        self.assertTrue(any('未确认' in m or '失败' in m for m in self.messages))
        self.assertIn(self.trade.pair, self.strategy.manual_open)
        self.assertIsNone(
            self.strategy.custom_exit(self.trade.pair, self.trade, self.now, 105.0, 0.5)
        )
        count = len(self.messages)
        self.strategy.check_full_exit(self.trade, self.now + timedelta(seconds=10))
        self.assertEqual(len(self.messages), count)

    def test_full_exit_invalid_keeps_position_and_warns(self):
        self.exchange.markets[self.trade.pair]['limits']['amount']['min'] = 20.0
        self.assertIsNone(self.adjust(111))
        self.assertIn(self.trade.pair, self.strategy.manual_open)
        self.assertTrue(self.messages)

    def test_native_adjustment_submits_full_instead_of_skipping_tiny_remainder(self):
        from freqtrade.freqtradebot import FreqtradeBot

        bot = object.__new__(FreqtradeBot)
        bot.strategy = self.strategy
        bot.exchange = self.exchange
        bot.exchange.remaining_floor = 800.0
        bot.exchange.get_rates = lambda *a, **kw: (111.0, 111.0)
        bot.exchange.get_max_pair_stake_amount = lambda *a, **kw: 10000.0
        bot.wallets = SimpleNamespace(get_available_stake_amount=lambda: 10000.0)
        self.trade.calc_profit_ratio = lambda rate: (rate - 100.0) / 100.0
        submitted = []
        bot.execute_trade_exit = lambda *a, **kw: submitted.append(kw) or True
        bot.check_and_call_adjust_trade_position(self.trade)
        self.assertEqual(len(submitted), 1)
        self.assertEqual(submitted[0]['sub_trade_amt'], self.trade.amount)
        self.assertEqual(submitted[0]['exit_tag'], 'manual_minimum_full_exit')

    def test_native_stake_roundtrip_preserves_validated_minimum_quantity(self):
        from freqtrade.freqtradebot import FreqtradeBot

        bot = object.__new__(FreqtradeBot)
        bot.strategy = self.strategy
        bot.exchange = self.exchange
        self.trade.calc_profit_ratio = lambda rate: 0.1
        self.exchange.get_rates = lambda *a, **kw: (111.0, 111.0)
        self.exchange.get_max_pair_stake_amount = lambda *a, **kw: 10000.0
        bot.wallets = SimpleNamespace(get_available_stake_amount=lambda: 10000.0)
        submitted = []
        bot.execute_trade_exit = lambda *a, **kw: submitted.append(kw) or True
        for total, stake, quantity in (
            (0.017, 1.0, 0.003),
            (0.275, 3.0, 0.082),
            (0.017, 0.3, 0.001),
        ):
            with self.subTest(total=total, stake=stake, quantity=quantity):
                self.trade.amount = total
                self.trade.stake_amount = stake
                self.trade.set_custom_data('assistant_execution', {'tp_targets': {'1': quantity}})
                self.exchange.markets[self.trade.pair]['limits']['amount']['min'] = quantity
                submitted.clear()
                bot.check_and_call_adjust_trade_position(self.trade)
                self.assertEqual(len(submitted), 1)
                self.assertEqual(submitted[0]['sub_trade_amt'], quantity)

    def test_protection_that_closes_position_does_not_emit_false_failure(self):
        self.exchange.remaining_floor = 800.0
        self.adjust(111)
        before = len(self.messages)
        from threading import Lock

        def close_on_stop(trade):
            trade.is_open = False

        self.strategy.execution_bot = SimpleNamespace(
            _exit_lock=Lock(), handle_stoploss_on_exchange=close_on_stop
        )
        from datetime import timedelta

        self.strategy.check_full_exit(self.trade, self.now + timedelta(seconds=5))
        self.assertEqual(len(self.messages), before)

    def test_partial_fallback_is_retried_for_remaining_position(self):
        self.exchange.remaining_floor = 800.0
        result = self.adjust(111)
        self.trade.amount = 6.0
        self.trade.stake_amount = 60.0
        order = SimpleNamespace(
            order_id='partially-filled',
            ft_order_side='sell',
            ft_order_tag=result[1],
            filled=4.0,
            amount=10.0,
            status='canceled',
        )
        self.trade.orders.append(order)
        self.strategy.order_filled(self.trade.pair, self.trade, order, self.now)
        from datetime import timedelta

        self.strategy.check_full_exit(self.trade, self.now + timedelta(seconds=5))
        self.assertEqual(self.adjust(105)[0], -60.0)

    def test_zero_fill_cancel_after_observer_does_not_blindly_retry(self):
        self.exchange.remaining_floor = 800.0
        self.adjust(111)
        self.trade.has_open_orders = True
        from datetime import timedelta

        self.strategy.check_full_exit(self.trade, self.now + timedelta(seconds=5))
        # Native manage_open_orders may reconcile this after our observer ran.
        self.trade.has_open_orders = False
        self.assertIsNone(
            self.strategy.custom_exit(self.trade.pair, self.trade, self.now, 105.0, 0.5)
        )

    def test_fill_replay_after_interrupted_commit_is_idempotent(self):
        result = self.adjust(111)
        self.trade.amount = 7.0
        self.trade.stake_amount = 70.0
        order = SimpleNamespace(
            order_id='crash',
            ft_order_side='sell',
            ft_order_tag=result[1],
            filled=3.0,
            amount=3.0,
            status='closed',
        )
        original = self.trade.set_custom_data
        interrupted = [False]

        def commit_then_interrupt(key, value):
            original(key, value)
            if key == 'assistant_execution' and not interrupted[0]:
                interrupted[0] = True
                raise RuntimeError('interrupted after durable commit')

        self.trade.set_custom_data = commit_then_interrupt
        with self.assertRaises(RuntimeError):
            self.strategy.order_filled(self.trade.pair, self.trade, order, self.now)
        self.trade.set_custom_data = original
        self.strategy.order_filled(self.trade.pair, self.trade, order, self.now)
        state = self.trade.get_custom_data('assistant_execution')
        self.assertEqual(state['stage'], 1)
        self.assertEqual(state['tp_filled']['1'], 3.0)
        self.assertEqual(state['seen']['crash'], 3.0)

    def test_old_closed_trade_cannot_delete_new_plan(self):
        self.strategy.set_manual_plan(
            self.trade.pair, {'trade_id': 999, 'direction': 'long', 'stop_loss': 90}
        )
        self.trade.is_open = False
        self.trade.amount = 0
        order = SimpleNamespace(
            order_id='old', ft_order_side='sell', ft_order_tag='manual_long_tp3', filled=3.0
        )
        self.strategy.order_filled(self.trade.pair, self.trade, order, self.now)
        self.assertEqual(self.strategy.manual_open[self.trade.pair]['trade_id'], 999)

    def test_explicit_commands_execute_on_next_bot_loop_and_cancel_cleanly(self):
        calls = []
        future = self.strategy.submit_command(lambda: calls.append('explicit') or 123)
        canceled = self.strategy.submit_command(lambda: calls.append('canceled'))
        canceled.cancel()
        self.assertFalse(future.done())
        self.assertEqual(calls, [])
        self.strategy.bot_loop_start(self.now)
        self.assertEqual(future.result(), 123)
        self.assertEqual(calls, ['explicit'])

    def test_failed_command_does_not_block_next_explicit_command(self):
        def fail():
            raise ValueError('invalid order')

        failed = self.strategy.submit_command(fail)
        following = self.strategy.submit_command(lambda: 'next command')
        self.strategy.bot_loop_start(self.now)
        with self.assertRaisesRegex(ValueError, 'invalid order'):
            failed.result()
        self.assertEqual(following.result(), 'next command')


if __name__ == '__main__':
    unittest.main()
