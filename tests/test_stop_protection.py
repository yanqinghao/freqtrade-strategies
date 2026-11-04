from notification_helpers import capture_alerts
import importlib.util
import sys
import tempfile
import unittest
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path
from types import SimpleNamespace

from test_manual_strategy import Position
from test_exit_policy import RuleExchange

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'user_data' / 'strategies'))


class StopProtectionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            importlib.util.find_spec('stop_protection'), 'stop confirmation missing'
        )
        import stop_protection
        from KamaFama_Dynamic import KamaFama_Dynamic
        from freqtrade.enums import RunMode
        from freqtrade.exchange import ROUND_UP

        self.module = stop_protection
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.now = datetime(2026, 11, 3, 12, tzinfo=timezone.utc)
        self.strategy = KamaFama_Dynamic(
            {
                'runmode': RunMode.LIVE,
                'dry_run': False,
                'user_data_dir': Path(self.temp.name),
                'stake_currency': 'USDT',
                'exchange': {'name': 'binance'},
            }
        )
        self.trade = Position(
            id=1,
            pair='BTC/USDT:USDT',
            is_open=True,
            is_short=False,
            amount=10.0,
            stake_amount=1000.0 / 3,
            leverage=3.0,
            open_rate=100.0,
            fee_open=0.0005,
            fee_close=0.0005,
            open_date=self.now - timedelta(hours=1),
            entry_side='buy',
            exit_side='sell',
            orders=[],
            data={},
            open_sl_orders=[SimpleNamespace(order_id='stop-1')],
        )
        self.plan = {
            'trade_id': 1,
            'direction': 'long',
            'stop_loss': 90.0,
            'exit_points': [110.0, 120.0, 130.0],
        }
        self.strategy.set_manual_plan(self.trade.pair, self.plan)
        self.exchange = RuleExchange(minimum=0.001, step='.001', cost=0.0, pair=self.trade.pair)
        self.exchange.markets[self.trade.pair]['info']['filters'].append(
            {'filterType': 'PRICE_FILTER', 'tickSize': '.1'}
        )

        def price(pair, value, rounding_mode=None):
            rounded = (Decimal(str(value)) / Decimal('.1')).to_integral_value(
                rounding=ROUND_CEILING if rounding_mode == ROUND_UP else ROUND_FLOOR
            ) * Decimal('.1')
            return float(rounded)

        self.exchange.price_to_precision = price
        self.raw = {
            'id': 'stop-1',
            'symbol': self.trade.pair,
            'side': 'sell',
            'type': 'stop_market',
            'status': 'open',
            'amount': 10.0,
            'filled': 0.0,
            'remaining': 10.0,
            'stopPrice': 90.0,
            'reduceOnly': True,
            'info': {'positionSide': 'BOTH', 'reduceOnly': True},
        }
        self.queries = []
        self.messages = []
        capture_alerts(self.strategy, self.messages)
        self.exchange.fetch_stoploss_order = lambda ident, pair: self.queries.append(
            ident
        ) or deepcopy(self.raw)
        self.strategy.dp = SimpleNamespace(
            _exchange=self.exchange, send_msg=lambda text, **kw: self.messages.append(text)
        )

    def check(self):
        return self.module.check_protection(self.strategy, self.trade, self.now, force=True)

    def test_same_incident_reason_changes_do_not_spam_and_recovery_is_once(self):
        self.raw['status'] = 'canceled'
        self.assertFalse(self.check())
        self.raw.update(status='open', amount=9, remaining=9)
        self.assertFalse(self.check())
        self.assertEqual(len(self.messages), 1)
        self.raw.update(amount=10, remaining=10)
        self.assertTrue(self.check())
        self.raw['stopPrice'] = 91
        self.assertTrue(self.check())
        self.assertEqual(len(self.messages), 2)

    def test_queue_storage_failure_does_not_consume_notification(self):
        from unittest.mock import patch

        self.raw['status'] = 'canceled'
        with patch.object(self.strategy.alert_queue, 'put', side_effect=OSError('disk full')):
            self.assertFalse(self.check())
        self.assertFalse(self.check())
        self.assertEqual(len(self.strategy.alert_queue.pending()), 1)

    def test_recovery_replaces_unsaved_failure_instead_of_replaying_stale_warning(self):
        from unittest.mock import patch

        self.assertTrue(self.check())
        self.strategy.alert_queue.acknowledge(self.strategy.alert_queue.pending()[0])
        self.raw['status'] = 'canceled'
        with patch.object(self.strategy.alert_queue, 'put', side_effect=OSError('disk full')):
            self.assertFalse(self.check())
        self.raw['status'] = 'open'
        self.assertTrue(self.check())
        rows = self.strategy.alert_queue.pending()
        self.assertEqual(len(rows), 1)
        self.assertIn('止损已确认', rows[0]['message'])
        self.assertEqual(self.strategy._unsaved_alerts, {})

    def test_active_full_reduce_only_stop_confirms_actual_receipt(self):
        self.assertTrue(self.check())
        status = self.strategy._execution(self.trade)['protection']
        self.assertEqual(status['status'], 'confirmed')
        self.assertEqual(status['stop_price'], 90.0)
        self.assertEqual(status['covered_amount'], 10.0)
        self.assertEqual(self.queries, ['stop-1'])
        self.assertTrue(any('止损已确认' in message for message in self.messages))

    def test_missing_cancelled_triggered_and_wrong_receipts_never_confirm(self):
        cases = (
            {'status': 'canceled'},
            {'status': 'closed'},
            {'status_stop': 'triggered'},
            {'side': 'buy'},
            {'symbol': 'ETH/USDT:USDT'},
            {'id': 'other'},
            {'stopPrice': None},
            {'stopPrice': float('nan')},
            {'type': 'take_profit_market'},
            {'amount': float('nan')},
            {'remaining': None, 'filled': None},
        )
        original = deepcopy(self.raw)
        for change in cases:
            with self.subTest(change=change):
                self.raw = {**deepcopy(original), **change}
                self.assertFalse(self.check())
        self.trade.open_sl_orders = []
        self.assertFalse(self.check())

    def test_partially_executing_stop_and_unrepresentable_price_do_not_confirm(self):
        original = deepcopy(self.raw)
        for change in (
            {'amount': 12.0, 'filled': 1.0, 'remaining': 11.0},
            {'stopPrice': '1e1000'},
            {'info': []},
        ):
            with self.subTest(change=change):
                self.raw = {**deepcopy(original), **change}
                self.assertFalse(self.check())

    def test_closing_flag_must_be_true_and_consistent(self):
        self.raw['reduceOnly'] = False
        self.raw['info']['reduceOnly'] = False
        self.assertFalse(self.check())
        self.raw['reduceOnly'] = 'false'
        self.raw['info']['reduceOnly'] = 'true'
        self.assertFalse(self.check())
        self.raw['reduceOnly'] = 'true'
        self.raw['info']['reduceOnly'] = 'true'
        self.assertTrue(self.check())

    def test_close_position_order_covers_whole_position_without_quantity(self):
        self.raw.update(amount=0.0, remaining=0.0, reduceOnly=False)
        self.raw['info'].update(reduceOnly=False, closePosition=True)
        self.assertTrue(self.check())

    def test_insufficient_coverage_rejects_but_oversize_reduce_only_is_valid(self):
        self.raw.update(amount=9.999, remaining=9.999)
        self.assertFalse(self.check())
        self.raw.update(amount=12.0, remaining=12.0)
        self.assertTrue(self.check())

    def test_contract_conversion_is_not_applied_twice(self):
        self.exchange.markets[self.trade.pair]['contractSize'] = 0.01
        self.trade.amount = 0.03
        self.raw.update(amount=0.03, remaining=0.03)
        self.assertTrue(self.check())
        self.raw.update(amount=0.02, remaining=0.02)
        self.assertFalse(self.check())

    def test_long_price_must_be_no_looser_than_rounded_protection(self):
        self.plan['stop_loss'] = 90.04
        self.strategy.set_manual_plan(self.trade.pair, self.plan)
        self.raw['stopPrice'] = 90.0
        self.assertFalse(self.check())
        self.raw['stopPrice'] = 90.1
        self.assertTrue(self.check())
        self.raw['stopPrice'] = 91.0
        self.assertTrue(self.check())

    def test_short_price_and_exit_side_are_reversed(self):
        self.trade.is_short = True
        self.trade.entry_side = 'sell'
        self.trade.exit_side = 'buy'
        self.plan.update(direction='short', stop_loss=110.06, exit_points=[90.0, 80.0, 70.0])
        self.strategy.set_manual_plan(self.trade.pair, self.plan)
        self.raw.update(side='buy', stopPrice=110.1)
        self.assertFalse(self.check())
        self.raw['stopPrice'] = 110.0
        self.assertTrue(self.check())
        self.raw['stopPrice'] = 109.0
        self.assertTrue(self.check())

    def test_query_failure_never_confirms_and_does_not_leak_exception(self):
        def fail(*args):
            raise TimeoutError('SECRET_SIGNED_URL')

        self.exchange.fetch_stoploss_order = fail
        self.assertFalse(self.check())
        self.assertNotIn('SECRET_SIGNED_URL', ' '.join(self.messages))
        self.assertNotEqual(
            self.strategy._execution(self.trade)['protection']['status'], 'confirmed'
        )

    def test_fetch_value_error_never_persists_or_sends_private_text(self):
        def fail(*args):
            raise ValueError('SECRET_SIGNED_URL')

        self.exchange.fetch_stoploss_order = fail
        self.assertFalse(self.check())
        self.assertNotIn('SECRET_SIGNED_URL', repr(self.strategy._execution(self.trade)))
        self.assertNotIn('SECRET_SIGNED_URL', ' '.join(self.messages))

    def test_inconsistent_quantity_fields_never_confirm_or_allow_add(self):
        for change in (
            {'amount': 1.0, 'remaining': 10.0},
            {'amount': 10.0, 'remaining': 11.0},
            {'amount': 11.0, 'remaining': 10.0},
        ):
            with self.subTest(change=change):
                self.raw.update(change)
                self.assertFalse(self.check())
                with self.assertRaisesRegex(ValueError, '止损尚未确认'):
                    self.strategy.ensure_entry_protection(self.trade)

    def test_initial_grace_blocks_add_without_alarming_then_recovers_once(self):
        self.trade.open_date = self.now
        self.trade.open_sl_orders = []
        self.assertFalse(self.check())
        self.assertEqual(self.messages, [])
        self.now += timedelta(seconds=16)
        self.assertFalse(self.check())
        self.assertEqual(len(self.messages), 1)
        self.now += timedelta(seconds=16)
        self.assertFalse(self.check())
        self.assertEqual(len(self.messages), 1)
        self.trade.open_sl_orders = [SimpleNamespace(order_id='stop-1')]
        self.assertTrue(self.check())
        self.assertEqual(len(self.messages), 2)

    def test_persisted_status_deduplicates_after_strategy_restart(self):
        self.raw['status'] = 'canceled'
        self.assertFalse(self.check())
        self.assertEqual(len(self.messages), 1)
        self.strategy = type(self.strategy)(self.strategy.config)
        capture_alerts(self.strategy, self.messages)
        self.strategy.dp = SimpleNamespace(
            _exchange=self.exchange, send_msg=lambda text, **kw: self.messages.append(text)
        )
        self.now += timedelta(seconds=60)
        self.assertFalse(self.check())
        self.assertEqual(len(self.messages), 1)
        self.raw['status'] = 'open'
        self.assertTrue(self.check())
        self.assertEqual(len(self.messages), 2)

    def test_periodic_cadence_is_bypassed_by_explicit_add_check(self):
        self.assertTrue(self.check())
        self.raw['status'] = 'canceled'
        self.now += timedelta(seconds=1)
        self.module.check_protection(self.strategy, self.trade, self.now)
        self.assertEqual(len(self.queries), 1)
        self.assertFalse(self.check())
        self.assertEqual(len(self.queries), 2)

    def test_simulation_never_claims_exchange_confirmation_or_queries_live(self):
        self.strategy.config['dry_run'] = True
        self.assertTrue(self.check())
        self.assertEqual(self.queries, [])
        self.assertEqual(self.strategy._execution(self.trade)['protection']['status'], 'simulated')

    def test_closed_and_unfilled_positions_do_not_emit_protection_claims(self):
        self.trade.amount = 0.0
        self.assertFalse(self.check())
        self.assertEqual(self.messages, [])
        self.trade.amount = 10.0
        self.trade.is_open = False
        self.assertFalse(self.check())
        self.assertEqual(self.messages, [])
