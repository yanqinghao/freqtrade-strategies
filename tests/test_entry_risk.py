import asyncio
import importlib.util
import json
import unittest
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import test_telegram_orders as order_tests


class EntryRiskTests(unittest.TestCase):
    setUp = order_tests.TelegramOrderTests.setUp
    position = order_tests.TelegramOrderTests.position

    def prepare(self):
        self.assertIsNotNone(importlib.util.find_spec('entry_risk'), 'entry risk engine missing')
        import entry_risk

        self.risk = entry_risk
        self.path = Path(self.strategy.state_file).with_name('assistant_risk.json')
        self.limits = {
            'max_margin_usdt': 100,
            'max_leverage': 20,
            'max_total_notional_usdt': 1000,
            'max_stop_loss_usdt': 10,
        }
        self.path.write_text(json.dumps(self.limits))
        self.exchange.markets['BTC/USDT:USDT']['taker'] = 0.0005
        self.exchange.get_fee = lambda *a, **kw: 0.0005
        self.request = self.module.parse_entry(['BTC', '10', '20', 'sl=97'], 'long')

    def assess(self, request=None, trade=None):
        request = request or self.request
        plan = self.risk.plan_for_entry(self.strategy, request, trade)
        return self.risk.assess_entry(self.strategy, request, trade, plan)

    def order(self, trade, *, amount, filled, opened=True, side='buy', price=100):
        from freqtrade.persistence import Order

        trade.orders.append(
            Order(
                order_id='order-' + str(len(trade.orders)),
                ft_order_side=side,
                ft_is_open=opened,
                ft_pair=trade.pair,
                order_type='limit',
                status='open' if opened else 'canceled',
                symbol=trade.pair,
                side=side,
                amount=amount,
                filled=filled,
                remaining=amount - filled,
                ft_amount=amount,
                ft_price=price,
                price=price,
                average=price,
                order_date=datetime.now(timezone.utc),
                ft_order_tag='manual_long|ttl=86400',
            )
        )
        self.Trade.commit()

    def test_user_example_long_and_short_include_entry_and_exit_fees(self):
        self.prepare()
        long = self.assess()
        self.assertEqual(long['notional'], 200)
        self.assertEqual(long['amount'], 2)
        self.assertEqual(long['stop_distance_pct'], 3)
        self.assertAlmostEqual(long['price_loss'], 6)
        self.assertAlmostEqual(long['fees'], 0.197)
        self.assertAlmostEqual(long['estimated_loss'], 6.197)
        self.assertTrue(long['allowed'])
        short = self.assess({**self.request, 'direction': 'short', 'stop_loss': 103})
        self.assertAlmostEqual(short['estimated_loss'], 6.203)
        self.assertTrue(short['allowed'])

    def test_100_margin_3x_passes_but_20x_breaches_notional_and_loss(self):
        self.prepare()
        lower = self.assess({**self.request, 'size': 100, 'leverage': 3})
        self.assertAlmostEqual(lower['estimated_loss'], 9.2955)
        self.assertTrue(lower['allowed'])
        higher = self.assess({**self.request, 'size': 100, 'leverage': 20})
        self.assertFalse(higher['allowed'])
        self.assertTrue(any('敞口' in r for r in higher['reasons']))
        self.assertTrue(any('亏损' in r for r in higher['reasons']))

    def test_caps_are_inclusive_and_hot_loaded_without_resizing(self):
        self.prepare()
        self.path.write_text(json.dumps({**self.limits, 'max_total_notional_usdt': 200}))
        self.assertTrue(self.assess()['allowed'])
        self.path.write_text(json.dumps({**self.limits, 'max_total_notional_usdt': 199.99}))
        result = self.assess()
        self.assertFalse(result['allowed'])
        self.assertEqual(result['amount'], 2)
        self.assertFalse(self.assess({**self.request, 'size': 100.01, 'leverage': 1})['allowed'])
        self.assertFalse(self.assess({**self.request, 'size': 0.1, 'leverage': 20.01})['allowed'])

    def test_missing_limits_are_explicit_and_bad_present_limits_fail_closed(self):
        self.prepare()
        self.path.unlink()
        self.assertIsNone(self.risk.load_limits(self.path))
        self.assertIsNone(self.assess()['limits'])
        for change in (
            {'max_margin_usdt': True},
            {'max_leverage': float('inf')},
            {'max_stop_loss_usdt': 0},
            {'max_total_notional_usdt': -1},
            {'unknown_limit': 10},
        ):
            self.path.write_text(json.dumps({**self.limits, **change}))
            with self.assertRaises(self.risk.EntryRiskError):
                self.assess()
        self.path.write_text('{}')
        with self.assertRaises(self.risk.EntryRiskError):
            self.assess()

    def test_partial_fill_is_counted_once_before_and_after_native_accounting(self):
        self.prepare()
        trade = self.position()
        trade.amount = 0
        self.order(trade, amount=4, filled=1)
        snap = self.risk.estimate_exposure(self.exchange, [trade])
        self.assertEqual(snap['position_notional'], 100)
        self.assertEqual(snap['pending_notional'], 300)
        self.assertEqual(snap['total_notional'], 400)
        trade.amount = 1
        self.assertEqual(self.risk.estimate_exposure(self.exchange, [trade])['total_notional'], 400)

    def test_terminal_positive_receipt_before_quantity_reconciliation_is_not_lost(self):
        self.prepare()
        trade = self.position()
        trade.amount = 0
        self.order(trade, amount=4, filled=1, opened=False)
        snap = self.risk.estimate_exposure(self.exchange, [trade])
        self.assertEqual(snap['total_notional'], 100)
        self.assertEqual(snap['pending_notional'], 0)

    def test_normalized_contract_quantities_are_not_multiplied_twice(self):
        self.prepare()
        self.exchange.markets[self.request['pair']]['contractSize'] = 0.01
        self.exchange.contract_size = 0.01
        trade = self.position()
        trade.amount = 0.01
        self.order(trade, amount=0.03, filled=0.01)
        self.assertAlmostEqual(
            self.risk.estimate_exposure(self.exchange, [trade])['total_notional'], 3
        )
        self.assertAlmostEqual(self.assess()['notional'], 200)

    def test_new_add_checks_whole_position_risk_not_only_this_order(self):
        self.prepare()
        trade = self.position()
        trade.amount = 2
        trade.leverage = 20
        trade.stake_amount = 10
        self.Trade.commit()
        plan = {'direction': 'long', 'trade_id': trade.id, 'stop_loss': 97, 'exit_points': []}
        self.strategy.set_manual_plan(trade.pair, plan)
        result = self.assess(trade=trade)
        self.assertAlmostEqual(result['estimated_loss'], 12.394)
        self.assertFalse(result['allowed'])

    def test_unknown_cross_pair_request_blocks_reliable_total_exposure(self):
        self.prepare()
        store = self.strategy.entry_requests
        store.reserve(
            'a' * 24, {'pair': 'ETH/USDT:USDT', 'direction': 'long', 'size': 10, 'leverage': 20}
        )
        store.set_status('a' * 24, 'unknown', {})
        result = self.assess()
        self.assertFalse(result['allowed'])
        self.assertTrue(any('待核实' in r for r in result['reasons']))

    def test_invalid_price_fee_and_market_data_do_not_become_zero_risk(self):
        self.prepare()

        def fail(*a, **kw):
            raise ValueError('SECRET_SIGNED_URL')

        self.exchange.get_fee = fail
        with self.assertRaises(self.risk.EntryRiskError) as caught:
            self.assess()
        self.assertNotIn('SECRET_SIGNED_URL', str(caught.exception))
        self.exchange.get_fee = lambda *a, **kw: None
        with self.assertRaises(self.risk.EntryRiskError):
            self.assess()
        self.exchange.get_fee = lambda *a, **kw: 0.0005
        self.exchange.get_rate = lambda *a, **kw: float('nan')
        with self.assertRaises(self.risk.EntryRiskError):
            self.assess()

    def test_preview_is_read_only_for_plans_tp_accounting_and_requests(self):
        self.prepare()
        trade = self.position()
        trade.amount = 2
        trade.leverage = 20
        plan = {
            'direction': 'long',
            'trade_id': trade.id,
            'stop_loss': 95,
            'exit_points': [120, 130],
        }
        self.strategy.set_manual_plan(trade.pair, plan)
        self.strategy._save_execution(
            trade, {'stage': 1, 'protected_stop': 100.1, 'seen': {'x': 1}}
        )
        before = deepcopy(self.strategy._execution(trade))
        content = Path(self.strategy.state_file).read_bytes()
        request = {**self.request, 'entry_price': 110, 'exit_points': [125, 135]}
        result = self.bot._preview_entry(request)
        self.assertIn('estimated_loss', result)
        self.assertEqual(Path(self.strategy.state_file).read_bytes(), content)
        self.assertEqual(self.strategy._execution(trade), before)
        self.assertFalse(self.strategy.entry_requests.path.exists())
        self.assertEqual(self.Trade.get_open_trade_count(), 1)

    def test_telegram_preview_uses_same_parameters_without_native_submission(self):
        self.prepare()
        sent = []

        async def send(text, **kwargs):
            sent.append(text)

        async def execute(function, *args):
            return function(*args)

        self.bot._send_msg, self.bot._await_command = send, execute
        self.rpc._rpc_force_entry = lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError('preview traded')
        )
        context = SimpleNamespace(args=['long', 'BTC', '10', '20', 'sl=97'])
        asyncio.run(self.bot._preview.__wrapped__(self.bot, None, context))
        self.assertEqual(len(sent), 1)
        self.assertIn('未下单', sent[0])
        self.assertIn('6.197', sent[0])
        self.assertFalse(self.strategy.entry_requests.path.exists())
        self.assertFalse(Path(self.strategy.state_file).exists())
