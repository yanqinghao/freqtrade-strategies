import asyncio
import importlib.util
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import test_telegram_orders as order_tests


class EntryRequestStoreTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('entry_requests'), 'durable requests missing')
        from entry_requests import EntryRequests

        self.cls = EntryRequests
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'requests.sqlite'
        self.store = EntryRequests(self.path)
        self.ref = 'a' * 24
        self.request = {'pair': 'BTC/USDT:USDT', 'direction': 'long', 'size': 100.0}

    def test_reservation_survives_restart_and_only_first_claim_is_new(self):
        row, fresh = self.store.reserve(self.ref, self.request)
        self.assertTrue(fresh)
        self.assertEqual(row['state'], 'reserved')
        row, fresh = self.cls(self.path).reserve(self.ref, self.request)
        self.assertFalse(fresh)
        self.assertEqual(row['request'], self.request)

    def test_same_reference_cannot_be_changed_or_reset(self):
        self.store.reserve(self.ref, self.request)
        self.store.set_status(self.ref, 'submitted', {'id': 7, 'order_ids': ['o1']})
        with self.assertRaises(ValueError):
            self.store.reserve(self.ref, {**self.request, 'size': 200.0})
        self.assertEqual(self.store.get(self.ref)['result']['id'], 7)

    def test_unresolved_is_per_pair_and_explicit_resolution_retains_history(self):
        self.store.reserve(self.ref, self.request)
        self.store.set_status(self.ref, 'unknown', {'message': '待核实'})
        self.assertEqual(len(self.store.unresolved(self.request['pair'])), 1)
        self.assertEqual(self.store.unresolved('ETH/USDT:USDT'), [])
        self.store.set_status(self.ref, 'resolved', {'message': '用户核实解除'})
        self.assertEqual(self.store.unresolved(self.request['pair']), [])
        self.assertFalse(self.store.reserve(self.ref, self.request)[1])
        self.assertEqual(self.store.recent()[0]['state'], 'resolved')

    def test_storage_corruption_is_not_reset_to_empty(self):
        self.path.write_bytes(b'not a sqlite database')
        with self.assertRaises(sqlite3.DatabaseError):
            self.store.reserve(self.ref, self.request)

    def test_identity_tracks_message_not_text_or_delivery_sequence(self):
        self.assertEqual(self.cls.reference(-12, 40), self.cls.reference(-12, 40))
        self.assertNotEqual(self.cls.reference(-12, 40), self.cls.reference(-12, 41))
        self.assertNotEqual(self.cls.reference(-12, 40), self.cls.reference(-13, 40))


class EntryRequestFlowTests(unittest.TestCase):
    setUp = order_tests.TelegramOrderTests.setUp
    position = order_tests.TelegramOrderTests.position

    def prepare(self):
        self.assertTrue(hasattr(self.bot, '_execute_entry_request'), 'request dispatch missing')
        self.ref, self.other = 'a' * 24, 'b' * 24
        self.calls = []

        def create(pair, price, **kwargs):
            from freqtrade.persistence import Order

            self.calls.append(kwargs)
            trade = self.Trade.get_trades([self.Trade.is_open.is_(True)]).first() or self.position()
            trade.orders.append(
                Order(
                    order_id=f'order-{len(self.calls)}',
                    ft_order_side='buy',
                    ft_is_open=False,
                    ft_pair=pair,
                    order_type='market',
                    status='closed',
                    symbol=pair,
                    side='buy',
                    amount=1.0,
                    filled=1.0,
                    remaining=0.0,
                    price=100.0,
                    ft_amount=1.0,
                    ft_price=100.0,
                    ft_order_tag=kwargs['enter_tag'],
                )
            )
            self.Trade.commit()
            return trade

        self.rpc._rpc_force_entry = create
        return self.strategy.entry_requests

    def test_telegram_replay_uses_message_identity_and_query_does_not_submit(self):
        self.prepare()
        from freqtrade.enums import SignalDirection

        sent = []

        async def send(text, **kw):
            sent.append(text)

        async def execute(function, *args):
            return function(*args)

        self.bot._manual_order_lock = asyncio.Lock()
        self.bot._send_msg, self.bot._await_command = send, execute
        context = SimpleNamespace(args=['BTC', '100', '3', 'sl=90'])

        async def scenario():
            for delivery, message in ((1, 77), (2, 77), (3, 78)):
                update = SimpleNamespace(
                    update_id=delivery,
                    effective_chat=SimpleNamespace(id=-12),
                    effective_message=SimpleNamespace(message_id=message),
                )
                await self.bot._force_enter.__wrapped__(
                    self.bot, update, context, SignalDirection.LONG
                )
            await self.bot._request.__wrapped__(self.bot, None, SimpleNamespace(args=[]))

        asyncio.run(scenario())
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(len(sent), 6)
        self.assertIn('/request <请求编号>', sent[3])
        self.assertIn(self.strategy.entry_requests.reference(-12, 77), sent[0])
        self.assertIn(self.strategy.entry_requests.reference(-12, 77), sent[1])

    def test_receipt_reports_actual_native_leverage(self):
        self.prepare()
        original = self.rpc._rpc_force_entry

        def adjusted(*args, **kwargs):
            trade = original(*args, **kwargs)
            trade.leverage = 2.0
            self.Trade.commit()
            return trade

        self.rpc._rpc_force_entry = adjusted
        row = self.bot._execute_entry_request(self.request, self.ref)
        self.assertIn('实际杠杆 2x', self.bot._format_entry_request(row))

    def test_result_storage_failure_recovers_native_order_without_resend(self):
        store = self.prepare()
        original = store.set_status

        def fail(ref, state, result):
            if state == 'submitted':
                raise OSError('disk full')
            return original(ref, state, result)

        with patch.object(store, 'set_status', side_effect=fail):
            with self.assertRaises(OSError):
                self.bot._execute_entry_request(self.request, self.ref)
        self.assertEqual(self.bot._entry_request_status(self.ref)['state'], 'submitted')
        self.bot._execute_entry_request(self.request, self.ref)
        self.assertEqual(len(self.calls), 1)

    def test_dispatch_marker_storage_failure_leaves_plan_and_exchange_untouched(self):
        store = self.prepare()
        original = store.set_status

        def fail(ref, state, result):
            if state == 'submitting':
                raise OSError('disk full')
            return original(ref, state, result)

        with patch.object(store, 'set_status', side_effect=fail):
            row = self.bot._execute_entry_request(self.request, self.ref)
        self.assertEqual(row['state'], 'rejected')
        self.assertEqual(self.strategy.manual_open, {})
        self.assertEqual(self.calls, [])

    def test_same_request_is_cached_across_strategy_restart_but_new_message_can_add(self):
        store = self.prepare()
        first = self.bot._execute_entry_request(self.request, self.ref)
        self.assertEqual(first['state'], 'submitted')
        self.strategy = type(self.strategy)(self.strategy.config)
        self.strategy.dp = SimpleNamespace(_exchange=self.exchange, send_msg=lambda *a, **kw: None)
        self.rpc._freqtrade.strategy = self.strategy
        again = self.bot._execute_entry_request(self.request, self.ref)
        self.assertEqual(first['result'], again['result'])
        self.assertEqual(len(self.calls), 1)
        self.bot._execute_entry_request(self.request, self.other)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(store.get(self.ref)['result']['order_ids'], ['order-1'])

    def test_paused_native_rejection_preserves_existing_plan_and_tp_state(self):
        self.prepare()
        from freqtrade.enums import SignalDirection, State
        from freqtrade.rpc import RPC

        trade = self.position()
        prior = {
            'trade_id': trade.id,
            'direction': 'long',
            'stop_loss': 80.0,
            'exit_points': [150.0],
        }
        self.strategy.set_manual_plan(trade.pair, prior)
        self.strategy._save_execution(trade, {'stage': 1, 'protected_stop': 100.0})
        before = self.strategy._execution(trade)
        self.rpc._freqtrade.state = State.PAUSED
        self.rpc._freqtrade.config = {'force_entry_enable': True}
        self.rpc._force_entry_validations = lambda pair, side: RPC._force_entry_validations(
            self.rpc, pair, side
        )

        def native_reject(*args, **kwargs):
            self.calls.append(kwargs)
            self.rpc._force_entry_validations(trade.pair, SignalDirection.LONG)

        self.rpc._rpc_force_entry = native_reject
        request = {**self.request, 'stop_loss': 95.0, 'exit_points': [110.0]}
        row = self.bot._execute_entry_request(request, self.ref)
        self.assertEqual(row['state'], 'rejected')
        self.assertEqual(self.strategy.manual_open[trade.pair], prior)
        self.assertEqual(self.strategy._execution(trade), before)
        self.assertEqual(self.calls, [])
        self.bot._execute_entry_request(request, self.ref)
        self.assertEqual(self.calls, [])

    def test_pause_during_account_checks_cannot_install_new_plan(self):
        self.prepare()
        from freqtrade.enums import SignalDirection, State
        from freqtrade.rpc import RPC

        trade = self.position()
        prior = {
            'trade_id': trade.id,
            'direction': 'long',
            'stop_loss': 80.0,
            'exit_points': [150.0],
        }
        self.strategy.set_manual_plan(trade.pair, prior)
        self.rpc._freqtrade.config = {'force_entry_enable': True}

        def eligibility(pair, side):
            if self.rpc._freqtrade.state != State.RUNNING:
                RPC._force_entry_validations(self.rpc, pair, side)

        self.rpc._force_entry_validations = eligibility
        original = self.strategy.ensure_entry_protection

        def pause(trade):
            original(trade)
            self.rpc._freqtrade.state = State.PAUSED

        self.strategy.ensure_entry_protection = pause

        def native_reject(*args, **kwargs):
            self.calls.append(kwargs)
            eligibility(trade.pair, SignalDirection.LONG)

        self.rpc._rpc_force_entry = native_reject
        row = self.bot._execute_entry_request(
            {**self.request, 'stop_loss': 95.0, 'exit_points': [110.0]}, self.ref
        )
        self.assertEqual(row['state'], 'rejected')
        self.assertEqual(self.strategy.manual_open[trade.pair], prior)
        self.assertEqual(self.calls, [])

    def test_rejected_preflight_is_cached_and_does_not_change_plan(self):
        store = self.prepare()
        bad = {**self.request, 'stop_loss': 101.0}
        row = self.bot._execute_entry_request(bad, self.ref)
        self.assertEqual(row['state'], 'rejected')
        self.assertEqual(self.strategy.manual_open, {})
        self.bot._execute_entry_request(bad, self.ref)
        self.assertEqual(self.calls, [])
        with self.assertRaises(ValueError):
            self.bot._execute_entry_request(self.request, self.ref)
        self.assertEqual(store.get(self.ref)['state'], 'rejected')

    def test_uncertain_submission_blocks_new_intents_for_same_pair_and_never_retries(self):
        store = self.prepare()

        def fail(*a, **kw):
            self.calls.append(kw)
            raise TimeoutError('SECRET_SIGNED_URL')

        self.rpc._rpc_force_entry = fail
        row = self.bot._execute_entry_request(self.request, self.ref)
        self.assertEqual(row['state'], 'unknown')
        self.assertIn(self.request['pair'], self.strategy.manual_open)
        self.bot._execute_entry_request(self.request, self.ref)
        blocked = self.bot._execute_entry_request(self.request, self.other)
        self.assertEqual(blocked['state'], 'rejected')
        self.assertEqual(len(self.calls), 1)
        self.assertNotIn('SECRET_SIGNED_URL', repr(store.recent()))

    def test_native_order_tag_recovers_crash_after_persistence(self):
        store = self.prepare()

        def fail_after_accept(*a, **kw):
            original(*a, **kw)
            raise TimeoutError('failed after native commit')

        original = self.rpc._rpc_force_entry
        self.rpc._rpc_force_entry = fail_after_accept
        self.bot._execute_entry_request(self.request, self.ref)
        self.assertEqual(self.bot._entry_request_status(self.ref)['state'], 'submitted')
        self.bot._execute_entry_request(self.request, self.ref)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(store.get(self.ref)['result']['order_ids'], ['order-1'])

    def test_submitting_without_native_record_stays_unknown_and_reserved_is_not_replayed(self):
        store = self.prepare()
        store.reserve(self.ref, self.request)
        store.set_status(self.ref, 'submitting', {})
        self.assertEqual(self.bot._entry_request_status(self.ref)['state'], 'unknown')
        store.reserve(self.other, self.request)
        self.assertEqual(self.bot._entry_request_status(self.other)['state'], 'rejected')
        self.bot._execute_entry_request(self.request, self.other)
        self.assertEqual(self.calls, [])

    def test_storage_failure_before_reservation_cannot_submit(self):
        store = self.prepare()
        with patch.object(store, 'reserve', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                self.bot._execute_entry_request(self.request, self.ref)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.strategy.manual_open, {})

    def test_explicit_resolution_requires_fresh_account_checks_and_never_resends(self):
        store = self.prepare()
        store.reserve(self.ref, self.request)
        store.set_status(self.ref, 'unknown', {})
        self.strategy.config['dry_run'] = False
        self.exchange.fetch_positions = lambda pair: []
        self.exchange._api.fetch_open_orders = lambda pair, **kw: [
            {
                'id': 'outside',
                'symbol': pair,
                'status': 'open',
                'side': 'buy',
                'reduceOnly': False,
                'info': {},
            }
        ]
        with self.assertRaises(ValueError):
            self.bot._resolve_entry_request(self.ref)
        self.assertEqual(store.get(self.ref)['state'], 'unknown')
        self.exchange._api.fetch_open_orders = lambda *a, **kw: []
        self.assertEqual(self.bot._resolve_entry_request(self.ref)['state'], 'resolved')
        self.bot._execute_entry_request(self.request, self.ref)
        self.assertEqual(self.calls, [])

    def test_request_tag_preserves_per_order_ttl(self):
        self.prepare()
        from entry_lifecycle import entry_ttl

        request = self.module.parse_entry(
            ['BTC', '100', '3', 'sl=90', 'price=100', 'ttl=4h'], 'long'
        )
        self.bot._execute_entry_request(request, self.ref)
        tag = self.calls[0]['enter_tag']
        self.assertIn('|req=' + self.ref, tag)
        self.assertEqual(entry_ttl(SimpleNamespace(ft_order_tag=tag)), 14400)
        self.assertIsNone(entry_ttl(SimpleNamespace(ft_order_tag='manual_long|req=' + self.ref)))
