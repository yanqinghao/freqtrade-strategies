import unittest
import time
from types import SimpleNamespace
import test_telegram_orders as order_tests
from test_volume_pause_policy import rows


class PauseIntegrationTests(unittest.TestCase):
    setUp = order_tests.TelegramOrderTests.setUp
    position = order_tests.TelegramOrderTests.position

    def prepare(self):
        from entry_pause import EntryPause

        self.controller = EntryPause(
            self.strategy, self.rpc._freqtrade, self.bot._execute_entry_request
        )
        self.strategy.entry_pause = self.controller
        self.ref = 'a' * 24
        self.request.update(entry_price=100.0, ttl_seconds=14400)
        now = time.time()
        volume = rows(now, 300, 28, 105)
        volume[-1][4:] = [104.0, 40.0]
        self.controller.snapshots[self.request['pair']] = dict(
            observed_at=now,
            minute=rows(now, 60, 10, 101),
            volume=volume,
            continuous_since=now - 1000,
            bid=101,
            ask=101,
        )
        self.calls = []
        self.rpc._rpc_force_entry = lambda *a, **kw: self.calls.append(kw)

    def test_first_request_during_shock_is_retained_without_outbound_order(self):
        self.prepare()
        row = self.bot._execute_entry_request(self.request, self.ref)
        self.assertEqual(self.calls, [])
        meta = self.strategy.entry_requests.get_pause(self.ref)
        self.assertIsNotNone(meta, 'initial limit must retain intent during shock')
        self.assertEqual(meta['phase'], 'paused')
        self.assertEqual(row['result']['pause_ref'], self.ref)
        self.bot._execute_entry_request(self.request, self.ref)
        self.assertEqual(self.calls, [])

    def test_request_query_and_explicit_cancel_include_paused_plan(self):
        self.prepare()
        self.bot._execute_entry_request(self.request, self.ref)
        self.assertTrue(
            hasattr(self.bot, '_cancel_entry_request'), 'paused intent cancellation missing'
        )
        row = self.bot._cancel_entry_request(self.ref)
        self.assertEqual(row['pause']['decision'], 'cancelled')
        self.assertIn('取消', self.bot._format_entry_request(row))
        self.assertEqual(self.calls, [])

    def test_sdk_guard_catches_shock_after_leverage_setup_but_allows_exit(self):
        self.prepare()
        from entry_risk import install_risk_guard, EntryRiskError

        now = time.time()
        store = self.strategy.entry_requests
        store.reserve(self.ref, self.request)
        store.set_status(self.ref, 'submitting', {})
        store.put_pause(
            dict(
                ref=self.ref,
                attempt_ref=self.ref,
                phase='submitting',
                decision='normal',
                created=now,
                deadline=now + 1000,
                request=self.request,
            )
        )
        outbound = []
        api = SimpleNamespace(create_order=lambda *a, **kw: outbound.append(kw))
        self.exchange._api = api
        self.exchange.create_order = lambda **kw: api.create_order(
            'BTC/USDT:USDT', 'limit', 'buy', 1, 100, {}
        )
        install_risk_guard(self.strategy, self.rpc._freqtrade)
        with self.strategy.risk_request(self.ref):
            with self.assertRaises(EntryRiskError):
                api.create_order('BTC/USDT:USDT', 'limit', 'buy', 1, 100, {})
        self.assertEqual(outbound, [])
        self.exchange.create_order = lambda **kw: outbound.append(kw)
        # Context is cleared after failed entry, so protective SDK calls continue.
        api.create_order('BTC/USDT:USDT', 'market', 'sell', 1, None, {'reduceOnly': True})
        self.assertEqual(len(outbound), 1)

    def test_explicit_price_replacement_retires_confirmed_old_pause_intent(self):
        import test_order_management as management_tests
        from entry_pause import EntryPause

        trade = management_tests.OrderManagementTests.prepare(self, pending=True)
        controller = EntryPause(self.strategy, self.rpc._freqtrade, self.bot._execute_entry_request)
        self.strategy.entry_pause = controller
        now = time.time()
        request = {**self.request, 'entry_price': 100.0, 'ttl_seconds': 3600}
        ref = 'a' * 24
        self.strategy.entry_requests.reserve(ref, request)
        self.strategy.entry_requests.put_pause(
            dict(
                ref=ref,
                attempt_ref=ref,
                request=request,
                created=now,
                deadline=now + 3600,
                phase='watching',
                decision='normal',
                attempts=0,
                trade_id=trade.id,
                order_id='entry-1',
            )
        )
        volume = rows(now, 300, 28, 105)
        volume[-1][4:] = [104.0, 40.0]
        controller.snapshots[trade.pair] = dict(observed_at=now, volume=volume)
        snap = self.management.snapshot(self.bot, trade.id)

        def remove(*args):
            self.Trade.session.delete(trade)
            self.Trade.commit()

        self.rpc._freqtrade.handle_cancel_enter = remove
        result = self.management.replace_entry(self.bot, trade.id, snap['revision'], 99.0, 'b' * 24)
        self.assertEqual(result['state'], 'submitted')
        self.assertEqual(result['result']['pause_ref'], 'b' * 24)
        self.assertEqual(
            self.strategy.entry_requests.get_pause('b' * 24)['request']['entry_price'], 99.0
        )
        self.assertEqual(self.strategy.entry_requests.get_pause(ref)['phase'], 'done')

    def test_new_plan_respects_observed_shock_even_if_forming_bar_reversed(self):
        self.prepare()
        snap = self.controller.snapshots[self.request['pair']]
        snap['signals'] = {'long': dict(at=snap['observed_at'] - 10, ratio=4.0)}
        snap['volume'][-1][4] = 106.0
        snap['volume'][-1][2] = 106.0
        row = self.bot._execute_entry_request(self.request, self.ref)
        self.assertEqual(self.calls, [])
        self.assertEqual(row['result']['pause_ref'], self.ref)
