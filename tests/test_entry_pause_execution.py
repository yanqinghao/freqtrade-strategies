import importlib.util
import unittest
from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace
import test_entry_lifecycle as lifecycle
from test_volume_pause_policy import rows


class PauseExecutionTests(unittest.TestCase):
    setUp = lifecycle.EntryLifecycleTests.setUp
    add_order = lifecycle.EntryLifecycleTests.add_order
    partial = lifecycle.EntryLifecycleTests.partial

    def prepare(self):
        self.assertIsNotNone(
            importlib.util.find_spec('entry_pause'), 'pause execution controller missing'
        )
        from entry_pause import EntryPause
        from freqtrade.enums import State

        self.bot.state = State.RUNNING
        self.submissions = []
        self.controller = EntryPause(
            self.strategy, self.bot, lambda request, ref: self.submissions.append((request, ref))
        )
        self.ref = 'a' * 24
        now = self.now.timestamp()
        self.meta = dict(
            ref=self.ref,
            attempt_ref=self.ref,
            phase='watching',
            decision='normal',
            created=now - 3600,
            checked_until=now - 120,
            deadline=now + 3600,
            trade_id=self.trade.id,
            order_id=self.order.order_id,
            attempts=0,
            request=dict(
                pair=self.trade.pair,
                direction='long',
                entry_price=100.0,
                stop_loss=90.0,
                size=20.0,
                leverage=20.0,
                ttl_seconds=7200,
                exit_points=[110.0, 120.0, 130.0],
            ),
        )
        self.strategy.entry_requests.put_pause(self.meta)
        self.candles = rows(now, 60, 10, price=101)
        self.volume = rows(now, 300, 28, price=105)
        self.volume[-1][4:] = [104.0, 40.0]
        self.snapshot = dict(
            observed_at=now,
            minute=self.candles,
            volume=self.volume,
            continuous_since=now - 3600,
            bid=101.0,
            ask=101.0,
        )
        self.controller.snapshots[self.trade.pair] = self.snapshot
        return self.controller

    def run_tick(self):
        self.controller.tick(self.now)
        return self.strategy.entry_requests.get_pause(self.ref)

    def test_shock_withdraws_zero_fill_but_keeps_recoverable_plan(self):
        self.prepare()
        meta = self.run_tick()
        self.assertEqual(meta['phase'], 'paused')
        self.assertEqual(meta['decision'], 'paused')
        self.assertEqual(self.cancels, ['entry-1'])
        self.assertEqual(self.Trade.get_open_trade_count(), 0)
        self.assertEqual(meta['request']['entry_price'], 100)
        self.assertEqual(meta['deadline'], self.meta['deadline'])
        self.assertEqual(self.submissions, [])

    def test_partial_during_cancel_is_accounted_and_never_replaced(self):
        self.prepare()
        self.cancelled.update(filled=1.0, remaining=9.0, cost=100.0)
        meta = self.run_tick()
        self.assertEqual(meta['phase'], 'filled')
        self.assertEqual(self.trade.amount, 1.0)
        self.run_tick()
        self.assertEqual(self.submissions, [])

    def test_unknown_cancel_receipt_never_allows_restore(self):
        self.prepare()
        self.cancelled['filled'] = None
        meta = self.run_tick()
        self.assertEqual(meta['phase'], 'canceling')
        self.assertEqual(self.Trade.get_open_trade_count(), 1)
        self.assertEqual(self.submissions, [])

    def test_sl_after_pause_is_terminal_and_rebound_does_not_revive(self):
        self.prepare()
        self.run_tick()
        self.snapshot['minute'][-1][3] = 89
        meta = self.run_tick()
        self.assertEqual(meta['decision'], 'invalidated')
        self.snapshot['minute'][-1][3] = 100
        self.run_tick()
        self.assertEqual(self.submissions, [])

    def test_user_cancel_of_paused_intent_survives_restart(self):
        self.prepare()
        self.run_tick()
        self.controller.cancel(self.ref)
        self.assertEqual(self.run_tick()['decision'], 'cancelled')
        self.assertEqual(self.submissions, [])

    def test_safe_restore_uses_original_limit_and_deadline_only_once(self):
        self.prepare()
        self.run_tick()
        self.now += timedelta(seconds=960)
        stamp = self.now.timestamp()
        self.snapshot.update(
            observed_at=stamp,
            minute=rows(stamp, 60, 30, 101),
            volume=rows(stamp, 300, 28, 101),
            bid=101,
            ask=101,
        )

        def submit(request, ref):
            self.submissions.append((deepcopy(request), ref))
            self.strategy.entry_requests.reserve(ref, request)
            return self.strategy.entry_requests.set_status(ref, 'unknown', {})

        self.controller.submit = submit
        meta = self.run_tick()
        self.assertEqual(len(self.submissions), 1)
        request, ref = self.submissions[0]
        self.assertEqual(request['entry_price'], 100.0)
        self.assertEqual(request['entry_deadline'], self.meta['deadline'])
        self.assertEqual(request['exit_points'], [110.0, 120.0, 130.0])
        self.assertNotEqual(ref, self.ref)
        self.run_tick()
        self.assertEqual(len(self.submissions), 1)
        self.assertEqual(meta['phase'], 'unknown')

    def test_stopped_bot_or_data_gap_does_not_restore_but_ttl_expires(self):
        self.prepare()
        self.run_tick()
        self.now += timedelta(seconds=960)
        stamp = self.now.timestamp()
        self.snapshot.update(
            observed_at=stamp, minute=rows(stamp, 60, 30, 101), volume=rows(stamp, 300, 28, 101)
        )
        from freqtrade.enums import State

        self.bot.state = State.PAUSED
        self.run_tick()
        self.assertEqual(self.submissions, [])
        self.bot.state = State.RUNNING
        self.now += timedelta(seconds=180)
        stamp = self.now.timestamp()
        self.snapshot.update(observed_at=stamp, volume=rows(stamp, 300, 28, 101))
        self.snapshot['minute'] = rows(stamp, 60, 2, 101)
        self.run_tick()
        self.assertEqual(self.submissions, [])
        self.now += timedelta(hours=2)
        self.assertEqual(self.run_tick()['decision'], 'expired')

    def test_cancel_retry_does_not_wait_for_market_feed_to_recover(self):
        self.prepare()
        self.cancelled['filled'] = None
        self.run_tick()
        self.cancelled['filled'] = 0.0
        self.now += timedelta(seconds=60)
        meta = self.run_tick()
        self.assertEqual(meta['phase'], 'paused')
        self.assertEqual(self.Trade.get_open_trade_count(), 0)

    def test_existing_tracked_initial_limit_is_adopted_after_upgrade(self):
        self.prepare()
        from entry_requests import EntryRequests

        # Simulate an existing request ledger from before this feature.
        self.controller.store = EntryRequests(
            self.strategy.entry_requests.path.with_name('old.sqlite')
        )
        self.strategy.entry_requests = self.controller.store
        self.controller.store.reserve(self.ref, self.meta['request'])
        self.controller.store.set_status(
            self.ref, 'submitted', dict(id=self.trade.id, order_ids=[self.order.order_id])
        )
        self.order.ft_order_tag += '|req=' + self.ref
        self.Trade.commit()
        self.assertTrue(
            hasattr(self.controller, 'adopt'), 'existing initial limit adoption missing'
        )
        self.controller.adopt()
        meta = self.controller.store.get_pause(self.ref)
        self.assertEqual(meta['order_id'], 'entry-1')
        self.assertEqual(meta['request']['entry_price'], 100.0)
        self.assertEqual(meta['deadline'], self.order.order_date_utc.timestamp() + 14400)

    def test_explicit_stop_edit_updates_version_without_extending_deadline(self):
        self.prepare()
        self.assertTrue(
            hasattr(self.controller, 'revise_trade'), 'explicit plan version update missing'
        )
        self.controller.revise_trade(
            self.trade.id, dict(direction='long', stop_loss=95.0, exit_points=[115.0, 120.0, 130.0])
        )
        meta = self.strategy.entry_requests.get_pause(self.ref)
        self.assertEqual(meta['request']['stop_loss'], 95.0)
        self.assertEqual(meta['request']['exit_points'][0], 115.0)
        self.assertEqual(meta['deadline'], self.meta['deadline'])
        self.assertEqual(meta['version'], 2)
        self.assertEqual(meta['request']['entry_price'], 100.0)

    def test_real_restoration_tag_keeps_parseable_original_deadline(self):
        self.prepare()
        self.run_tick()
        self.now += timedelta(seconds=960)
        stamp = self.now.timestamp()
        self.snapshot.update(
            observed_at=stamp,
            minute=rows(stamp, 60, 30, 101),
            volume=rows(stamp, 300, 28, 101),
            bid=101,
            ask=101,
        )
        meta = self.strategy.entry_requests.get_pause(self.ref)
        meta['deadline'] += 0.375
        self.strategy.entry_requests.put_pause(meta)
        received = []

        def submit(request, ref):
            from entry_lifecycle import entry_ttl

            tag = (
                'manual_long|ttl='
                + str(request['ttl_seconds'])
                + '|until='
                + str(request['entry_deadline'])
                + '|req='
                + ref
            )
            fake_order = SimpleNamespace(ft_order_tag=tag, order_date_utc=self.now)
            received.append(entry_ttl(fake_order))
            self.strategy.entry_requests.reserve(ref, request)
            return self.strategy.entry_requests.set_status(ref, 'unknown', {})

        self.controller.submit = submit
        self.run_tick()
        self.assertEqual(received, [2640.0])

    def test_receipt_settlement_resolves_old_unknown_before_trade_is_deleted(self):
        self.prepare()
        store = self.strategy.entry_requests
        store.reserve(self.ref, self.meta['request'])
        store.set_status(self.ref, 'unknown', {})
        self.run_tick()
        self.assertEqual(store.unresolved(self.trade.pair), [])
        self.assertEqual(store.get(self.ref)['result']['order_ids'], ['entry-1'])

    def test_confirmed_replacement_does_not_block_explicit_new_plan(self):
        self.prepare()
        self.controller.cancel_trade(self.trade.id, 'replaced')
        self.controller.withdraw(self.strategy.entry_requests.get_pause(self.ref))
        new = 'b' * 24
        self.controller.prepare(self.meta['request'], new, now=self.now.timestamp() + 1)
        self.assertIsNotNone(self.strategy.entry_requests.get_pause(new))

    def test_initial_data_wait_must_not_later_bypass_recovery_range(self):
        self.prepare()
        self.Trade.session.delete(self.trade)
        self.Trade.commit()
        self.controller.cancel(self.ref)
        meta = self.strategy.entry_requests.get_pause(self.ref)
        meta['phase'] = 'done'
        self.strategy.entry_requests.put_pause(meta)
        self.controller.snapshots.clear()
        new = 'c' * 24
        self.controller.prepare(self.meta['request'], new, now=self.now.timestamp())
        self.now += timedelta(seconds=960)
        stamp = self.now.timestamp()
        self.controller.snapshots[self.trade.pair] = dict(
            observed_at=stamp,
            minute=rows(stamp, 60, 30, 115),
            volume=rows(stamp, 300, 28, 115),
            continuous_since=stamp - 1000,
            bid=115,
            ask=115,
        )
        self.run_tick()
        self.assertEqual(self.submissions, [])
        self.assertEqual(self.strategy.entry_requests.get_pause(new)['decision'], 'waiting_range')
