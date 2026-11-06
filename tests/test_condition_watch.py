import importlib.util
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'user_data' / 'strategies'))
from assistant_alerts import AlertQueue

T = 1793966400  # 2026-11-06 12:00 UTC, hour boundary


class WatchTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            importlib.util.find_spec('condition_watch'), 'condition monitor missing'
        )
        import condition_watch as watch

        self.m = watch
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.queue = AlertQueue(Path(self.tmp.name) / 'alerts.sqlite')
        self.store = watch.WatchStore(self.queue)
        self.owner = [1, 2, None]

    def create(self, mode='reject', side='short', **kw):
        spec = dict(
            mode=mode,
            side=side,
            pair='BTC/USDT:USDT',
            tf='1h',
            low=100,
            high=102,
            level=101,
            confirm=99 if side == 'short' else 103,
            invalid=None,
            window=3,
            ttl=86400,
        )
        spec.update(kw)
        return self.store.create(spec, self.owner, str(len(self.store.list(self.owner))), now=T)

    def bar(self, start=T, o=99, h=101, low=98, c=98.5):
        return [start * 1000, o, h, low, c, 10]

    def step(self, w, rows=(), price=None, now=T + 3601):
        quote = None if price is None else dict(price=price, time=now)
        result, message = self.m.evaluate(w, list(rows), quote, now)
        self.store.apply(w, result, message, now=now)
        return next(r for r in self.store.list(self.owner) if r['id'] == w['id'])

    def test_five_modes_both_directions(self):
        for side in ('short', 'long'):
            short = side == 'short'
            with self.subTest(side=side):
                w = self.create('touch', side)
                self.assertEqual(self.step(w, price=101, now=T + 1)['state'], 'confirmed')
                w = self.create('close', side)
                bar = self.bar(o=101, h=105, low=97, c=98 if short else 104)
                self.assertEqual(self.step(w, [bar])['state'], 'confirmed')
                for mode in ('reject', 'fake'):
                    w = self.create(mode, side)
                    self.assertEqual(self.step(w, [bar])['state'], 'confirmed')
                w = self.create('retest', side)
                first = self.bar(o=101, h=105, low=97, c=98 if short else 104)
                w = self.step(w, [first])
                self.assertEqual(w['state'], 'retest')
                self.assertEqual(
                    self.step(
                        w,
                        [self.bar(T + 3600, o=101, h=105, low=97, c=98 if short else 104)],
                        now=T + 7201,
                    )['state'],
                    'confirmed',
                )

    def test_touch_then_this_close_and_window_expiration(self):
        w = self.create(window=1)
        w = self.step(w, price=101, now=T + 120)
        self.assertEqual(w['state'], 'confirm')
        w = self.step(w, [self.bar(c=100)], now=T + 3601)
        self.assertEqual(w['state'], 'expired')
        self.assertIn('未确认', w['reason'])

    def test_wicks_before_creation_and_before_breakout_never_arm(self):
        w = self.create()
        w['created'] = T + 100
        w = self.step(w, [self.bar()], price=98, now=T + 3601)
        self.assertEqual(w['state'], 'waiting')
        w = self.create('retest')
        w = self.step(w, [self.bar()])
        self.assertEqual(w['state'], 'retest')
        w = self.step(w, [self.bar()], price=98, now=T + 3610)
        self.assertEqual(w['state'], 'retest')

    def test_live_touch_cannot_confirm_using_earlier_close(self):
        w = self.create()
        w = self.step(w, [self.bar(h=99)], price=101)
        self.assertEqual(w['state'], 'confirm')
        self.assertEqual(w['armed'], T + 3600)

    def test_invalidation_wins_over_confirmation_and_expiry(self):
        w = self.create(invalid=104)
        w = self.step(w, [self.bar(h=105)])
        self.assertEqual(w['state'], 'invalid')
        w = self.create(ttl=14400)
        self.assertEqual(self.step(w, price=101, now=T + 14400)['state'], 'expired')

    def test_unclosed_stale_bad_and_equal_closes_do_not_confirm(self):
        for row, now in (
            (self.bar(), T + 100),
            (self.bar(), T + 4000),
            (self.bar(h=90), T + 3601),
            (self.bar(c=99), T + 3601),
            (self.bar(c=float('nan')), T + 3601),
        ):
            w = self.create('close')
            self.assertNotEqual(self.step(w, [row], now=now)['state'], 'confirmed')

    def test_restart_dedup_cancel_race_and_transaction_rollback(self):
        w = self.create('touch')
        result, msg = self.m.evaluate(w, [], dict(price=101, time=T + 1), T + 1)
        self.store.cancel(w['id'], self.owner, now=T + 1)
        self.assertFalse(self.store.apply(w, result, msg, now=T + 2))
        self.assertEqual(self.queue.pending(now=T + 2), [])
        w = self.create('touch')
        w = self.step(w, price=101, now=T + 2)
        self.assertEqual(len(self.queue.pending(now=T + 2)), 1)
        self.queue.acknowledge(self.queue.pending(now=T + 2)[0])
        self.store = self.m.WatchStore(AlertQueue(self.queue.path))
        self.step(w, price=101, now=T + 3)
        self.assertEqual(self.queue.pending(now=T + 3), [])
        w = self.create('touch')
        result, msg = self.m.evaluate(w, [], dict(price=101, time=T + 4), T + 4)
        with closing(self.queue._connect()) as db, db:
            db.execute(
                'CREATE TRIGGER fail_notice BEFORE INSERT ON alerts '
                "BEGIN SELECT RAISE(ABORT, 'disk'); END"
            )
        import sqlite3

        with self.assertRaises(sqlite3.IntegrityError):
            self.store.apply(w, result, msg, now=T + 4)
        self.assertEqual(
            next(x for x in self.store.list(self.owner) if x['id'] == w['id'])['state'], 'waiting'
        )

    def test_validation_idempotent_create_ownership_and_priority(self):
        w = self.create('touch')
        self.assertEqual(self.store.create(w['spec'], self.owner, '0', now=T + 3)['id'], w['id'])
        self.assertEqual(self.store.list([2, 3, None]), [])
        with self.assertRaises(ValueError):
            self.store.cancel(w['id'], [2, 3, None], now=T)
        for kw in (
            {'low': 103},
            {'confirm': 101},
            {'invalid': 100},
            {'ttl': 0},
            {'pair': 'DOGE/USDT:USDT'},
            {'window': 0},
            {'level': float('nan')},
        ):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                self.create(**kw)
        w = self.step(w, price=101, now=T + 1)
        self.queue.put('stop:1', '保护未确认', now=T + 2)
        self.assertEqual(self.queue.pending(limit=1, now=T + 2)[0]['key'], 'stop:1')
        self.assertEqual(len(self.queue.pending(now=T + 302)), 1)

    def test_observed_mid_candle_touch_can_confirm_but_restart_cannot_replay_old_close(self):
        w = self.create(window=1)
        w['created'] = T + 100
        w = self.step(w, price=101, now=T + 120)
        w = self.step(w, [self.bar()], now=T + 3601)
        self.assertEqual(w['state'], 'confirmed')
        w = self.create('close')
        self.assertEqual(self.step(w, [self.bar()], now=T + 3700)['state'], 'waiting')

    def test_retest_must_touch_after_break_and_before_deadline(self):
        w = self.create('retest', window=1)
        w = self.step(w, [self.bar()])
        w = self.step(w, [self.bar(T + 3600, o=98, h=99, low=95, c=97)], now=T + 7201)
        self.assertEqual(w['state'], 'expired')
        w = self.create('retest')
        w = self.step(w, [self.bar()])
        w = self.step(w, price=101, now=T + 4000)
        self.assertEqual(w['state'], 'confirm')
        w = self.step(w, [self.bar(T + 3600, o=98, h=101, low=95, c=97)], now=T + 7201)
        self.assertEqual(w['state'], 'confirmed')

    def test_missing_confirmation_candle_expires_instead_of_using_later_close(self):
        w = self.create(window=1)
        w = self.step(w, price=101, now=T + 120)
        w = self.step(w, [self.bar(T + 3600)], now=T + 7201)
        self.assertEqual(w['state'], 'expired')
        self.assertIn('不补发', w['reason'])

    def test_quote_received_before_close_keeps_its_order_across_slow_fetch(self):
        w = self.create('close', invalid=104)
        w['created'] = T + 100
        result, _ = self.m.evaluate(w, [self.bar(h=105)], dict(price=105, time=T + 3599), T + 3601)
        self.assertEqual(result['state'], 'invalid')
        w = self.create(window=1)
        w['created'] = T + 100
        result, _ = self.m.evaluate(w, [self.bar()], dict(price=101, time=T + 3599), T + 3601)
        self.assertEqual(result['state'], 'confirmed')

    def test_background_cache_pair_failure_freshness_and_worker_schedule(self):
        from concurrent.futures import Future
        from datetime import datetime, timezone

        now = [T + 10]
        requests = []

        def fetch(pair, tf):
            requests.append((pair, tf))
            if pair.startswith('ETH'):
                raise RuntimeError('private data should not be logged')
            return [self.bar(o=100, h=102, low=99, c=101)]

        a = self.create('touch')
        b = self.create('touch')
        eth = self.create('touch', pair='ETH/USDT:USDT')
        monitor = self.m.ConditionMonitor(self.store, fetch=fetch, clock=lambda: now[0])
        self.addCleanup(monitor.close)
        monitor.check()
        rows = {w['id']: w for w in self.store.list(self.owner)}
        self.assertEqual(rows[a['id']]['state'], 'confirmed')
        self.assertEqual(rows[b['id']]['state'], 'confirmed')
        self.assertEqual(rows[eth['id']]['state'], 'waiting')
        self.assertIn('不可用', rows[eth['id']]['health'])
        self.assertEqual(requests.count(('BTC/USDT:USDT', '1m')), 1)
        calls = []

        class Executor:
            def submit(self, fn):
                future = Future()
                calls.append((fn, future))
                return future

            def shutdown(self, **kwargs):
                pass

        worker = self.m.ConditionMonitor(self.store, fetch=fetch, executor=Executor())
        self.addCleanup(worker.close)
        worker.tick(datetime.fromtimestamp(T, timezone.utc))
        worker.tick(datetime.fromtimestamp(T + 20, timezone.utc))
        self.assertEqual(len(calls), 1, 'pending work must not queue duplicate requests')
        calls[0][1].set_result(None)
        worker.tick(datetime.fromtimestamp(T + 20, timezone.utc))
        self.assertEqual(len(calls), 2)


if __name__ == '__main__':
    unittest.main()
