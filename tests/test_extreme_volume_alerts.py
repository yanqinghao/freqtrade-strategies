import importlib.util
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'user_data' / 'strategies'))
from assistant_alerts import AlertQueue


class ExtremeVolumeTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('extreme_volume_alerts'))
        from extreme_volume_alerts import ExtremeVolumeAlerts

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.queue = AlertQueue(Path(self.temp.name) / 'queue.sqlite')
        self.detector = ExtremeVolumeAlerts(self.queue, clock=lambda: self.now)
        self.now = datetime(2026, 11, 6, 12, 2, tzinfo=timezone.utc)
        self.volume = 100.0
        self.close = 102.0

    def rows(self):
        boundary = int(self.now.timestamp()) // 300 * 300000
        return [[boundary - (24 - i) * 300000, 100, 101, 99, 100, 10] for i in range(24)] + [
            [boundary, 100, max(103, self.close), min(97, self.close), self.close, self.volume]
        ]

    def fetch(self, pair, timeframe):
        self.assertEqual(timeframe, '5m')
        rows = self.rows()
        if not pair.startswith('BTC/'):
            rows[-1][5] = 10
        return rows

    def check(self):
        self.detector.check(self.fetch)
        return self.queue.pending(now=self.now.timestamp())

    def test_forming_candle_alerts_without_price_cross_and_survives_restart(self):
        messages = self.check()
        self.assertEqual(len(messages), 1)
        for word in ('BTC', '盘中', '10.00', '+2.00%', '24', '累计', '未收盘'):
            self.assertIn(word, messages[0]['message'])
        self.queue.acknowledge(messages[0])
        self.assertEqual(self.check(), [])
        self.detector = type(self.detector)(AlertQueue(self.queue.path), clock=lambda: self.now)
        self.assertEqual(self.check(), [], 'restart after delivery must not replay')

    def test_no_prorating_and_no_ordinary_volume_alert(self):
        self.volume = 99
        self.now = self.now.replace(minute=0, second=10)
        self.assertEqual(self.check(), [])

    def test_same_direction_cooldown_but_reverse_direction_new_bar_alerts(self):
        first = self.check()[0]
        self.queue.acknowledge(first)
        self.now += timedelta(minutes=5)
        self.assertEqual(self.check(), [])
        self.close = 98
        self.assertEqual(len(self.check()), 1)

    def test_last_closed_bar_allowed_only_for_boundary_grace(self):
        rows = self.rows()
        self.now = self.now.replace(minute=5, second=10)
        self.detector.check(lambda pair, tf: rows)
        self.assertEqual(len(self.queue.pending(now=self.now.timestamp())), 3)
        for row in self.queue.pending(now=self.now.timestamp()):
            self.assertIn('刚收盘', row['message'])
        fresh = type(self.detector)(
            AlertQueue(Path(self.temp.name) / 'fresh.sqlite'), clock=lambda: self.now
        )
        self.now += timedelta(seconds=30)
        fresh.check(lambda pair, tf: rows)
        self.assertEqual(fresh.queue.pending(now=self.now.timestamp()), [])

    def test_bad_gap_zero_volume_and_future_data_never_emit(self):
        cases = []
        rows = self.rows()
        rows.pop(5)
        cases.append(rows)
        rows = self.rows()
        rows[-1][5] = float('nan')
        cases.append(rows)
        rows = self.rows()
        rows[-2][5] = -1
        cases.append(rows)
        rows = self.rows()
        for row in rows[:-1]:
            row[5] = 0
        cases.append(rows)
        rows = self.rows()
        rows[-1][0] += 300000
        cases.append(rows)
        rows = self.rows()
        rows[-1][2] = 90
        cases.append(rows)
        for rows in cases:
            self.detector.check(lambda p, t: rows)
            self.assertEqual(self.queue.pending(now=self.now.timestamp()), [])

    def test_one_pair_failure_does_not_suppress_other_pairs(self):
        def fetch(pair, timeframe):
            if pair.startswith('ETH/'):
                raise RuntimeError('private exchange detail')
            return self.rows()

        with self.assertLogs('extreme_volume_alerts', level='WARNING') as captured:
            self.detector.check(fetch)
        self.assertEqual(len(self.queue.pending(now=self.now.timestamp())), 2)
        self.assertNotIn('private exchange detail', '\n'.join(captured.output))

    def test_detection_time_is_refreshed_after_slow_fetch(self):
        rows = self.rows()
        self.now = self.now.replace(minute=4, second=50)

        def fetch(pair, timeframe):
            self.now = self.now.replace(minute=6, second=0)
            return rows

        self.detector.check(fetch)
        self.assertEqual(self.queue.pending(now=self.now.timestamp()), [])

    def test_fetch_crossing_close_uses_actual_detection_phase_and_time(self):
        rows = self.rows()
        self.now = self.now.replace(minute=4, second=50)

        def fetch(pair, timeframe):
            self.now = self.now.replace(minute=5, second=10)
            return rows

        self.detector.check(fetch)
        message = self.queue.pending(now=self.now.timestamp())[0]
        self.assertIn('刚收盘', message['message'])
        self.assertIn('20:05:10 北京时间', message['message'])
        self.assertEqual(message['updated'], self.now.timestamp())

    def test_urgent_worker_runs_while_hourly_future_is_still_pending(self):
        from concurrent.futures import Future
        from extreme_volume_alerts import ExtremeVolumeMonitor
        from market_monitor import HourlyMarketMonitor

        class Executor:
            def __init__(self):
                self.jobs = []

            def submit(self, fn, *args):
                future = Future()
                self.jobs.append((fn, args, future))
                return future

        slow, fast = Executor(), Executor()
        hourly = HourlyMarketMonitor(executor=slow)
        urgent = ExtremeVolumeMonitor(self.queue, fetch=self.fetch, executor=fast)
        urgent._alerts = self.detector
        hourly.tick(self.now)
        urgent.tick(self.now)
        self.assertEqual(len(slow.jobs), 1)
        self.assertEqual(len(fast.jobs), 1)
        fn, args, future = fast.jobs[-1]
        future.set_result(fn(*args))
        self.assertEqual(len(self.queue.pending(now=self.now.timestamp())), 1)
        urgent.tick(self.now.replace(second=5))
        self.assertEqual(len(fast.jobs), 1)
        urgent.tick(self.now.replace(second=10))
        self.assertEqual(len(fast.jobs), 2)
        self.assertFalse(slow.jobs[0][2].done())


class MarketQueueTests(unittest.TestCase):
    def test_failed_enqueue_rolls_back_dedup_record(self):
        import sqlite3

        with tempfile.TemporaryDirectory() as tmp:
            queue = AlertQueue(Path(tmp) / 'queue.sqlite')
            key = 'market-volume:BTC:1'
            queue.put(key, 'existing', now=100)
            with self.assertRaises(sqlite3.IntegrityError):
                queue.put_market_once(key, 'new', group='BTC:up', now=100)
            queue.acknowledge(queue.pending(now=100)[0])
            self.assertTrue(queue.put_market_once(key, 'retry', group='BTC:up', now=101))

    def test_pruning_old_events_cannot_leave_stale_pending_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue = AlertQueue(Path(tmp) / 'queue.sqlite')
            queue.put_market_once('market-volume:BTC:1', 'old', group='BTC:up', now=100)
            queue.put_market_once('market-volume:ETH:2', 'new', group='ETH:up', now=90000)
            self.assertEqual([r['message'] for r in queue.pending(now=90000)], ['new'])

    def test_dedup_expiry_and_critical_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue = AlertQueue(Path(tmp) / 'queue.sqlite')
            self.assertTrue(
                queue.put_market_once('market-volume:BTC:1', 'volume', group='BTC:up', now=100)
            )
            self.assertFalse(
                queue.put_market_once('market-volume:BTC:1', 'again', group='BTC:up', now=110)
            )
            queue.put('stop:1', 'critical', now=120)
            self.assertEqual(queue.pending(limit=1, now=120)[0]['key'], 'stop:1')
            self.assertEqual([r['key'] for r in queue.pending(now=401)], ['stop:1'])
            queue = AlertQueue(queue.path)
            self.assertFalse(
                queue.put_market_once('market-volume:BTC:1', 'again', group='BTC:up', now=1100)
            )
            self.assertTrue(
                queue.put_market_once('market-volume:BTC:2', 'new', group='BTC:up', now=1100)
            )


if __name__ == '__main__':
    unittest.main()
