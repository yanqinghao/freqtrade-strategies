import importlib.util
import unittest
from datetime import datetime, timezone
from concurrent.futures import Future
from test_volume_pause_policy import rows


class PauseMonitorTests(unittest.TestCase):
    def test_live_plan_refreshes_volume_even_when_advisory_cache_is_fresh(self):
        from entry_pause_monitor import EntryPauseMonitor
        from market_monitor import SharedVolumeCandles

        now, calls = 1800000030.0, []
        cache = SharedVolumeCandles(clock=lambda: now)
        cache.get('BTC/USDT:USDT', lambda: rows(now, 300, 28))

        class Client:
            def fetch_ohlcv(self, pair, timeframe, **kwargs):
                calls.append(timeframe)
                data = rows(now, 300 if timeframe == '5m' else 60, 28 if timeframe == '5m' else 10)
                if timeframe == '5m':
                    data[-1][4:] = [99.0, 40.0]
                return data

            def fetch_order_book(self, pair, **kwargs):
                calls.append('book')
                return dict(bids=[[100.0, 1]], asks=[[101.0, 1]])

        monitor = EntryPauseMonitor(clock=lambda: now, volume_candles=cache)
        self.addCleanup(monitor.close)
        monitor._client = Client()
        snapshot = monitor.collect({'BTC/USDT:USDT': now - 120})['BTC/USDT:USDT']
        self.assertEqual(calls, ['5m', '1m', 'book'])
        self.assertEqual(snapshot['signals']['long']['ratio'], 4.0)
        self.assertEqual(snapshot['ask'], 101.0)
        self.assertEqual(
            cache.get('BTC/USDT:USDT', lambda: self.fail('cache should be fresh'))[1],
            snapshot['volume'],
        )

    def test_idle_monitor_fetches_only_volume_and_reuses_it_for_extreme_alerts(self):
        from entry_pause_monitor import EntryPauseMonitor
        from extreme_volume_alerts import ExtremeVolumeMonitor
        from market_monitor import SharedVolumeCandles

        now, calls = 1800000030.0, []

        class Client:
            def fetch_ohlcv(self, pair, timeframe, **kwargs):
                calls.append((pair, timeframe))
                return rows(now, 300, 28)

            def fetch_order_book(self, *args, **kwargs):
                raise AssertionError('idle monitor must not request the book')

        cache = SharedVolumeCandles(clock=lambda: now)
        monitor = EntryPauseMonitor(clock=lambda: now, volume_candles=cache)
        extreme = ExtremeVolumeMonitor(None, volume_candles=cache)
        self.addCleanup(monitor.close)
        self.addCleanup(extreme.close)
        monitor._client = extreme._client = Client()
        snapshots = monitor.collect({'BTC/USDT:USDT': None, 'ETH/USDT:USDT': None})
        for pair in ('BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT'):
            extreme._fetch_public(pair, '5m')
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(tf == '5m' for _, tf in calls))
        self.assertNotIn('minute', snapshots['BTC/USDT:USDT'])
        self.assertEqual(snapshots['BTC/USDT:USDT']['observed_at'], now)

    def test_cached_volume_keeps_original_age_for_entry_safety(self):
        from entry_pause_monitor import EntryPauseMonitor
        from market_monitor import SharedVolumeCandles

        now = 1800000030.0
        cache = SharedVolumeCandles(clock=lambda: now)
        cache.get('BTC/USDT:USDT', lambda: rows(now, 300, 28))
        now += 9
        monitor = EntryPauseMonitor(clock=lambda: now, volume_candles=cache)
        self.addCleanup(monitor.close)
        snapshot = monitor.collect({'BTC/USDT:USDT': None})['BTC/USDT:USDT']
        self.assertEqual(snapshot['observed_at'], 1800000030.0)

    def test_scheduler_requests_history_only_for_live_plans(self):
        from entry_pause_monitor import EntryPauseMonitor

        class Executor:
            def submit(self, fn, needs):
                self.needs = needs
                return Future()

            def shutdown(self, **kwargs):
                pass

        executor = Executor()
        monitor = EntryPauseMonitor(executor=executor)
        self.addCleanup(monitor.close)
        now = datetime.fromtimestamp(1800000030, timezone.utc)
        monitor.tick(
            now,
            [
                dict(phase='paused', created=1799999700, request={'pair': 'BTC/USDT:USDT'}),
                dict(phase='done', created=1799999000, request={'pair': 'ETH/USDT:USDT'}),
            ],
        )
        self.assertEqual(executor.needs, {'BTC/USDT:USDT': 1799999700, 'ETH/USDT:USDT': None})

        # After becoming idle, a newly created ETH plan must re-enable history/book I/O.
        monitor._future.set_result({})
        monitor.tick(datetime.fromtimestamp(1800000040, timezone.utc), [])
        self.assertEqual(executor.needs, {'BTC/USDT:USDT': None, 'ETH/USDT:USDT': None})
        monitor._future.set_result({})
        monitor.tick(
            datetime.fromtimestamp(1800000050, timezone.utc),
            [
                dict(phase='watching', created=1800000045, request={'pair': 'ETH/USDT:USDT'}),
            ],
        )
        self.assertEqual(executor.needs, {'BTC/USDT:USDT': None, 'ETH/USDT:USDT': 1799999930})

    def test_collection_retains_gaps_and_uses_only_public_data(self):
        self.assertIsNotNone(
            importlib.util.find_spec('entry_pause_monitor'), 'public pause monitor missing'
        )
        from entry_pause_monitor import EntryPauseMonitor

        now = 1800000030.0

        class Client:
            def fetch_ohlcv(self, pair, timeframe, since=None, limit=None):
                return rows(now, 300 if timeframe == '5m' else 60, 28 if timeframe == '5m' else 10)

            def fetch_order_book(self, pair, limit):
                return dict(bids=[[100.0, 1]], asks=[[101.0, 1]])

        monitor = EntryPauseMonitor(clock=lambda: now)
        self.addCleanup(monitor.close)
        monitor._client = Client()
        first = monitor.collect({'BTC/USDT:USDT': now - 120})
        self.assertEqual(first['BTC/USDT:USDT']['continuous_since'], now)
        now += 20
        second = monitor.collect({'BTC/USDT:USDT': now - 120})
        self.assertEqual(second['BTC/USDT:USDT']['continuous_since'], now - 20)
        now += 40
        third = monitor.collect({'BTC/USDT:USDT': now - 120})
        self.assertEqual(third['BTC/USDT:USDT']['continuous_since'], now)
        self.assertEqual(third['BTC/USDT:USDT']['ask'], 101)

    def test_pending_worker_never_blocks_trading_thread(self):
        self.assertIsNotNone(
            importlib.util.find_spec('entry_pause_monitor'), 'public pause monitor missing'
        )
        from entry_pause_monitor import EntryPauseMonitor

        monitor = EntryPauseMonitor()
        self.addCleanup(monitor.close)
        monitor._future = Future()
        self.assertEqual(monitor.tick(datetime.now(timezone.utc), []), {})

    def test_valid_shock_is_delivered_when_price_history_fetch_fails(self):
        from entry_pause_monitor import EntryPauseMonitor

        now = 1800000030.0

        class Client:
            def fetch_ohlcv(self, pair, timeframe, since=None, limit=None):
                if timeframe == '1m':
                    raise TimeoutError('history down')
                data = rows(now, 300, 28)
                data[-1][4:] = [99.0, 40.0]
                return data

        monitor = EntryPauseMonitor(clock=lambda: now)
        self.addCleanup(monitor.close)
        monitor._client = Client()
        monitor.collect({'BTC/USDT:USDT': now - 120})
        monitor._future = Future()  # Even while other I/O remains pending, drain known volume.
        snapshots = monitor.tick(datetime.fromtimestamp(now, timezone.utc), [])
        self.assertIn('BTC/USDT:USDT', snapshots)
        self.assertEqual(snapshots['BTC/USDT:USDT']['signals']['long']['ratio'], 4.0)
        self.assertNotIn('minute', snapshots['BTC/USDT:USDT'])

    def test_observed_intrabar_shock_survives_later_direction_reversal(self):
        from entry_pause_monitor import EntryPauseMonitor

        now = 1800000030.0
        adverse = True

        class Client:
            def fetch_ohlcv(self, pair, timeframe, since=None, limit=None):
                data = rows(now, 300 if timeframe == '5m' else 60, 28 if timeframe == '5m' else 10)
                if timeframe == '5m':
                    data[-1][4:] = [99.0 if adverse else 101.0, 40.0]
                return data

            def fetch_order_book(self, pair, limit):
                return dict(bids=[[100.0, 1]], asks=[[101.0, 1]])

        monitor = EntryPauseMonitor(clock=lambda: now)
        self.addCleanup(monitor.close)
        monitor._client = Client()
        monitor.collect({'BTC/USDT:USDT': now - 120})
        adverse = False
        now += 10
        monitor.collect({'BTC/USDT:USDT': now - 120})
        snapshots = monitor.drain()
        signal = snapshots['BTC/USDT:USDT']['signals']['long']
        self.assertIsNotNone(signal)
        self.assertEqual(signal['at'], 1800000030.0)
