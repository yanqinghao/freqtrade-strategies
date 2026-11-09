import importlib.util
import unittest
from datetime import datetime, timezone
from concurrent.futures import Future
from test_volume_pause_policy import rows


class PauseMonitorTests(unittest.TestCase):
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
