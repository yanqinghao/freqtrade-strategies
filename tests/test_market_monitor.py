import importlib.util
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from datetime import datetime, timezone
from pathlib import Path

STRATEGIES = Path(__file__).resolve().parents[1] / 'user_data' / 'strategies'
sys.path.insert(0, str(STRATEGIES))


class SharedVolumeTests(unittest.TestCase):
    def test_live_refresh_does_not_wait_for_advisory_or_get_overwritten_by_older_result(self):
        from market_monitor import SharedVolumeCandles

        now = [1800000030.0]
        cache = SharedVolumeCandles(clock=lambda: now[0])
        started, release = Event(), Event()

        def slow_fetch():
            started.set()
            release.wait(2)
            return [['old']]

        with ThreadPoolExecutor(max_workers=2) as pool:
            slow = pool.submit(cache.get, 'BTC', slow_fetch)
            try:
                self.assertTrue(started.wait(1))
                now[0] += 1
                live = pool.submit(cache.get, 'BTC', lambda: [['new']], refresh=True)
                self.assertEqual(live.result(timeout=1)[1], [['new']])
            finally:
                release.set()
            slow.result(timeout=1)
        self.assertEqual(
            cache.get('BTC', lambda: self.fail('must reuse newest sample'))[1], [['new']]
        )

    def test_slow_sol_fetch_cannot_block_entry_volume_for_btc(self):
        from market_monitor import SharedVolumeCandles

        cache = SharedVolumeCandles()
        started, release = Event(), Event()

        def slow_fetch():
            started.set()
            release.wait(2)
            return [[1]]

        with ThreadPoolExecutor(max_workers=2) as pool:
            slow = pool.submit(cache.get, 'SOL', slow_fetch)
            try:
                self.assertTrue(started.wait(1))
                fast = pool.submit(cache.get, 'BTC', lambda: [[2]])
                self.assertEqual(fast.result(timeout=1)[1], [[2]])
            finally:
                release.set()
            self.assertEqual(slow.result(timeout=1)[1], [[1]])

    def test_concurrent_consumers_share_one_fetch_and_original_observation_time(self):
        from market_monitor import SharedVolumeCandles

        now, calls = [1800000030.0], []
        cache = SharedVolumeCandles(clock=lambda: now[0])

        def fetch():
            calls.append(True)
            now[0] += 1
            return [[1, 2, 3]]

        with ThreadPoolExecutor(max_workers=2) as pool:
            first, second = list(pool.map(lambda _: cache.get('BTC', fetch), range(2)))
        self.assertEqual(len(calls), 1)
        self.assertEqual(first, (1800000030.0, [[1, 2, 3]]))
        self.assertEqual(second, first)
        first[1][0][0] = 99
        self.assertEqual(cache.get('BTC', fetch)[1], [[1, 2, 3]])

    def test_cache_expires_at_ten_seconds_and_at_candle_boundary_without_stale_fallback(self):
        from market_monitor import SharedVolumeCandles

        now, calls = [1800000299.0], []
        cache = SharedVolumeCandles(clock=lambda: now[0])

        def fetch():
            calls.append(now[0])
            return [[now[0]]]

        cache.get('BTC', fetch)
        now[0] += 1  # New 5m candle must be fetched even though only one second passed.
        self.assertEqual(cache.get('BTC', fetch)[1], [[1800000300.0]])
        now[0] += 10
        with self.assertRaises(TimeoutError):
            cache.get('BTC', lambda: (_ for _ in ()).throw(TimeoutError()))
        self.assertEqual(cache.get('BTC', fetch)[1], [[1800000310.0]])
        self.assertEqual(len(calls), 3)


class MarketMonitorTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.find_spec('market_monitor')
        self.assertIsNotNone(spec, 'higher-timeframe monitor is not implemented')
        import market_monitor

        self.module = market_monitor
        self.now = datetime(2026, 11, 3, 12, tzinfo=timezone.utc)

    def candles(self, timeframe, descending=False):
        width = {'1d': 86400000, '4h': 14400000, '1h': 3600000, '15m': 900000}[timeframe]
        boundary = int(self.now.timestamp() * 1000) // width * width
        rows = []
        for i in range(80):
            close = 200 - i if descending else 100 + i
            rows.append([boundary - (80 - i) * width, close - 1, close + 2, close - 2, close, 10])
        # An extreme, still-open candle must not change trend or reference levels.
        rows.append([boundary, 999999, 999999, 0.01, 999999, 10])
        return rows

    def test_analysis_excludes_open_candles_on_both_timeframes(self):
        for timeframe in ('1d', '4h'):
            with self.subTest(timeframe=timeframe):
                result = self.module.analyze_candles(self.candles(timeframe), timeframe, self.now)
                self.assertEqual(result['close'], 179)
                self.assertEqual(result['trend'], '偏多')
                self.assertLess(result['resistance'], 1000)
                bearish = self.module.analyze_candles(
                    self.candles(timeframe, True), timeframe, self.now
                )
                self.assertEqual(bearish['trend'], '偏空')

    def test_hourly_digest_is_one_short_message_with_observation_conditions(self):
        monitor = self.module.HourlyMarketMonitor(lambda p, t: self.candles(t))
        self.addCleanup(monitor.close)
        messages = monitor._run_cycle(self.now, True)
        self.assertEqual(len(messages), 1)
        self.assertLess(len(messages[0]), 1800)
        for word in ('BTC', 'ETH', 'SOL', '观察', '支撑', '阻力', '/monitor'):
            self.assertIn(word, messages[0])

    def test_missing_data_digest_does_not_offer_directional_guidance(self):
        def fail(pair, timeframe):
            raise ValueError('missing')

        monitor = self.module.HourlyMarketMonitor(fail)
        self.addCleanup(monitor.close)
        with self.assertLogs('market_monitor', level='WARNING'):
            message = monitor._run_cycle(self.now, True)[0]
        self.assertIn('暂不判断', message)
        self.assertNotIn('顺势', message)

    def test_bad_volume_does_not_hide_valid_price_trends(self):
        def fetch(pair, timeframe):
            rows = self.candles(timeframe)
            rows[-2][5] = None
            return rows

        messages = self.module.build_reports(fetch, self.now)
        self.assertFalse(any('行情暂不可用' in message for message in messages))

    def test_digest_exposes_period_trends_and_distances_from_timestamped_close(self):
        message = self.module.build_reports(
            lambda p, t: self.candles(t, descending=t == '1d'), self.now, compact=True
        )[0]
        self.assertIn('日线 偏空 / 4h 偏多\n1h 偏多 / 15m 偏多', message)
        self.assertIn('距支撑 34.08%', message)  # (179 - 118) / 179
        self.assertIn('距阻力 1.12%', message)  # (181 - 179) / 179
        self.assertIn('11-03 20:00:00 北京时间', message)
        self.assertNotIn('日线支撑', message)
        self.assertNotIn('观察：观察', message)

    def test_digest_labels_already_crossed_level_instead_of_negative_distance(self):
        def fetch(pair, timeframe):
            rows = self.candles(timeframe)
            if timeframe == '15m':
                rows[-2][4] = 190
            return rows

        message = self.module.build_reports(fetch, self.now, compact=True)[0]
        self.assertIn('已高于阻力 4.74%', message)  # (190 - 181) / 190
        self.assertNotIn('距阻力 -', message)

    def test_missing_short_price_never_uses_old_price_for_distance(self):
        def fetch(pair, timeframe):
            if timeframe == '15m':
                raise ValueError('missing')
            return self.candles(timeframe)

        with self.assertLogs('market_monitor', level='WARNING'):
            message = self.module.build_reports(fetch, self.now, compact=True)[0]
        self.assertIn('15m 不可用', message)
        self.assertIn('距离不可用', message)
        self.assertNotIn('距支撑', message)

    def test_three_reports_use_four_timeframes_but_only_two_for_levels(self):
        requests = []

        def fetch(pair, timeframe):
            requests.append((pair, timeframe))
            return self.candles(timeframe)

        messages = self.module.build_reports(fetch, self.now)
        self.assertEqual(len(messages), 3)
        self.assertEqual({p for p, _ in requests}, set(self.module.MONITOR_PAIRS))
        self.assertEqual({t for _, t in requests}, {'1d', '4h', '1h', '15m'})
        self.assertEqual(len(requests), 12)
        for message, coin in zip(messages, ('BTC', 'ETH', 'SOL')):
            self.assertIn(coin, message)
            self.assertIn('日线', message)
            self.assertIn('4小时', message)
            self.assertIn('15分钟', message)
            self.assertIn('1小时', message)
            self.assertEqual(message.count('参考支撑'), 2)
            self.assertIn('支撑', message)
            self.assertIn('阻力', message)
            self.assertIn('不触发下单', message)

    def test_short_timeframes_only_classify_trend_and_regime(self):
        self.assertTrue(hasattr(self.module, 'analyze_trend'), 'short-term trend analysis missing')
        result = self.module.analyze_trend(self.candles('15m'), '15m', self.now)
        self.assertEqual(result['trend'], '偏多')
        self.assertNotIn('support', result)
        self.assertNotIn('resistance', result)
        with self.assertRaises(ValueError):
            self.module.analyze_candles(self.candles('15m'), '15m', self.now)
        self.assertIn(
            '回调', self.module.classify_regime({'1d': '偏多', '4h': '偏多', '1h': '偏空', '15m': '偏空'})
        )
        self.assertIn(
            '反弹', self.module.classify_regime({'1d': '偏空', '4h': '偏空', '1h': '偏多', '15m': '偏多'})
        )

    def test_failure_for_one_coin_does_not_hide_other_reports(self):
        def fetch(pair, timeframe):
            if pair.startswith('ETH/'):
                raise RuntimeError('network unavailable')
            return self.candles(timeframe)

        with self.assertLogs('market_monitor', level='WARNING'):
            messages = self.module.build_reports(fetch, self.now)
        self.assertEqual(len(messages), 3)
        self.assertIn('暂不可用', messages[1])
        self.assertIn('偏多', messages[0])
        self.assertIn('偏多', messages[2])

    def test_monitor_is_hourly_and_does_not_block_trading(self):
        from concurrent.futures import Future

        class Executor:
            def __init__(self):
                self.jobs = []

            def submit(self, fn, *args):
                future = Future()
                self.jobs.append((fn, args, future))
                return future

        executor = Executor()
        monitor = self.module.HourlyMarketMonitor(lambda p, t: self.candles(t), executor=executor)
        self.assertEqual(monitor.tick(self.now), [])
        self.assertEqual(len(executor.jobs), 1)
        monitor.tick(self.now)
        self.assertEqual(len(executor.jobs), 1)
        executor.jobs[0][2].set_result(['BTC', 'ETH', 'SOL'])
        self.assertEqual(monitor.tick(self.now), ['BTC', 'ETH', 'SOL'])
        self.assertEqual(len(executor.jobs), 1)
        monitor.tick(self.now.replace(hour=13))
        self.assertEqual(len(executor.jobs), 2)

    def test_insufficient_or_invalid_data_is_rejected(self):
        for rows in ([], self.candles('1d')[:10], [[0, 1, 2, 0, float('nan'), 1]] * 60):
            with self.subTest(rows=len(rows)):
                with self.assertRaises(ValueError):
                    self.module.analyze_candles(rows, '1d', self.now)

    def test_alert_checks_are_minutely_without_repeating_hourly_reports_or_fetches(self):
        from concurrent.futures import Future
        from market_alerts import MarketAlerts
        import inspect

        self.assertIn('alerts', inspect.signature(self.module.HourlyMarketMonitor).parameters)

        class Executor:
            def __init__(self):
                self.jobs = []

            def submit(self, fn, *args):
                future = Future()
                self.jobs.append((fn, args, future))
                return future

            def finish(self):
                fn, args, future = self.jobs[-1]
                future.set_result(fn(*args))

        requests = []

        def fetch(pair, timeframe):
            requests.append((pair, timeframe))
            return self.candles(timeframe)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'alerts.json'
            executor = Executor()
            monitor = self.module.HourlyMarketMonitor(
                fetch, executor=executor, alerts=MarketAlerts(path)
            )
            self.assertEqual(monitor.tick(self.now), [])
            self.assertEqual(
                monitor.tick(self.now.replace(minute=1)), [], 'pending data must never block'
            )
            self.assertEqual(len(executor.jobs), 1)
            executor.finish()
            self.assertEqual(len(monitor.tick(self.now)), 1)
            self.assertEqual(len(requests), 12, 'hourly and alerts share a per-cycle snapshot')
            before = path.read_text()
            self.assertEqual(len(monitor.reports(self.now)), 3)
            self.assertEqual(path.read_text(), before, 'manual reports cannot consume alert state')
            self.now = self.now.replace(minute=1)
            self.assertEqual(monitor.tick(self.now), [])
            self.assertEqual(len(executor.jobs), 2)
            executor.finish()
            self.assertEqual(monitor.tick(self.now), [])
            self.now = self.now.replace(hour=13, minute=0)
            monitor.tick(self.now)
            executor.finish()
            self.assertEqual(len(monitor.tick(self.now)), 1)

    def test_alert_storage_failure_does_not_discard_hourly_reports(self):
        import inspect
        from market_alerts import MarketAlerts

        self.assertIn('alerts', inspect.signature(self.module.HourlyMarketMonitor).parameters)
        with tempfile.TemporaryDirectory() as directory:
            monitor = self.module.HourlyMarketMonitor(
                lambda p, t: self.candles(t), alerts=MarketAlerts(directory)
            )
            self.addCleanup(monitor.close)
            with self.assertLogs('market_monitor', level='ERROR'):
                messages = monitor._run_cycle(self.now, True)
            self.assertEqual(len(messages), 1)


if __name__ == '__main__':
    unittest.main()
