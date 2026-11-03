import importlib.util
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

STRATEGIES = Path(__file__).resolve().parents[1] / 'user_data' / 'strategies'
sys.path.insert(0, str(STRATEGIES))


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


if __name__ == '__main__':
    unittest.main()
