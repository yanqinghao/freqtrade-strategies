import importlib.util
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'user_data' / 'strategies'))


class MarketAlertTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            importlib.util.find_spec('market_alerts'), 'closed-candle alerts missing'
        )
        from market_alerts import MarketAlerts

        self.cls = MarketAlerts
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'market_alert_state.json'
        self.monitor = self.cls(self.path)
        self.now = datetime(2026, 11, 3, 12, tzinfo=timezone.utc)
        self.close = 100.0
        self.volume = 10.0
        self.previous_volume = 10.0
        self.resistance = 105.0
        self.support = 95.0
        self.fail_pair = None
        self.check()  # Starting the process establishes a baseline, never a historic alert.

    def rows(self, timeframe):
        width = {'15m': 900000, '4h': 14400000, '1d': 86400000}[timeframe]
        boundary = int(self.now.timestamp() * 1000) // width * width
        rows = [
            [
                boundary - (80 - i) * width,
                100.0,
                self.resistance,
                self.support,
                100.0,
                self.previous_volume,
            ]
            for i in range(80)
        ]
        if timeframe == '15m':
            rows[-1][1:6] = [
                100.0,
                max(110.0, self.close),
                min(90.0, self.close),
                self.close,
                self.volume,
            ]
        rows.append([boundary, 100.0, 999999.0, 0.01, 999999.0, 999999.0])
        return rows

    def fetch(self, pair, timeframe):
        if pair == self.fail_pair:
            raise RuntimeError('public data unavailable')
        rows = self.rows(timeframe)
        if pair != 'BTC/USDT:USDT' and timeframe == '15m':
            rows[-2][4:6] = [100.0, 10.0]
        return rows

    def check(self, fetch=None):
        return self.monitor.check(fetch or self.fetch, self.now)

    def advance(self, minutes=15, close=100.0, volume=10.0):
        self.now += timedelta(minutes=minutes)
        self.close, self.volume = close, volume

    def test_initial_snapshot_and_open_candle_do_not_emit_alerts(self):
        self.assertEqual(self.check(), [])
        self.advance()
        self.assertEqual(self.check(), [])
        self.assertTrue(self.path.exists())

    def test_volume_uses_previous_twenty_bars_and_merges_with_breakout(self):
        self.advance(close=110.0, volume=20.0)
        messages = self.check()
        self.assertEqual(len(messages), 1)
        message = messages[0]
        for word in ('BTC', '放量', '2.00', '突破', '4小时', '日线', '105', '110', '10.00%'):
            self.assertIn(word, message)
        self.assertEqual(self.check(), [], 'same closed candle must not be repeated')

    def test_volume_without_a_price_cross_does_not_push(self):
        self.advance(volume=30.0)
        messages = self.check()
        self.assertEqual(messages, [])

    def test_failed_reference_timeframe_preserves_volume_and_healthy_reference(self):
        self.advance(close=110.0, volume=30.0)

        def fetch(pair, timeframe):
            if timeframe == '4h':
                raise RuntimeError('4h public data unavailable')
            return self.fetch(pair, timeframe)

        with self.assertLogs('market_alerts', level='WARNING'):
            messages = self.check(fetch)
        self.assertEqual(len(messages), 1)
        self.assertIn('放量', messages[0])
        self.assertIn('日线阻力 105', messages[0])
        self.assertNotIn('4小时阻力', messages[0])

    def test_missing_references_do_not_promote_volume_to_key_alert(self):
        self.advance(volume=30.0)

        def fetch(pair, timeframe):
            if timeframe != '15m':
                raise RuntimeError('reference data unavailable')
            return self.fetch(pair, timeframe)

        with self.assertLogs('market_alerts', level='WARNING'):
            messages = self.check(fetch)
        self.assertEqual(messages, [])

    def test_unparseable_or_overflowing_volume_does_not_hide_price_cross(self):
        for previous, current in (
            (10.0, None),
            (10.0, 'invalid'),
            (None, 10.0),
            ('invalid', 10.0),
            (1e308, 1e308),
        ):
            with self.subTest(previous=previous, current=current):
                self.advance(minutes=75, close=110.0, volume=current)
                self.previous_volume = previous
                messages = self.check()
                self.assertEqual(len(messages), 1)
                self.assertIn('突破', messages[0])
                self.assertIn('量比不可用', messages[0])

    def test_breakdown_is_independent_of_volume(self):
        self.advance(close=90.0)
        messages = self.check()
        self.assertEqual(len(messages), 1)
        self.assertIn('跌破', messages[0])
        self.assertIn('95', messages[0])
        self.assertNotIn('放量：', messages[0])
        self.assertIn('未达2倍放量阈值', messages[0])

    def test_wicks_do_not_count_as_confirmed_crosses(self):
        self.advance()
        self.assertEqual(self.check(), [])  # High 110 and low 90 crossed both reference levels.

    def test_level_refresh_cannot_create_a_cross_when_price_stays_above_it(self):
        self.advance()

        def fetch(pair, timeframe):
            rows = self.fetch(pair, timeframe)
            if timeframe != '15m':
                for row in rows:
                    row[1:5] = [90.0, 99.0, 85.0, 90.0]
            return rows

        self.assertEqual(self.check(fetch), [])

    def test_reference_cannot_use_higher_candle_closing_with_event(self):
        self.advance(minutes=240, close=110.0)

        def fetch(pair, timeframe):
            rows = self.fetch(pair, timeframe)
            if timeframe == '4h':
                rows[-2][2:5] = [200.0, 95.0, 110.0]
            return rows

        messages = self.check(fetch)
        self.assertEqual(len(messages), 1)
        self.assertIn('4小时阻力 105', messages[0])

    def test_cooldown_covers_level_changes_and_does_not_delay_replay(self):
        self.advance(close=110.0, volume=30.0)
        self.assertEqual(len(self.check()), 1)
        self.advance(close=112.0, volume=30.0)
        self.resistance = 106.0
        followup = self.check()
        self.assertEqual(len(followup), 1)
        self.assertIn('原4小时阻力 105', followup[0])
        self.assertNotIn('收盘异动', followup[0])
        self.advance(minutes=45, close=112.0, volume=30.0)
        self.assertEqual(len(self.check()), 1)
        self.advance(close=100.0)
        self.assertIn('未守住', self.check()[0])
        self.assertEqual(self.check(), [], 'follow-up is once; suppressed events are not replayed')

    def test_restart_preserves_cooldown_and_starts_from_current_baseline(self):
        self.advance(close=110.0, volume=30.0)
        self.assertEqual(len(self.check()), 1)
        self.monitor = self.cls(self.path)
        self.assertEqual(self.check(), [])
        self.advance(close=110.0, volume=30.0)
        self.assertIn('异动后续', self.check()[0])
        self.assertEqual(self.check(), [])
        self.advance(minutes=45, close=110.0, volume=30.0)
        self.assertEqual(len(self.check()), 1)
        data = json.loads(self.path.read_text())
        self.assertIn('BTC/USDT:USDT', data['pairs'])

    def test_recovery_uses_latest_bar_only_and_skips_nonadjacent_closes(self):
        self.advance(minutes=90)

        def historical_spike(pair, timeframe):
            rows = self.fetch(pair, timeframe)
            if timeframe == '15m':
                rows[-4][4:6] = [110.0, 30.0]
            return rows

        self.assertEqual(self.check(historical_spike), [])
        self.advance(close=110.0, volume=30.0)

        def missing_previous(pair, timeframe):
            rows = self.fetch(pair, timeframe)
            if timeframe == '15m':
                del rows[-3]
            return rows

        with self.assertLogs('market_alerts', level='WARNING'):
            self.assertEqual(self.check(missing_previous), [])

    def test_zero_or_invalid_volume_never_produces_a_volume_alert(self):
        for volume in (0.0, -1.0, float('nan'), float('inf')):
            with self.subTest(volume=volume):
                self.advance(minutes=75, close=110.0, volume=volume)
                self.previous_volume = 0.0
                messages = self.check()
                self.assertEqual(len(messages), 1)
                self.assertNotIn('放量', messages[0])
                self.assertIn('量比不可用', messages[0])

    def test_stale_data_is_not_processed(self):
        self.advance(close=110.0, volume=30.0)

        def stale(pair, timeframe):
            rows = self.fetch(pair, timeframe)
            if timeframe == '15m':
                rows = rows[:-2]
            return rows

        with self.assertLogs('market_alerts', level='WARNING'):
            self.assertEqual(self.check(stale), [])
        self.assertEqual(len(self.check()), 1, 'fresh retry must still be eligible')

    def test_failure_for_one_coin_does_not_block_other_coins(self):
        self.advance(volume=30.0)
        self.fail_pair = 'ETH/USDT:USDT'

        def fetch(pair, timeframe):
            rows = self.fetch(pair, timeframe)
            if timeframe == '15m':
                rows[-2][4] = 110.0
                rows[-2][5] = 30.0
            return rows

        with self.assertLogs('market_alerts', level='WARNING'):
            messages = self.check(fetch)
        self.assertEqual(len(messages), 2)
        self.assertTrue(messages[0].startswith('【收盘异动】BTC'))
        self.assertTrue(messages[1].startswith('【收盘异动】SOL'))

    def test_persistence_failure_does_not_consume_event_or_emit_untracked_alert(self):
        self.advance(close=110.0, volume=30.0)
        with patch('market_alerts.update_state', side_effect=OSError('disk unavailable')):
            with self.assertRaises(OSError):
                self.check()
        self.assertEqual(len(self.check()), 1)

    def test_crossing_magnitude_low_volume_and_exact_next_check_time(self):
        self.advance(close=105.21, volume=11)
        message = self.check()[0]
        self.assertIn('高于 0.20%', message)
        self.assertIn('未达2倍放量阈值', message)
        self.assertIn('20:30:00 北京时间', message)

    def test_followup_freezes_original_levels_and_emits_only_once(self):
        self.advance(close=110)
        self.check()
        self.resistance = 120
        self.advance(close=106)
        messages = self.check()
        self.assertEqual(len(messages), 1)
        self.assertIn('异动后续', messages[0])
        self.assertIn('原4小时阻力 105', messages[0])
        self.assertIn('仍在位上', messages[0])
        self.assertNotIn('120', messages[0])
        self.assertEqual(self.check(), [])
        self.advance(close=106)
        self.assertEqual(self.check(), [])

    def test_failed_breakout_and_equal_close_are_not_reported_as_holding(self):
        for price in (104, 105):
            with self.subTest(price=price):
                self.advance(minutes=75, close=110)
                self.check()
                self.advance(close=price)
                message = self.check()[0]
                self.assertIn('未守住', message)
                self.assertNotIn('仍在位上', message)

    def test_breakdown_followup_uses_original_support(self):
        self.advance(close=90)
        self.check()
        self.support = 80
        self.advance(close=96)
        message = self.check()[0]
        self.assertIn('原4小时支撑 95', message)
        self.assertIn('已收回', message)

    def test_restart_retains_pending_followup_but_does_not_replay_initial_event(self):
        self.advance(close=110)
        self.check()
        self.monitor = self.cls(self.path)
        self.assertEqual(self.check(), [])
        self.advance(close=106)
        self.assertIn('异动后续', self.check()[0])
        self.monitor = self.cls(self.path)
        self.assertEqual(self.check(), [])

    def test_restart_at_target_candle_checks_pending_but_skipped_target_expires(self):
        self.advance(close=110)
        self.check()
        self.advance(close=106)
        self.monitor = self.cls(self.path)
        self.assertIn('异动后续', self.check()[0])
        self.advance(minutes=75, close=110)
        self.check()
        self.advance(minutes=30, close=100)
        self.monitor = self.cls(self.path)
        self.assertEqual(self.check(), [])
        self.assertNotIn('followup', json.loads(self.path.read_text())['pairs']['BTC/USDT:USDT'])

    def test_followup_storage_failure_keeps_event_for_retry(self):
        self.advance(close=110)
        self.check()
        self.advance(close=106)
        with patch('market_alerts.update_state', side_effect=OSError('disk unavailable')):
            with self.assertRaises(OSError):
                self.check()
        self.assertIn('异动后续', self.check()[0])
        self.assertEqual(self.check(), [])

    def test_breakdown_below_and_equal_outcomes_are_distinct(self):
        self.advance(close=90)
        self.check()
        self.advance(close=94)
        self.assertIn('仍在位下', self.check()[0])
        self.advance(minutes=75, close=90)
        self.check()
        self.advance(close=95)
        message = self.check()[0]
        self.assertIn('回到原位', message)
        self.assertNotIn('仍在位下', message)

    def test_followup_retains_pending_when_target_data_temporarily_fails(self):
        self.advance(close=110)
        self.check()
        self.advance(close=106)
        self.fail_pair = 'BTC/USDT:USDT'
        with self.assertLogs('market_alerts', level='WARNING'):
            self.assertEqual(self.check(), [])
        self.fail_pair = None

        def missing_references(pair, timeframe):
            if timeframe != '15m':
                raise ValueError('unavailable')
            return self.fetch(pair, timeframe)

        with self.assertLogs('market_alerts', level='WARNING'):
            message = self.check(missing_references)[0]
        self.assertIn('原4小时阻力 105', message)
        self.assertIn('仍在位上', message)

    def test_three_coin_followups_and_reverse_events_are_one_bounded_push(self):
        from market_monitor import HourlyMarketMonitor

        def fetch(pair, timeframe):
            return self.rows(timeframe)

        worker = HourlyMarketMonitor(fetch=fetch, alerts=self.monitor)
        self.addCleanup(worker.close)
        self.advance(close=110)
        self.assertEqual(len(worker._run_cycle(self.now, False)), 1)
        self.advance(close=90)
        messages = worker._run_cycle(self.now, False)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].count('异动后续'), 3)
        self.assertEqual(messages[0].count('收盘异动'), 3)
        self.assertLess(len(messages[0].encode('utf-16-le')) // 2, 4096)
