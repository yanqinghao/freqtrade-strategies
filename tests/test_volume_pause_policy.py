import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'user_data' / 'strategies'))


def rows(now, width, count, price=100):
    last = int(now) // width * width
    return [
        [(last - (count - 1 - i) * width) * 1000, price, price + 1, price - 1, price, 10.0]
        for i in range(count)
    ]


class VolumePausePolicyTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            importlib.util.find_spec('entry_volume'), 'intrabar pause policy missing'
        )
        import entry_volume

        self.module = entry_volume
        self.now = 1800000030.0
        self.meta = dict(
            created=self.now - 120,
            checked_until=self.now - 120,
            deadline=self.now + 3600,
            last_adverse=self.now - 901,
            request=dict(direction='long', entry_price=95.0, stop_loss=90.0),
        )

    def test_forming_actual_volume_blocks_without_extrapolation(self):
        data = rows(self.now, 300, 28)
        data[-1][4:] = [99.0, 40.0]
        signal = self.module.analyze_volume(data, self.now, self.now, 'long')
        self.assertEqual(signal['ratio'], 4)
        self.assertEqual(signal['at'], self.now)
        self.assertIsNone(self.module.analyze_volume(data, self.now, self.now, 'short'))
        data[-1][-1] = 3.0
        self.assertIsNone(self.module.analyze_volume(data, self.now, self.now, 'long'))

    def test_stale_snapshot_and_missing_or_invalid_candles_fail_closed(self):
        data = rows(self.now, 300, 28)
        for changed in [data[:-1], data[:10] + data[11:], data[::-1]]:
            with self.assertRaises(ValueError):
                self.module.analyze_volume(changed, self.now, self.now, 'long')
        with self.assertRaises(ValueError):
            self.module.analyze_volume(data, self.now - 31, self.now, 'long')
        data[-1][-1] = float('nan')
        with self.assertRaises(ValueError):
            self.module.analyze_volume(data, self.now, self.now, 'long')

    def test_no_entry_cross_resumes_after_quiet_period_without_waiting_for_touch(self):
        data = rows(self.now, 60, 10, price=96.1)
        for row in data:
            row[3] = 95.1
        result = self.module.advance_pause(self.meta, data, None, self.now, self.now)
        self.assertEqual(result['decision'], 'ready')
        self.assertFalse(result['crossed'])
        self.assertEqual(result['deadline'], self.meta['deadline'])

    def test_cross_requires_completed_reclaim_not_a_previous_close(self):
        data = rows(self.now, 60, 10)
        data[-1][1:5] = [100, 101, 94, 94.5]
        result = self.module.advance_pause(self.meta, data, None, self.now, self.now)
        self.assertEqual(result['decision'], 'waiting_reclaim')
        later = self.now + 60
        data[-1][4] = 96
        data.append([int(later) // 60 * 60000, 96, 97, 95.5, 96, 10])
        result = self.module.advance_pause(result, data, None, later, later)
        self.assertEqual(result['decision'], 'ready')

    def test_stop_touch_is_terminal_even_after_rebound(self):
        data = rows(self.now, 60, 10)
        data[-2][3] = 89
        result = self.module.advance_pause(self.meta, data, None, self.now, self.now)
        self.assertEqual(result['decision'], 'invalidated')
        self.assertEqual(result['low_water'], 89)

    def test_new_shock_extends_pause_but_never_order_deadline(self):
        data = rows(self.now, 60, 10)
        result = self.module.advance_pause(
            self.meta, data, {'at': self.now, 'ratio': 4}, self.now, self.now
        )
        self.assertEqual(result['decision'], 'paused')
        self.assertEqual(result['deadline'], self.meta['deadline'])
        self.assertEqual(result['last_adverse'], self.now)

    def test_history_gap_does_not_skip_possible_stop_and_resume(self):
        self.meta['checked_until'] = self.now - 3600
        with self.assertRaises(ValueError):
            self.module.advance_pause(self.meta, rows(self.now, 60, 10), None, self.now, self.now)

    def test_expiry_is_terminal_even_with_new_shock(self):
        self.meta['deadline'] = self.now
        result = self.module.advance_pause(self.meta, [], {'at': self.now}, self.now, self.now)
        self.assertEqual(result['decision'], 'expired')

    def test_far_close_or_current_quote_cannot_restore(self):
        data = rows(self.now, 60, 10, price=96)
        result = self.module.advance_pause(self.meta, data, None, self.now, self.now, quote=96.26)
        self.assertEqual(result['decision'], 'waiting_range')
        result = self.module.advance_pause(
            self.meta, rows(self.now, 60, 10), None, self.now, self.now, quote=95.5
        )
        self.assertEqual(result['decision'], 'waiting_range')

    def test_current_quote_cross_cannot_reuse_previous_candle(self):
        data = rows(self.now, 60, 10, price=96)
        for row in data:
            row[3] = 95.5
        result = self.module.advance_pause(self.meta, data, None, self.now, self.now, quote=94.9)
        self.assertEqual(result['decision'], 'waiting_reclaim')
        result = self.module.advance_pause(result, data, None, self.now, self.now, quote=95.5)
        self.assertEqual(result['decision'], 'waiting_reclaim')

    def test_quote_stop_touch_is_permanent(self):
        result = self.module.advance_pause(
            self.meta, rows(self.now, 60, 10), None, self.now, self.now, quote=90
        )
        self.assertEqual(result['decision'], 'invalidated')
        result = self.module.advance_pause(
            result, rows(self.now, 60, 10), None, self.now, self.now, quote=96
        )
        self.assertEqual(result['decision'], 'invalidated')

    def test_short_reclaim_and_proximity_are_symmetric(self):
        self.meta['request'] = dict(direction='short', entry_price=105.0, stop_loss=110.0)
        data = rows(self.now, 60, 10, price=104)
        result = self.module.advance_pause(self.meta, data, None, self.now, self.now, quote=104)
        self.assertEqual(result['decision'], 'ready')
        result = self.module.advance_pause(self.meta, data, None, self.now, self.now, quote=105.1)
        self.assertEqual(result['decision'], 'waiting_reclaim')

    def test_second_cross_requires_a_new_completed_reclaim(self):
        self.meta.update(crossed=True, crossed_at=self.now - 120)
        data = rows(self.now, 60, 10, 96)
        for row in data:
            row[3] = 95.5
        result = self.module.advance_pause(self.meta, data, None, self.now, self.now, quote=94.9)
        self.assertEqual(result['decision'], 'waiting_reclaim')
        result = self.module.advance_pause(result, data, None, self.now, self.now, quote=96)
        self.assertEqual(result['decision'], 'waiting_reclaim')

    def test_later_candle_cross_also_requires_a_new_completed_reclaim(self):
        self.meta.update(crossed=True, crossed_at=self.now - 120)
        data = rows(self.now, 60, 10, 96)
        for row in data:
            row[3] = 95.5
        data[-1][3] = 94.9
        result = self.module.advance_pause(self.meta, data, None, self.now, self.now, quote=96)
        self.assertEqual(result['decision'], 'waiting_reclaim')
