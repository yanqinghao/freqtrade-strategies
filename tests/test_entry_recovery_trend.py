import unittest
from copy import deepcopy

from test_volume_pause_policy import rows
from entry_volume import advance_pause


def flat_rows(now, width, count, price):
    data = rows(now, width, count, price)
    for row in data:
        row[1:5] = [price] * 4
    return data


class RecoveryTrendTests(unittest.TestCase):
    def setUp(self):
        self.now = 1800000030.0
        self.meta = dict(
            created=self.now - 120,
            checked_until=self.now - 120,
            deadline=self.now + 3600,
            last_adverse=self.now - 901,
            request=dict(direction='long', entry_price=100.0, stop_loss=90.0),
        )

    def trend(self, direction='震荡'):
        return dict(direction=direction, observed_at=self.now, closed_at=1800000000.0)

    def evaluate(self, quote=99, close=99, trend='震荡', meta=None, data=None):
        return advance_pause(
            meta or self.meta,
            flat_rows(self.now, 60, 10, close) if data is None else data,
            None,
            self.now,
            self.now,
            quote=quote,
            trend=self.trend(trend) if isinstance(trend, str) else trend,
        )

    def test_both_immediate_fill_regions_include_exact_quarter_r_boundary(self):
        for direction, stop, quote, expected in [
            ('long', 90, 97.5, 'ready'),
            ('long', 90, 97.49, 'waiting_range'),
            ('short', 110, 102.5, 'ready'),
            ('short', 110, 102.51, 'waiting_range'),
        ]:
            with self.subTest(direction=direction, quote=quote):
                self.meta['request'].update(direction=direction, stop_loss=stop)
                result = self.evaluate(quote=quote, close=quote)
                self.assertEqual(result['decision'], expected)
                self.assertEqual(result['request'], self.meta['request'])
                self.assertEqual(result['deadline'], self.meta['deadline'])

    def test_immediate_fills_veto_only_opposite_trend_including_quote_equal_to_entry(self):
        for direction, stop, quote, trend, expected in [
            ('long', 90, 99, '偏多', 'ready'),
            ('long', 90, 99, '震荡', 'ready'),
            ('long', 90, 99, '偏空', 'waiting_trend'),
            ('short', 110, 101, '偏空', 'ready'),
            ('short', 110, 101, '震荡', 'ready'),
            ('short', 110, 101, '偏多', 'waiting_trend'),
            ('long', 90, 100, '偏空', 'waiting_trend'),
            ('short', 110, 100, '偏多', 'waiting_trend'),
        ]:
            with self.subTest(direction=direction, quote=quote, trend=trend):
                self.meta['request'].update(direction=direction, stop_loss=stop)
                self.assertEqual(
                    self.evaluate(quote=quote, close=quote, trend=trend)['decision'], expected
                )

    def test_resting_limits_need_neither_proximity_nor_trend_data(self):
        for direction, stop, price in [('long', 90, 150), ('short', 110, 50)]:
            with self.subTest(direction=direction):
                self.meta['request'].update(direction=direction, stop_loss=stop)
                self.assertEqual(
                    self.evaluate(quote=price, close=price, trend=None)['decision'], 'ready'
                )

    def test_region_reclaim_cannot_reuse_a_close_before_breach_or_keep_resetting_at_entry(self):
        for direction, stop, outside, inside in [
            ('long', 90, 97.4, 99),
            ('short', 110, 102.6, 101),
        ]:
            with self.subTest(direction=direction):
                self.meta['request'].update(direction=direction, stop_loss=stop)
                breached = self.evaluate(quote=outside, close=inside)
                self.assertEqual(breached['decision'], 'waiting_range')
                reclaim = self.evaluate(quote=inside, close=inside, meta=breached)
                self.assertEqual(reclaim['decision'], 'waiting_reclaim')
                later = self.now + 60
                data = flat_rows(later, 60, 11, inside)
                result = advance_pause(
                    reclaim,
                    data,
                    None,
                    later,
                    later,
                    quote=inside,
                    trend={**self.trend(), 'observed_at': later},
                )
                self.assertEqual(result['decision'], 'ready')

    def test_current_candle_region_breach_needs_its_close(self):
        data = flat_rows(self.now, 60, 10, 99)
        data[-1][3] = 97.4
        self.assertEqual(self.evaluate(data=data)['decision'], 'waiting_reclaim')

    def test_completed_close_must_be_inside_region_even_if_quote_has_recovered(self):
        self.assertEqual(self.evaluate(quote=99, close=97.4)['decision'], 'waiting_reclaim')
        self.meta['request'].update(direction='short', stop_loss=110)
        self.assertEqual(self.evaluate(quote=101, close=102.6)['decision'], 'waiting_reclaim')

    def test_return_to_resting_side_retains_original_entry_reclaim(self):
        meta = {**self.meta, 'crossed': True, 'crossed_at': self.now - 120}
        self.assertEqual(
            self.evaluate(quote=101, close=99, meta=meta)['decision'], 'waiting_reclaim'
        )
        self.assertEqual(self.evaluate(quote=101, close=101, meta=meta)['decision'], 'ready')

    def test_missing_invalid_stale_or_wrong_period_trend_blocks_only_immediate_fill(self):
        invalid = [
            None,
            {},
            self.trend('unknown'),
            {**self.trend(), 'observed_at': self.now - 31},
            {**self.trend(), 'observed_at': self.now + 1},
            {**self.trend(), 'closed_at': 1799999100},
            {**self.trend(), 'closed_at': 1800000900},
            {**self.trend(), 'observed_at': float('nan')},
        ]
        for trend in invalid:
            with self.subTest(trend=trend):
                self.assertEqual(self.evaluate(trend=trend)['decision'], 'waiting_trend_data')
                self.assertEqual(
                    self.evaluate(quote=110, close=110, trend=trend)['decision'], 'ready'
                )

    def test_new_quiet_period_and_hard_stop_still_precede_trend(self):
        meta = {**self.meta, 'last_adverse': self.now - 899}
        self.assertEqual(self.evaluate(meta=meta)['decision'], 'paused')
        self.assertEqual(self.evaluate(quote=90, trend=None)['decision'], 'invalidated')
        meta = {**self.meta, 'deadline': self.now}
        self.assertEqual(self.evaluate(meta=meta)['decision'], 'expired')

    def test_trend_expiry_at_next_15m_boundary_cannot_authorize_entry(self):
        later = 1800000901.0
        result = advance_pause(
            self.meta,
            flat_rows(later, 60, 30, 99),
            None,
            later,
            later,
            quote=99,
            trend={**self.trend(), 'observed_at': later - 2},
        )
        self.assertEqual(result['decision'], 'waiting_trend_data')

    def test_trend_classifier_uses_completed_candles_with_existing_ema_rules(self):
        from entry_volume import analyze_recovery_trend

        for step, expected in [(1, '偏多'), (-1, '偏空'), (0, '震荡')]:
            with self.subTest(step=step):
                data = flat_rows(self.now, 900, 121, 200)
                for i, row in enumerate(data):
                    row[1:5] = [200 + step * i] * 4
                # A forming candle must not reverse the completed-candle direction.
                data[-1][1:5] = [1 if step == 1 else 10000] * 4
                result = analyze_recovery_trend(data, self.now, self.now)
                self.assertEqual(result, {**self.trend(expected)})

    def test_trend_classifier_rejects_incomplete_invalid_and_stale_data(self):
        from entry_volume import analyze_recovery_trend

        data = flat_rows(self.now, 900, 51, 200)
        self.assertEqual(analyze_recovery_trend(data, self.now, self.now)['direction'], '震荡')
        bad_ohlc = deepcopy(data)
        bad_ohlc[-2][3] = 201
        invalid = [data[:-1], data[1:], data[:20] + data[21:], data[::-1], bad_ohlc]
        for changed in invalid:
            with self.subTest(length=len(changed)), self.assertRaises(ValueError):
                analyze_recovery_trend(changed, self.now, self.now)
        with self.assertRaises(ValueError):
            analyze_recovery_trend(data, self.now - 31, self.now)
