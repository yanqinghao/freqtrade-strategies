import sys
import unittest
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'user_data/strategies'))


def round_price(value, short):
    from freqtrade.exchange import price_to_precision, ROUND_DOWN, ROUND_UP
    from ccxt import TICK_SIZE

    return price_to_precision(
        float(value), 0.01, TICK_SIZE, rounding_mode=ROUND_DOWN if short else ROUND_UP
    )


class TargetMathTests(unittest.TestCase):
    def convert(self, short=False, **fields):
        from entry_targets import convert_targets

        plan = dict(direction='short' if short else 'long', **fields)
        before = deepcopy(plan)
        result = convert_targets(plan, 100, 1, 0.001, 0.001, round_price)
        self.assertEqual(plan, before)
        return result

    def test_fee_aware_cash_targets_match_hand_calculated_prices(self):
        long = self.convert(loss=2, profit=[1, 2, 3])
        self.assertEqual(long['stop_loss'], 98.2)
        self.assertEqual(long['exit_points'], [101.21, 102.21, 103.21])
        short = self.convert(True, loss=2, profit=[1, 2, 3])
        self.assertEqual(short['stop_loss'], 101.79)
        self.assertEqual(short['exit_points'], [98.8, 97.8, 96.8])

    def test_percent_is_underlying_move_and_can_mix_with_cash(self):
        long = self.convert(sl_pct=1, tp_pct=[1, 2, 3])
        short = self.convert(True, sl_pct=1, tp_pct=[1, 2, 3])
        self.assertEqual((long['stop_loss'], long['exit_points']), (99, [101, 102, 103]))
        self.assertEqual((short['stop_loss'], short['exit_points']), (101, [99, 98, 97]))
        self.assertEqual(self.convert(loss=2, tp_pct=[1, 2])['exit_points'], [101, 102])

    def test_unaffordable_fees_invalid_prices_and_collapsed_ticks_reject(self):
        from entry_targets import TargetError

        for short, fields in [
            (False, dict(loss=0.01, profit=[1])),
            (True, dict(loss=2, profit=[101])),
            (False, dict(sl_pct=100, tp_pct=[1])),
            (True, dict(sl_pct=1, tp_pct=[100])),
            (False, dict(loss=2, profit=[0.001, 0.002])),
        ]:
            with self.subTest(short=short, fields=fields), self.assertRaises(TargetError):
                self.convert(short, **fields)


class TargetParsingTests(unittest.TestCase):
    def setUp(self):
        from test_telegram_assistant import load_telegram

        self.module = load_telegram()

    def test_units_roundtrip_without_derived_prices_becoming_user_arguments(self):
        request = self.module.parse_entry(['BTC', '10', '5', 'loss=2', 'tp_pct=1,2,3'], 'long')
        self.assertEqual(request['loss'], 2)
        self.assertEqual(request['tp_pct'], [1, 2, 3])
        materialized = dict(
            request, stop_loss=98, exit_points=[101, 102, 103], target_basis={'anchored': False}
        )
        command = self.module.entry_command(materialized)
        self.assertIn('loss=2', command)
        self.assertIn('tp_pct=1,2,3', command)
        self.assertNotIn('sl=', command)
        self.assertNotIn(' tp=', command)
        self.assertEqual(self.module.parse_entry(command.split()[1:], 'long'), request)

    def test_each_leg_has_one_unit_and_three_distinct_targets_at_most(self):
        for fields in [
            ('sl=90', 'loss=2'),
            ('loss=2', 'sl_pct=1'),
            ('tp=110', 'profit=2'),
            ('profit=1', 'tp_pct=2'),
            ('loss=nan',),
            ('sl_pct=0',),
            ('loss=2', 'profit=1,2,3,4'),
            ('loss=2', 'profit=1,1'),
            ('loss=2', 'tp_pct=2,1'),
        ]:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self.module.parse_entry(['BTC', '10', '5', *fields], 'long')
