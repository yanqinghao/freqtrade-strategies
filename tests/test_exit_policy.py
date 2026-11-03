import importlib.util
import sys
import unittest
from decimal import Decimal, ROUND_DOWN
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'user_data' / 'strategies'))


class RuleExchange:
    """Public exchange metadata and contract rounding, without network access."""

    def __init__(self, minimum=0.1, step='.1', cost=50.0, contract_size=1.0, pair='COIN/USDT:USDT'):
        self.contract_size = contract_size
        self.step = Decimal(step)
        self.remaining_floor = 0.0
        self.markets = {
            pair: {
                'contract': True,
                'linear': True,
                'contractSize': contract_size,
                'limits': {
                    'amount': {'min': minimum, 'max': 100000},
                    'cost': {'min': cost, 'max': None},
                },
                'info': {
                    'filters': [
                        {
                            'filterType': 'LOT_SIZE',
                            'minQty': str(minimum),
                            'maxQty': '100000',
                            'stepSize': step,
                        }
                    ]
                },
            }
        }

    def amount_to_contract_precision(self, pair, amount):
        contracts = Decimal(str(amount)) / Decimal(str(self.contract_size))
        rounded = (contracts / self.step).to_integral_value(rounding=ROUND_DOWN) * self.step
        return float(rounded * Decimal(str(self.contract_size)))

    def get_min_pair_stake_amount(self, *args):
        return self.remaining_floor

    def get_rate(self, *args, **kwargs):
        return 100.0


class ExitPolicyTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('exit_policy'), 'shared exit policy missing')
        from exit_policy import plan_exit

        self.plan = plan_exit
        self.exchange = RuleExchange()

    def decide(self, amount=1.0, requested=0.3, price=100.0, exempt=False, floor=0.0):
        return self.plan(
            self.exchange,
            'COIN/USDT:USDT',
            amount,
            requested,
            price,
            min_notional_exempt=exempt,
            min_remaining_value=floor,
        )

    def test_partial_below_applicable_notional_becomes_full(self):
        quantity, reason = self.decide()
        self.assertEqual(quantity, 1.0)
        self.assertTrue(reason)

    def test_reduce_only_exemption_preserves_valid_partial(self):
        self.assertEqual(self.decide(exempt=True), (0.3, None))

    def test_tiny_remainder_becomes_full(self):
        self.assertEqual(self.decide(requested=0.8)[0], 1.0)
        self.assertEqual(self.decide(requested=0.8, exempt=True), (0.8, None))
        self.assertEqual(self.decide(requested=0.8, exempt=True, floor=30.0)[0], 1.0)

    def test_native_remaining_floor_is_quote_value_not_margin_or_leveraged_again(self):
        self.assertEqual(self.decide(requested=0.5, exempt=True, floor=40.0), (0.5, None))

    def test_market_minimum_quantity_and_step_are_checked(self):
        self.exchange.markets['COIN/USDT:USDT']['info']['filters'].append(
            {'filterType': 'MARKET_LOT_SIZE', 'minQty': '.4', 'maxQty': '100000', 'stepSize': '.1'}
        )
        self.assertEqual(self.decide(exempt=True)[0], 1.0)
        self.assertEqual(self.decide(requested=0.5, exempt=True), (0.5, None))

    def test_zero_after_rounding_becomes_full(self):
        self.assertEqual(self.decide(requested=0.01, exempt=True)[0], 1.0)

    def test_full_close_also_invalid_is_not_submitted(self):
        quantity, reason = self.decide(amount=0.2, requested=0.1)
        self.assertIsNone(quantity)
        self.assertIn('full', reason)

    def test_contract_size_is_applied_once(self):
        self.exchange = RuleExchange(minimum=1.0, step='1', cost=0, contract_size=0.01)
        self.assertEqual(self.decide(amount=0.1, requested=0.03, exempt=True), (0.03, None))
        self.assertEqual(self.decide(amount=0.1, requested=0.003, exempt=True)[0], 0.1)

    def test_valid_partial_and_full_are_preserved(self):
        self.assertEqual(self.decide(amount=10, requested=3), (3.0, None))
        self.assertEqual(self.decide(requested=1), (1.0, None))

    def test_missing_rules_and_invalid_price_fail_closed(self):
        for value in (0, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                self.decide(price=value)
        self.exchange.markets.clear()
        with self.assertRaises(ValueError):
            self.decide()

    def test_binary_float_noise_does_not_turn_valid_contracts_into_dust(self):
        self.exchange = RuleExchange(minimum=1.0, step='1', cost=0.0, contract_size=0.1)
        self.exchange.amount_to_contract_precision = lambda pair, amount: 0.1 * 3
        quantity, reason = self.decide(amount=0.1 * 3, requested=0.1 * 3, exempt=True)
        self.assertIsNone(reason)
        self.assertAlmostEqual(quantity, 0.3)

    def test_remaining_position_can_exceed_single_order_maximum(self):
        market = self.exchange.markets['COIN/USDT:USDT']
        market['limits']['amount']['max'] = 100.0
        market['info']['filters'][0]['maxQty'] = '100'
        self.assertEqual(self.decide(amount=200, requested=80), (80.0, None))
