import importlib.util
import sys
import unittest
from types import SimpleNamespace
from pathlib import Path
from test_exit_policy import RuleExchange

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'user_data' / 'strategies'))


class EntryGuardTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            importlib.util.find_spec('entry_guard'), 'entry position guard missing'
        )
        from entry_guard import ensure_position_matches

        self.guard = ensure_position_matches
        self.pair = 'BTC/USDT:USDT'
        self.exchange = RuleExchange(minimum=0.001, step='.001', cost=0.0, pair=self.pair)
        self.positions = []
        self.queries = []

        def fetch(pair):
            self.queries.append(pair)
            return self.positions

        self.exchange.fetch_positions = fetch
        self.trade = SimpleNamespace(amount=0.003, is_short=False, is_open=True)

    def position(self, contracts=0.003, side='long', **kwargs):
        return {
            'symbol': self.pair,
            'contracts': contracts,
            'side': side,
            'contractSize': 1.0,
            'hedged': False,
            'info': {'positionSide': 'BOTH'},
            **kwargs,
        }

    def test_flat_account_allows_new_entry_and_queries_target_fresh(self):
        self.guard(self.exchange, self.pair, None)
        self.guard(self.exchange, self.pair, None)
        self.assertEqual(self.queries, [self.pair, self.pair])

    def test_external_position_blocks_new_entry(self):
        self.positions = [self.position()]
        with self.assertRaisesRegex(ValueError, '未接管'):
            self.guard(self.exchange, self.pair, None)

    def test_same_direction_and_amount_allow_add(self):
        self.positions = [self.position()]
        self.guard(self.exchange, self.pair, self.trade)

    def test_direction_mismatch_blocks_add(self):
        self.positions = [self.position(side='short')]
        with self.assertRaisesRegex(ValueError, '方向'):
            self.guard(self.exchange, self.pair, self.trade)

    def test_one_lot_mismatch_is_not_tolerated(self):
        self.positions = [self.position(contracts=0.004)]
        with self.assertRaisesRegex(ValueError, '数量'):
            self.guard(self.exchange, self.pair, self.trade)

    def test_binary_float_noise_is_tolerated(self):
        self.trade.amount = 0.1 * 3
        self.positions = [self.position(contracts=0.3)]
        self.guard(self.exchange, self.pair, self.trade)

    def test_missing_exchange_position_blocks_local_position(self):
        with self.assertRaisesRegex(ValueError, '数量|空仓'):
            self.guard(self.exchange, self.pair, self.trade)

    def test_contract_size_is_applied_once(self):
        self.exchange.markets[self.pair]['contractSize'] = 0.01
        self.trade.amount = 0.03
        self.positions = [self.position(contracts=3.0, contractSize=0.01)]
        self.guard(self.exchange, self.pair, self.trade)
        self.positions[0]['contracts'] = 4.0
        with self.assertRaisesRegex(ValueError, '数量'):
            self.guard(self.exchange, self.pair, self.trade)

    def test_micro_lot_mismatch_cannot_hide_in_absolute_tolerance(self):
        self.exchange.markets[self.pair]['info']['filters'][0]['stepSize'] = '0.000000000001'
        self.trade.amount = 1.0
        self.positions = [self.position(contracts=1.000000000001)]
        with self.assertRaisesRegex(ValueError, '数量'):
            self.guard(self.exchange, self.pair, self.trade)

    def test_other_symbols_do_not_block_target_entry(self):
        self.positions = [self.position(symbol='ETH/USDT:USDT')]
        self.guard(self.exchange, self.pair, None)

    def test_zero_position_row_is_flat(self):
        self.positions = [self.position(contracts=0.0, side=None)]
        self.guard(self.exchange, self.pair, None)

    def test_malformed_or_hedged_positions_fail_closed(self):
        for values in (
            {'contracts': None},
            {'contracts': float('nan')},
            {'contracts': float('inf')},
            {'contracts': -1.0},
            {'side': None},
            {'hedged': True},
            {'symbol': None},
            {'contractSize': 0.1},
            {'info': {'positionSide': 'LONG'}},
            {'info': 'invalid'},
        ):
            with self.subTest(values=values):
                self.positions = [self.position(**values)]
                with self.assertRaises(ValueError):
                    self.guard(self.exchange, self.pair, self.trade)
        self.positions = [self.position(), self.position()]
        with self.assertRaises(ValueError):
            self.guard(self.exchange, self.pair, self.trade)

    def test_malformed_response_or_local_position_fails_closed(self):
        for response in (None, {}, 'unknown', [None]):
            with self.subTest(response=response):
                self.positions = response
                with self.assertRaises(ValueError):
                    self.guard(self.exchange, self.pair, None)
        self.positions = []
        self.trade.amount = 0.0
        with self.assertRaises(ValueError):
            self.guard(self.exchange, self.pair, self.trade)

    def test_network_failure_is_not_treated_as_empty_and_is_sanitized(self):
        def fail(pair):
            raise TimeoutError('apiKey=TOP_SECRET&signature=SECRET_SIG')

        self.exchange.fetch_positions = fail
        with self.assertRaisesRegex(ValueError, '无法核对') as caught:
            self.guard(self.exchange, self.pair, None)
        self.assertNotIn('TOP_SECRET', str(caught.exception))
        self.assertNotIn('SECRET_SIG', str(caught.exception))

    def test_simulation_does_not_contact_real_account(self):
        self.guard(SimpleNamespace(), self.pair, self.trade, dry_run=True)
