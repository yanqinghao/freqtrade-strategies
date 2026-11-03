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


class ExternalEntryOrderTests(unittest.TestCase):
    def setUp(self):
        import entry_guard

        self.assertTrue(hasattr(entry_guard, 'ensure_no_entry_orders'))
        self.guard = entry_guard.ensure_no_entry_orders
        self.pair = 'BTC/USDT:USDT'
        self.regular, self.conditional, self.calls = [], [], []

        def fetch(pair, **kwargs):
            self.calls.append((pair, kwargs))
            return self.conditional if kwargs.get('params', {}).get('trigger') else self.regular

        self.exchange = SimpleNamespace(_api=SimpleNamespace(fetch_open_orders=fetch))
        self.row = {
            'id': 'outside-1',
            'symbol': self.pair,
            'status': 'open',
            'side': 'buy',
            'type': 'limit',
            'reduceOnly': False,
            'info': {'positionSide': 'BOTH', 'reduceOnly': False},
        }

    def test_empty_queries_both_regular_and_conditional_fresh(self):
        self.guard(self.exchange, self.pair)
        self.guard(self.exchange, self.pair)
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(self.calls[1], (self.pair, {'params': {'trigger': True}}))

    def test_regular_and_conditional_entry_orders_block_both_sides(self):
        for conditional in (False, True):
            for side in ('buy', 'sell'):
                with self.subTest(conditional=conditional, side=side):
                    self.regular, self.conditional = [], []
                    row = {
                        **self.row,
                        'side': side,
                        'type': 'stop_market' if conditional else 'limit',
                    }
                    (self.conditional if conditional else self.regular).append(row)
                    with self.assertRaisesRegex(ValueError, '挂单'):
                        self.guard(self.exchange, self.pair)

    def test_explicit_reduce_only_or_close_position_orders_do_not_block(self):
        self.regular = [{**self.row, 'reduceOnly': True, 'info': {'reduceOnly': True}}]
        self.conditional = [
            {
                **self.row,
                'side': 'sell',
                'closePosition': True,
                'info': {'reduceOnly': False, 'closePosition': True},
            }
        ]
        self.guard(self.exchange, self.pair)

    def test_unknown_or_contradictory_flags_and_mode_fail_closed(self):
        for change in (
            {'reduceOnly': True},
            {'reduceOnly': 'invalid'},
            {'info': []},
            {'side': None},
            {'status': None},
            {'id': None},
            {'info': {'positionSide': 'LONG'}},
            {'symbol': None},
        ):
            with self.subTest(change=change):
                self.regular = [{**self.row, **change}]
                with self.assertRaises(ValueError):
                    self.guard(self.exchange, self.pair)

    def test_other_symbols_and_confirmed_terminal_orders_do_not_block(self):
        self.regular = [{**self.row, 'symbol': 'ETH/USDT:USDT'}, {**self.row, 'status': 'canceled'}]
        self.guard(self.exchange, self.pair)

    def test_query_failures_and_malformed_responses_are_not_empty(self):
        def fail(*args, **kwargs):
            raise ValueError('SECRET_SIGNED_URL')

        self.exchange._api.fetch_open_orders = fail
        with self.assertRaises(ValueError) as caught:
            self.guard(self.exchange, self.pair)
        self.assertNotIn('SECRET_SIGNED_URL', str(caught.exception))
        for response in (None, {}, [None]):
            self.exchange._api.fetch_open_orders = lambda *a, **kw: response
            with self.assertRaises(ValueError):
                self.guard(self.exchange, self.pair)

    def test_conditional_query_failure_cannot_be_hidden_by_empty_regular(self):
        def fetch(*args, **kwargs):
            if kwargs:
                raise TimeoutError('private')
            return []

        self.exchange._api.fetch_open_orders = fetch
        with self.assertRaises(ValueError):
            self.guard(self.exchange, self.pair)

    def test_simulation_never_queries_private_orders(self):
        self.guard(SimpleNamespace(), self.pair, dry_run=True)
