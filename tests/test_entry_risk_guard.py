import unittest
from types import SimpleNamespace

import test_entry_risk as risk_tests
import test_telegram_orders as order_tests


class EntryRiskGuardTests(unittest.TestCase):
    setUp = order_tests.TelegramOrderTests.setUp
    position = order_tests.TelegramOrderTests.position

    def prepare(self):
        risk_tests.EntryRiskTests.prepare(self)
        self.assertTrue(hasattr(self.risk, 'install_risk_guard'), 'native final risk guard missing')
        self.calls = []
        self.exchange.create_order = lambda **kw: self.calls.append(kw) or {'id': 'accepted'}
        self.risk.install_risk_guard(self.strategy, self.rpc._freqtrade)
        self.kw = {
            'pair': self.request['pair'],
            'ordertype': 'market',
            'side': 'buy',
            'amount': 2.0,
            'rate': 100.0,
            'leverage': 20.0,
            'reduceOnly': False,
            'initial_order': True,
        }
        self.strategy.set_manual_plan(
            self.request['pair'], {'direction': 'long', 'stop_loss': 97, 'exit_points': []}
        )

    def test_passing_entry_and_reduce_only_exit_preserve_native_arguments(self):
        self.prepare()
        self.assertEqual(self.exchange.create_order(**self.kw)['id'], 'accepted')
        self.assertEqual(self.calls, [self.kw])
        self.path.write_text('{bad')
        self.exchange.get_rate = lambda *a, **kw: (_ for _ in ()).throw(AssertionError('exit read'))
        exit_kw = {**self.kw, 'side': 'sell', 'reduceOnly': True}
        self.exchange.create_order(**exit_kw)
        self.assertEqual(self.calls[-1], exit_kw)

    def test_final_quantity_and_leverage_are_checked_even_without_preview(self):
        self.prepare()
        for change in ({'amount': 20.0}, {'leverage': 21.0}, {'amount': 1.01, 'leverage': 1.0}):
            with self.subTest(change=change):
                with self.assertRaises(self.risk.EntryRiskError):
                    self.exchange.create_order(**{**self.kw, **change})
        self.assertEqual(self.calls, [])

    def test_native_execute_entry_minimum_stake_adjustment_is_rejected_before_exchange(self):
        self.prepare()
        from freqtrade.freqtradebot import FreqtradeBot

        bot = object.__new__(FreqtradeBot)
        bot.strategy = self.strategy
        bot.exchange = self.exchange
        bot.get_valid_enter_price_and_stake = lambda *a: (100.0, 101.0, 1.0)
        with self.assertRaisesRegex(self.risk.EntryRiskError, '保证金'):
            bot.execute_entry(self.request['pair'], 100.0, enter_tag='manual_long', leverage_=1.0)
        self.assertEqual(self.calls, [])

    def test_native_add_skips_confirmation_but_cannot_skip_final_guard(self):
        self.prepare()
        from freqtrade.freqtradebot import FreqtradeBot

        trade = self.position()
        trade.amount = 0.1
        trade.leverage = 20
        self.Trade.commit()
        bot = object.__new__(FreqtradeBot)
        bot.strategy = self.strategy
        bot.exchange = self.exchange
        bot.get_valid_enter_price_and_stake = lambda *a: (100.0, 101.0, 20.0)
        bot.handle_similar_open_order = lambda *a: False
        self.strategy.confirm_trade_entry = lambda **kw: (_ for _ in ()).throw(
            AssertionError('native add calls confirm')
        )
        with self.assertRaises(self.risk.EntryRiskError):
            bot.execute_entry(
                trade.pair,
                10.0,
                trade=trade,
                enter_tag='manual_long',
                mode='pos_adjust',
                leverage_=20.0,
            )
        self.assertEqual(self.calls, [])

    def test_only_current_submitting_request_is_ignored_during_final_check(self):
        self.prepare()
        store = self.strategy.entry_requests
        store.reserve('a' * 24, self.request)
        store.set_status('a' * 24, 'submitting', {})
        with self.strategy.risk_request('a' * 24):
            self.exchange.create_order(**self.kw)
            store.reserve('b' * 24, {**self.request, 'pair': 'ETH/USDT:USDT'})
            store.set_status('b' * 24, 'unknown', {})
            with self.assertRaisesRegex(self.risk.EntryRiskError, '待核实'):
                self.exchange.create_order(**self.kw)
        self.assertIsNone(self.strategy._risk_request_ref)
        self.assertEqual(len(self.calls), 1)

    def test_known_final_rejection_restores_plan_and_is_not_unknown(self):
        self.prepare()
        trade = self.position()
        trade.amount = 0.1
        trade.leverage = 20
        self.Trade.commit()
        old = {'direction': 'long', 'trade_id': trade.id, 'stop_loss': 90.0, 'exit_points': [150.0]}
        self.strategy.set_manual_plan(trade.pair, old)

        def native(*a, **kw):
            return self.exchange.create_order(**{**self.kw, 'amount': 20.0})

        self.rpc._rpc_force_entry = native
        row = self.bot._execute_entry_request({**self.request, 'exit_points': [110.0]}, 'a' * 24)
        self.assertEqual(row['state'], 'rejected')
        self.assertEqual(self.strategy.manual_open[trade.pair], old)
        self.assertIsNone(self.strategy._risk_request_ref)
        self.assertEqual(self.calls, [])

    def test_requested_breach_is_rejected_before_plan_or_native_dispatch(self):
        self.prepare()
        self.strategy.delete_manual_plan(self.request['pair'])
        native = []
        self.rpc._rpc_force_entry = lambda *a, **kw: native.append(kw)
        row = self.bot._execute_entry_request({**self.request, 'size': 100}, 'a' * 24)
        self.assertEqual(row['state'], 'rejected')
        self.assertNotIn(self.request['pair'], self.strategy.manual_open)
        self.assertEqual(native, [])

    def test_unknown_native_submission_errors_keep_their_original_type(self):
        self.prepare()

        def fail(**kw):
            raise TimeoutError('SECRET_SIGNED_URL')

        from test_exit_policy import RuleExchange

        second = RuleExchange(minimum=0.001, step='.001', cost=0, pair=self.request['pair'])
        second.markets[self.request['pair']]['taker'] = 0.0005
        second.get_fee = lambda *a, **kw: 0.0005
        second.create_order = fail
        self.strategy.dp = SimpleNamespace(_exchange=second, send_msg=lambda *a, **kw: None)
        self.risk.install_risk_guard(self.strategy, SimpleNamespace(exchange=second))
        with self.assertRaises(TimeoutError):
            second.create_order(**self.kw)

    def test_reinstall_uses_current_strategy_without_nesting_old_guard(self):
        self.prepare()
        new = type(self.strategy)(self.strategy.config)
        new.dp = SimpleNamespace(_exchange=self.exchange, send_msg=lambda *a, **kw: None)
        new.risk_limits_file = self.path.with_name('different-risk.json')
        import json

        new.risk_limits_file.write_text(json.dumps({**self.limits, 'max_margin_usdt': 1}))
        self.risk.install_risk_guard(new, self.rpc._freqtrade)
        self.risk.install_risk_guard(new, self.rpc._freqtrade)
        with self.assertRaisesRegex(self.risk.EntryRiskError, '保证金'):
            self.exchange.create_order(**self.kw)
        self.assertEqual(self.calls, [])

    def test_missing_policy_preserves_legacy_submission_and_bad_policy_blocks_entry(self):
        self.prepare()
        self.path.unlink()
        self.exchange.get_fee = lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError('missing policy estimates')
        )
        self.exchange.create_order(**self.kw)
        self.assertEqual(len(self.calls), 1)
        self.path.write_text('{}')
        with self.assertRaises(self.risk.EntryRiskError):
            self.exchange.create_order(**self.kw)
        self.assertEqual(len(self.calls), 1)

    def test_upward_native_price_rounding_can_breach_margin_and_loss_limits(self):
        self.prepare()
        from freqtrade.exchange import price_to_precision
        from ccxt import TICK_SIZE

        self.exchange.price_to_precision = lambda pair, rate: price_to_precision(
            rate, 0.01, TICK_SIZE
        )
        self.exchange._order_needs_price = lambda side, kind: kind != 'market'
        self.strategy.set_manual_plan(
            self.request['pair'], {'direction': 'long', 'stop_loss': 14.2, 'exit_points': []}
        )
        params = {**self.kw, 'ordertype': 'limit', 'rate': 14.2857, 'amount': 7.0, 'leverage': 1.0}
        with self.assertRaisesRegex(self.risk.EntryRiskError, '保证金'):
            self.exchange.create_order(**params)
        import json

        self.path.write_text(
            json.dumps({**self.limits, 'max_margin_usdt': 1000, 'max_stop_loss_usdt': 0.71})
        )
        with self.assertRaisesRegex(self.risk.EntryRiskError, '亏损'):
            self.exchange.create_order(**params)
        self.assertEqual(self.calls, [])

    def test_downward_native_price_rounding_uses_actual_permitted_margin(self):
        self.prepare()
        from freqtrade.exchange import price_to_precision
        from ccxt import TICK_SIZE
        import json

        self.exchange.price_to_precision = lambda pair, rate: price_to_precision(
            rate, 0.01, TICK_SIZE
        )
        self.exchange._order_needs_price = lambda side, kind: kind != 'market'
        self.strategy.set_manual_plan(
            self.request['pair'], {'direction': 'long', 'stop_loss': 14.2, 'exit_points': []}
        )
        self.path.write_text(json.dumps({**self.limits, 'max_margin_usdt': 99.98}))
        params = {**self.kw, 'ordertype': 'limit', 'rate': 14.2849, 'amount': 7.0, 'leverage': 1.0}
        self.exchange.create_order(**params)
        self.assertEqual(self.calls, [params])
