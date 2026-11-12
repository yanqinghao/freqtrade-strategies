import json
import unittest
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import test_entry_risk as risk_tests


class UnitTargetLifecycleTests(unittest.TestCase):
    setUp = risk_tests.EntryRiskTests.setUp
    position = risk_tests.EntryRiskTests.position
    order = risk_tests.EntryRiskTests.order

    def prepare(self, short=False):
        risk_tests.EntryRiskTests.prepare(self)
        self.exchange.get_fee = lambda *a, **kw: 0.001
        self.exchange.markets['BTC/USDT:USDT']['taker'] = 0.001
        from freqtrade.exchange import price_to_precision
        from ccxt import TICK_SIZE

        self.exchange.price_to_precision = lambda pair, value, **kw: price_to_precision(
            value, 0.01, TICK_SIZE, **kw
        )
        self.request = self.module.parse_entry(
            ['BTC', '20', '5', 'loss=2', 'profit=1,2,3'], 'short' if short else 'long'
        )

    def provisional(self):
        from entry_risk import plan_for_entry

        return plan_for_entry(self.strategy, self.request, None)

    def filled(self, price=100, amount=1, short=False):
        self.prepare(short)
        plan = self.provisional()
        trade = self.position(short)
        trade.open_rate = price
        trade.amount = amount
        trade.fee_open = trade.fee_close = 0.001
        trade.stake_amount = price * amount / 5
        trade.leverage = 5
        trade.recalc_open_trade_value()
        self.order(
            trade, amount=amount, filled=amount, opened=False, side=trade.entry_side, price=price
        )
        order = trade.orders[-1]
        order.status = 'closed'
        plan['trade_id'] = trade.id
        self.strategy.set_manual_plan(trade.pair, plan)
        trade.set_custom_data('assistant_plan', plan)
        self.Trade.commit()
        return trade, order

    def test_preview_materializes_units_without_saving_or_reserving(self):
        self.prepare()
        before = deepcopy(self.strategy.manual_open)
        result = self.strategy.preview_entry(self.request, None)
        self.assertTrue(result['allowed'])
        self.assertEqual(result['stop_price'], 98.2)
        self.assertEqual(result['resolved_targets'], [101.21, 102.21, 103.21])
        self.assertAlmostEqual(result['estimated_loss'], 1.9982)
        self.assertEqual(self.strategy.manual_open, before)
        self.assertEqual(self.strategy.entry_requests.recent(), [])

    def test_first_fill_freezes_actual_quantity_and_targets_survive_partial_tp_restart(self):
        trade, order = self.filled(price=102, amount=0.5)
        self.strategy.order_filled(trade.pair, trade, order, datetime.now(timezone.utc))
        plan = self.strategy._plan(trade)
        self.assertTrue(plan['target_basis']['anchored'])
        self.assertEqual(plan['target_basis']['rate'], 102)
        self.assertEqual(plan['target_basis']['quantity'], 0.5)
        expected = list(plan['exit_points'])
        trade.amount = 0.2
        trade.fee_close = 0.002
        state = self.strategy._execution(trade)
        state.update(stage=1, tp_filled={'1': 0.15}, protected_stop=102.21)
        self.strategy._save_execution(trade, state)
        self.strategy.manual_open = json.loads(Path(self.strategy.state_file).read_text())[
            'manual_open'
        ]
        self.assertEqual(self.strategy._plan(trade)['exit_points'], expected)
        self.strategy.order_filled(trade.pair, trade, order, datetime.now(timezone.utc))
        self.assertEqual(self.strategy._plan(trade)['exit_points'], expected)

    def test_terminal_partial_entry_recovery_anchors_without_fill_callback(self):
        trade, order = self.filled(price=101, amount=0.4)
        order.status = 'canceled'
        self.strategy._recover_entry_targets(trade)
        plan = self.strategy._plan(trade)
        self.assertTrue(plan['target_basis']['anchored'])
        self.assertEqual(plan['target_basis']['quantity'], 0.4)

    def test_addon_and_setmanual_relative_units_reject_instead_of_rebasing(self):
        self.prepare()
        trade = self.position()
        with self.assertRaises(ValueError):
            self.strategy.preview_entry(self.request, trade)
        with self.assertRaises(ValueError):
            self.bot._update_plan(str(trade.id), ['loss=2', 'profit=1,2'])

    def test_pause_stores_executable_prices_and_retains_raw_units(self):
        self.prepare()
        from entry_pause import EntryPause

        controller = EntryPause(self.strategy, self.rpc._freqtrade, self.bot._execute_entry_request)
        self.strategy.entry_pause = controller
        self.request.update(entry_price=100, ttl_seconds=86400)
        row = self.bot._execute_entry_request(self.request, 'a' * 24)
        self.assertEqual(row['result']['pause_ref'], 'a' * 24)
        meta = self.strategy.entry_requests.get_pause('a' * 24)
        self.assertEqual(meta['request']['stop_loss'], 98.2)
        self.assertEqual(meta['request']['profit'], [1, 2, 3])
        self.assertIn('target_basis', meta['request'])
        copied = self.module.entry_command(meta['request'])
        self.assertIn('loss=2', copied)
        self.assertNotIn(' sl=', copied)

    def test_final_amount_cannot_exceed_user_loss_budget_even_with_caps_disabled(self):
        self.prepare()
        plan = self.provisional()
        Path(self.strategy.risk_limits_file).unlink()
        result = self.risk.assess_entry(self.strategy, self.request, None, plan, rate=100, amount=2)
        self.assertFalse(result['allowed'])
        self.assertTrue(any('预算' in r for r in result['reasons']))

    def test_anchored_native_copy_repairs_provisional_file_without_losing_tp_progress(self):
        trade, order = self.filled()
        provisional = deepcopy(self.strategy.manual_open[trade.pair])
        self.strategy.order_filled(trade.pair, trade, order, datetime.now(timezone.utc))
        anchored = deepcopy(trade.get_custom_data('assistant_plan'))
        self.strategy.set_manual_plan(trade.pair, provisional)
        self.strategy._save_execution(trade, dict(stage=1, tp_filled={'1': 0.3}))
        self.assertEqual(self.strategy._plan(trade)['target_basis'], anchored['target_basis'])
        self.assertEqual(self.strategy._execution(trade)['stage'], 1)

    def test_native_guard_enforces_cash_budget_without_optional_global_limits(self):
        self.prepare()
        plan = self.provisional()
        Path(self.strategy.risk_limits_file).unlink()
        self.strategy.set_manual_plan(self.request['pair'], plan)
        outbound = []
        self.exchange.create_order = lambda **kw: outbound.append(kw)
        self.risk.install_risk_guard(self.strategy, self.rpc._freqtrade)
        kw = dict(
            pair=self.request['pair'],
            ordertype='market',
            side='buy',
            amount=2,
            rate=100,
            leverage=5,
            reduceOnly=False,
            initial_order=True,
        )
        with self.assertRaisesRegex(self.risk.EntryRiskError, '预算'):
            self.exchange.create_order(**kw)
        self.assertEqual(outbound, [])
        self.exchange.create_order(**dict(kw, side='sell', reduceOnly=True))
        self.assertEqual(len(outbound), 1)

    def test_price_override_before_late_fill_cannot_resurrect_relative_tp(self):
        trade, order = self.filled()
        self.bot._update_plan(str(trade.id), ['tp=110,120,130'])
        self.strategy.order_filled(trade.pair, trade, order, datetime.now(timezone.utc))
        plan = self.strategy._plan(trade)
        self.assertNotIn('profit', plan)
        self.assertEqual(plan['exit_points'], [110, 120, 130])

    def test_conflicting_frozen_records_latch_protective_exit(self):
        trade, order = self.filled()
        self.strategy.order_filled(trade.pair, trade, order, datetime.now(timezone.utc))
        changed = deepcopy(self.strategy.manual_open[trade.pair])
        changed['target_basis']['quantity'] = 999
        self.strategy.set_manual_plan(trade.pair, changed)
        self.strategy._plan(trade)
        self.assertEqual(
            self.strategy._execution(trade)['full_exit']['exit_tag'],
            'manual_target_conversion_failed',
        )

    def test_post_fill_conversion_failure_keeps_sl_and_records_exit_intent(self):
        trade, order = self.filled()
        old_stop = self.strategy.manual_open[trade.pair]['stop_loss']
        trade.fee_open = 0.1
        self.strategy.order_filled(trade.pair, trade, order, datetime.now(timezone.utc))
        plan = self.strategy._plan(trade)
        self.assertEqual(plan['stop_loss'], old_stop)
        self.assertTrue(plan['target_basis']['failed'])
        self.assertIn('full_exit', self.strategy._execution(trade))

    def test_closed_old_entry_callback_cannot_overwrite_new_coin_plan(self):
        trade, order = self.filled()
        trade.is_open = False
        self.Trade.commit()
        other = self.position()
        new = dict(direction='long', trade_id=other.id, stop_loss=90, exit_points=[150])
        self.strategy.set_manual_plan(trade.pair, new)
        before = deepcopy(self.strategy.manual_open[trade.pair])
        self.strategy.order_filled(trade.pair, trade, order, datetime.now(timezone.utc))
        self.assertEqual(self.strategy.manual_open[trade.pair], before)

    def test_recovery_then_tp1_then_late_initial_callback_preserves_progress(self):
        trade, order = self.filled()
        self.strategy._recover_entry_targets(trade)
        state = self.strategy._execution(trade)
        state.update(stage=1, tp_filled={'1': 0.3}, initial_amount=1, stage2_amount=0.7)
        self.strategy._save_execution(trade, state)
        trade.amount = 0.7
        self.strategy.order_filled(trade.pair, trade, order, datetime.now(timezone.utc))
        current = self.strategy._execution(trade)
        self.assertEqual(current['stage'], 1)
        self.assertEqual(current['tp_filled'], {'1': 0.3})
        self.assertEqual(current['initial_amount'], 1)

    def test_zero_recorded_fee_is_not_replaced_by_nonzero_estimate(self):
        trade, order = self.filled()
        trade.fee_open = trade.fee_close = 0
        self.strategy.order_filled(trade.pair, trade, order, datetime.now(timezone.utc))
        plan = self.strategy._plan(trade)
        self.assertEqual(plan['target_basis']['fee_open'], 0)
        self.assertEqual(plan['target_basis']['fee_close'], 0)
        self.assertEqual(plan['exit_points'], [101, 102, 103])
