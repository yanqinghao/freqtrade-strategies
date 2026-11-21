import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock

import test_telegram_orders as order_tests


class OrderManagementTests(unittest.TestCase):
    setUp = order_tests.TelegramOrderTests.setUp
    position = order_tests.TelegramOrderTests.position

    def prepare(self, pending=False, short=False):
        from freqtrade.persistence import Order
        import order_management as management

        self.management = management
        trade = self.position(short)
        plan = dict(self.request, trade_id=trade.id)
        if short:
            plan.update(direction='short', stop_loss=110, exit_points=[90, 80, 70])
        self.strategy.set_manual_plan(trade.pair, plan)
        self.strategy._plan(trade)
        if pending:
            trade.amount = 0
            order = Order(
                order_id='entry-1',
                ft_order_side=trade.entry_side,
                ft_is_open=True,
                ft_pair=trade.pair,
                order_type='limit',
                status='open',
                symbol=trade.pair,
                side=trade.entry_side,
                amount=3,
                filled=0,
                remaining=3,
                price=100,
                ft_amount=3,
                ft_price=100,
                order_date=datetime.now(timezone.utc),
                ft_order_tag='manual_' + plan['direction'] + '|ttl=3600',
            )
            trade.orders.append(order)
            self.receipt = dict(
                id=order.order_id,
                symbol=trade.pair,
                side=trade.entry_side,
                type='limit',
                status='open',
                amount=3.0,
                filled=0.0,
                price=100.0,
            )
            self.exchange.fetch_order = Mock(side_effect=lambda *a: dict(self.receipt))
            self.exchange.cancel_order_with_result = Mock(
                side_effect=lambda *a: dict(self.receipt, status='canceled')
            )
        self.exchange.get_rate = Mock(return_value=100.0)
        self.Trade.commit()
        return trade

    def test_snapshot_refreshes_price_and_receipt_without_submitting(self):
        trade = self.prepare(pending=True)
        snap = self.management.snapshot(self.bot, trade.id)
        self.assertEqual(snap['kind'], 'pending')
        self.assertEqual(snap['rate'], 100)
        self.exchange.get_rate.assert_called_once_with(
            trade.pair, side='exit', is_short=False, refresh=True
        )
        self.assertEqual(snap['orders'][0]['filled'], 0)
        self.exchange.cancel_order_with_result.assert_not_called()

    def test_snapshot_failure_disables_actions_and_preserves_plan(self):
        trade = self.prepare(pending=True)
        self.exchange.fetch_order.side_effect = OSError('offline')
        snap = self.management.snapshot(self.bot, trade.id)
        self.assertFalse(snap['fresh'])
        with self.assertRaisesRegex(ValueError, '核实'):
            self.management.edit_plan(self.bot, trade.id, snap['revision'], {'stop_loss': 95})
        self.assertEqual(self.strategy.manual_open[trade.pair]['stop_loss'], 90)

    def test_future_tp_edit_keeps_completed_stages_and_quantities(self):
        trade = self.prepare()
        state = dict(
            stage=2,
            tp_filled={'1': 3, '2': 3.5},
            tp_targets={'1': 3, '2': 3.5},
            initial_amount=10,
            stage2_amount=7,
            tp2_completed_at=123,
            protected_stop=110,
        )
        self.strategy._save_execution(trade, state)
        snap = self.management.snapshot(self.bot, trade.id)
        self.management.edit_plan(self.bot, trade.id, snap['revision'], {'tp3': 140})
        self.assertEqual(self.strategy._plan(trade)['exit_points'], [110, 120, 140])
        self.assertEqual(self.strategy._execution(trade), state)
        snap = self.management.snapshot(self.bot, trade.id)
        with self.assertRaisesRegex(ValueError, '完成'):
            self.management.edit_plan(self.bot, trade.id, snap['revision'], {'tp1': 115})

    def test_tail_preset_preserves_tp1_protection_and_final_target(self):
        trade = self.prepare()
        self.exchange.get_rate.return_value = 111
        self.strategy._save_execution(trade, {'stage': 1, 'protected_stop': 101})
        snap = self.management.snapshot(self.bot, trade.id)
        self.management.edit_plan(
            self.bot, trade.id, snap['revision'], {'tp2': 111, 'tail_stop': 'keep'}
        )
        state = self.strategy._execution(trade)
        state['stage'] = 2
        self.strategy._save_execution(trade, state)
        plan = self.strategy._plan(trade)
        self.assertEqual(plan['exit_points'], [110, 111, 130])
        self.assertEqual(self.strategy._stop_price(trade, plan), 101)
        self.assertIsNone(self.strategy.custom_exit(trade.pair, trade, None, 105, 0))
        self.assertEqual(self.management.snapshot(self.bot, trade.id)['stop'], 101)

    def test_active_tail_edits_lower_long_or_raise_short_to_fee_adjusted_cost(self):
        from ccxt import TICK_SIZE
        from freqtrade.exchange import price_to_precision

        for short, old, rate, floor in [(False, 110, 115, 100.11), (True, 90, 85, 99.90)]:
            trade = self.prepare(short=short)
            trade.price_precision, trade.precision_mode_price = 0.01, TICK_SIZE
            self.exchange.price_to_precision = (
                lambda pair, value, rounding_mode: price_to_precision(
                    value, 0.01, TICK_SIZE, rounding_mode=rounding_mode
                )
            )
            self.exchange.get_rate.return_value = rate
            self.strategy._save_execution(trade, {'stage': 2, 'protected_stop': old})
            trade.stop_loss = old
            snap = self.management.snapshot(self.bot, trade.id)
            for invalid in (rate, 100):
                with self.assertRaises(ValueError):
                    self.management.edit_plan(
                        self.bot, trade.id, snap['revision'], {'tail_stop': invalid}
                    )
            self.rpc._freqtrade.handle_stoploss_on_exchange = Mock(return_value=False)
            self.management.edit_plan(self.bot, trade.id, snap['revision'], {'tail_stop': 'cost'})
            self.assertEqual(self.strategy._stop_price(trade, self.strategy._plan(trade)), floor)
            self.assertEqual(trade.stop_loss, floor)
            self.assertEqual(self.management.snapshot(self.bot, trade.id)['stop'], floor)

    def test_live_tail_replacement_checks_both_exchange_receipts(self):
        self.check_live_stop_replacement('tail_stop')

    def test_live_hard_sl_replacement_checks_both_exchange_receipts(self):
        self.check_live_stop_replacement('stop_loss')

    def check_live_stop_replacement(self, field):
        from unittest.mock import patch
        from freqtrade.persistence import Order
        from ccxt import TICK_SIZE

        trade = self.prepare()
        trade.price_precision, trade.precision_mode_price = 0.01, TICK_SIZE
        self.strategy._save_execution(trade, {'stage': 2, 'protected_stop': 110})
        trade.stop_loss = 110
        self.strategy.config['dry_run'] = False
        self.exchange.get_rate.return_value = 112
        self.exchange.fetch_positions = Mock(
            return_value=[
                dict(symbol=trade.pair, side='long', contracts=trade.amount, hedged=False)
            ]
        )
        self.exchange.price_to_precision = lambda pair, value, rounding_mode: round(value, 2)
        raw = dict(
            id='old',
            symbol=trade.pair,
            side='sell',
            type='stop_market',
            status='open',
            amount=trade.amount,
            remaining=trade.amount,
            filled=0,
            stopPrice=110,
            reduceOnly=True,
        )

        def tracked(row):
            trade.orders.append(
                Order.parse_from_ccxt_object(
                    row, trade.pair, 'stoploss', trade.amount, row['stopPrice']
                )
            )

        tracked(raw)
        self.Trade.commit()
        bot = self.rpc._freqtrade
        self.exchange.fetch_stoploss_order = Mock(
            side_effect=[raw, raw, dict(raw, status='closed', filled=trade.amount)]
        )
        self.exchange.cancel_stoploss_order = Mock()
        bot.handle_stoploss_on_exchange = Mock()
        bot.update_trade_state = Mock(side_effect=lambda t, ident, row, **kw: t.update_order(row))

        def change(price=103):
            snap = self.management.snapshot(self.bot, trade.id)
            return self.management.edit_plan(self.bot, trade.id, snap['revision'], {field: price})

        with self.assertRaisesRegex(ValueError, '成交待核实'):
            change()
        bot.handle_stoploss_on_exchange.assert_not_called()
        self.assertEqual(trade.stop_loss, 110)
        self.exchange.fetch_stoploss_order.side_effect = None
        self.exchange.fetch_stoploss_order.return_value = raw
        with patch('tail_stop.sleep'), self.assertRaisesRegex(ValueError, '新止损未设置'):
            change()
        self.assertEqual(self.exchange.cancel_stoploss_order.call_count, 2)
        bot.update_trade_state.assert_not_called()
        bot.handle_stoploss_on_exchange.assert_not_called()
        from freqtrade.exceptions import TemporaryError

        new = dict(raw, id='new', stopPrice=103)
        self.exchange.cancel_stoploss_order.side_effect = TemporaryError('ack unavailable')
        # A cancellation may still read as open before its terminal receipt is visible.
        self.exchange.fetch_stoploss_order.side_effect = [
            raw,
            TemporaryError('receipt unavailable'),
            raw,
            dict(raw, status='canceled', filled=None),
            dict(raw, status='canceled'),
            new,
        ]
        bot.handle_stoploss_on_exchange.side_effect = lambda t: tracked(new)
        self.assertIn('交易所已确认', change())
        self.assertEqual(trade.stop_loss, 103)
        self.assertEqual(self.strategy._execution(trade)['stage'], 2)
        self.assertEqual([o.order_id for o in trade.open_sl_orders], ['new'])
        self.assertIsNone(self.strategy.custom_exit(trade.pair, trade, None, 105, 0))

        self.Trade.session.expire_all()
        restarted = type(self.strategy)(self.strategy.config)
        self.assertEqual(restarted._stop_price(trade, restarted._plan(trade)), 103)
        self.exchange.fetch_stoploss_order.side_effect = [dict(new, stopPrice=110)]
        from stop_protection import check_protection

        self.assertFalse(
            check_protection(self.strategy, trade, datetime.now(timezone.utc), force=True)
        )
        # A clean cancel followed by a rejected replacement must never be shown as confirmed.
        self.exchange.fetch_stoploss_order.side_effect = [
            *([dict(new, stopPrice=110)] if field == 'tail_stop' else []),
            new,
            dict(new, status='canceled'),
        ]
        bot.handle_stoploss_on_exchange.side_effect = lambda t: False
        self.assertIn('尚未确认', change(102 if field == 'stop_loss' else 103))
        self.assertEqual(trade.open_sl_orders, [])

    def test_pending_can_edit_rules_and_tp_but_changed_fill_invalidates_edit(self):
        trade = self.prepare(pending=True)
        snap = self.management.snapshot(self.bot, trade.id)
        self.management.edit_plan(
            self.bot, trade.id, snap['revision'], {'soft_sl': 92, 'soft_tf': '4h'}
        )
        snap = self.management.snapshot(self.bot, trade.id)
        self.management.edit_plan(self.bot, trade.id, snap['revision'], {'tp2': 125})
        snap = self.management.snapshot(self.bot, trade.id)
        self.receipt['filled'] = 0.5
        with self.assertRaisesRegex(ValueError, '变化'):
            self.management.edit_plan(self.bot, trade.id, snap['revision'], {'stop_loss': 95})

    def test_short_sl_respects_market_price_and_tp_order_is_not_silently_sorted(self):
        trade = self.prepare(short=True)
        snap = self.management.snapshot(self.bot, trade.id)
        for patch in ({'stop_loss': 99}, {'tp2': 95}, {'tp1': float('nan')}):
            with self.assertRaises(ValueError):
                self.management.edit_plan(self.bot, trade.id, snap['revision'], patch)
        self.management.edit_plan(self.bot, trade.id, snap['revision'], {'stop_loss': 105})
        self.assertEqual(self.strategy.manual_open[trade.pair]['stop_loss'], 105)

    def test_manual_sl_correction_reaches_native_and_strategy_after_restart(self):
        for short, wrong, corrected in [(False, 98, 89), (True, 102, 112)]:
            with self.subTest(short=short):
                trade = self.prepare(short=short)
                self.strategy.order_types = dict(
                    self.strategy.order_types, stoploss_on_exchange=False
                )
                self.exchange.price_to_precision = lambda pair, value, rounding_mode: value
                snap = self.management.snapshot(self.bot, trade.id)
                self.management.edit_plan(
                    self.bot, trade.id, snap['revision'], {'stop_loss': wrong}
                )
                trade.stop_loss = self.strategy._stop_price(trade, self.strategy._plan(trade))
                snap = self.management.snapshot(self.bot, trade.id)
                if short:
                    self.bot._update_plan(str(trade.id), [f'sl={corrected}'])
                else:
                    self.management.edit_plan(
                        self.bot, trade.id, snap['revision'], {'stop_loss': corrected}
                    )
                self.Trade.session.expire_all()
                restarted = type(self.strategy)(self.strategy.config)
                self.assertEqual(restarted._stop_price(trade, restarted._plan(trade)), corrected)
                self.assertEqual(trade.stop_loss, corrected)
                self.assertEqual(self.management.snapshot(self.bot, trade.id)['stop'], corrected)

    def test_next_tp_resumes_automatic_protection_after_manual_correction(self):
        trade = self.prepare()
        self.strategy.order_types = dict(self.strategy.order_types, stoploss_on_exchange=False)
        self.exchange.price_to_precision = lambda pair, value, rounding_mode: value
        self.exchange.get_rate.return_value = 115
        self.strategy._save_execution(trade, {'protected_stop': 105, 'stage': 1})
        self.bot._update_plan(str(trade.id), ['sl=97'])
        plan = self.strategy._plan(trade)
        self.assertEqual(self.strategy._stop_price(trade, plan), 97)
        self.assertEqual(self.management.snapshot(self.bot, trade.id)['stop'], 97)
        state = self.strategy._execution(trade)
        state['stage'] = 2
        self.strategy._save_execution(trade, state)
        self.assertEqual(self.strategy._stop_price(trade, plan), 110)
        self.assertEqual(self.management.snapshot(self.bot, trade.id)['stop'], 110)

    def test_pending_sl_correction_replaces_cached_protection_without_exchange_order(self):
        trade = self.prepare(pending=True)
        self.strategy._save_execution(trade, {'protected_stop': 90})
        snap = self.management.snapshot(self.bot, trade.id)
        self.management.edit_plan(self.bot, trade.id, snap['revision'], {'stop_loss': 85})
        self.assertEqual(self.management.snapshot(self.bot, trade.id)['stop'], 85)

    def test_replacement_partial_cancel_never_resubmits(self):
        trade = self.prepare(pending=True)
        snap = self.management.snapshot(self.bot, trade.id)
        self.exchange.cancel_order_with_result.side_effect = lambda *a: dict(
            self.receipt, status='canceled', filled=0.1
        )
        self.rpc._freqtrade.update_trade_state = Mock()
        self.bot._submit_entry = Mock()
        result = self.management.replace_entry(self.bot, trade.id, snap['revision'], 99, 'a' * 24)
        self.assertIn('成交', result['result']['message'])
        self.bot._submit_entry.assert_not_called()
        self.rpc._freqtrade.update_trade_state.assert_called_once()

    def test_unknown_cancel_is_durable_and_never_retried(self):
        trade = self.prepare(pending=True)
        snap = self.management.snapshot(self.bot, trade.id)
        cancel = self.exchange.cancel_order_with_result
        cancel.side_effect = OSError('timeout')
        self.bot._submit_entry = Mock()
        first = self.management.replace_entry(self.bot, trade.id, snap['revision'], 99, 'b' * 24)
        second = self.management.replace_entry(self.bot, trade.id, snap['revision'], 99, 'b' * 24)
        self.assertEqual(first['state'], 'unknown')
        self.assertEqual(second['state'], 'unknown')
        self.assertEqual(cancel.call_count, 1)
        self.bot._submit_entry.assert_not_called()

    def test_replacement_keeps_deadline_and_risk_submission_path(self):
        trade = self.prepare(pending=True)
        snap = self.management.snapshot(self.bot, trade.id)
        deadline = trade.orders[0].order_date_utc + timedelta(seconds=3600)

        def remove(*args):
            self.Trade.session.delete(trade)
            self.Trade.commit()

        self.rpc._freqtrade.handle_cancel_enter = Mock(side_effect=remove)
        self.bot._submit_entry = Mock(return_value={'id': 2, 'order_ids': ['new']})
        result = self.management.replace_entry(self.bot, trade.id, snap['revision'], 99, 'c' * 24)
        self.assertEqual(result['state'], 'submitted')
        request = self.bot._submit_entry.call_args.args[0]
        self.assertEqual(request['entry_deadline'], int(deadline.timestamp()))
        self.assertEqual(request['size'], 100)
        self.assertEqual(request['leverage'], 3)
        self.assertEqual(request['entry_price'], 99)
        self.assertEqual(self.bot._submit_entry.call_args.kwargs['request_ref'], 'c' * 24)

    def test_unknown_full_close_blocks_a_new_confirmation(self):
        trade = self.prepare()
        snap = self.management.snapshot(self.bot, trade.id)
        self.bot._close_position = Mock(side_effect=OSError('accepted but response lost'))
        with self.assertRaises(OSError):
            self.management.close_or_cancel(self.bot, trade.id, snap['revision'], 'close')
        fresh = self.management.snapshot(self.bot, trade.id)
        with self.assertRaises(ValueError):
            self.management.close_or_cancel(self.bot, trade.id, fresh['revision'], 'close')
        self.assertEqual(self.bot._close_position.call_count, 1)
        self.assertIn('attempt_at', self.strategy._execution(trade)['full_exit'])

    def test_cancel_tiny_positive_fill_uses_native_accounting_not_zero_cleanup(self):
        trade = self.prepare(pending=True)
        snap = self.management.snapshot(self.bot, trade.id)
        self.exchange.cancel_order_with_result.side_effect = lambda *a: dict(
            self.receipt, status='canceled', filled=1e-12
        )
        self.rpc._freqtrade.update_trade_state = Mock()
        self.rpc._freqtrade.handle_cancel_enter = Mock()
        self.management.close_or_cancel(self.bot, trade.id, snap['revision'], 'cancel')
        self.rpc._freqtrade.update_trade_state.assert_called_once()
        self.rpc._freqtrade.handle_cancel_enter.assert_not_called()

    def test_replacement_absolute_deadline_survives_order_timestamp_change(self):
        from entry_lifecycle import entry_ttl

        trade = self.prepare(pending=True)
        order = trade.orders[0]
        until = int(order.order_date_utc.timestamp() + 3600)
        order.ft_order_tag = f'manual_long|ttl=3500|until={until}|req=' + 'd' * 24
        order.order_date += timedelta(seconds=300)
        self.assertEqual(order.order_date_utc.timestamp() + entry_ttl(order), until)

    def test_final_create_boundary_rejects_expired_replacement_even_without_risk_limits(self):
        from entry_risk import install_risk_guard, EntryRiskError

        trade = self.prepare(pending=True)
        ref = 'e' * 24
        request = dict(self.request, entry_deadline=1)
        self.strategy.entry_requests.reserve(ref, request)
        self.strategy.entry_requests.set_status(ref, 'submitting', {})
        original = Mock()
        self.exchange.create_order = original
        install_risk_guard(self.strategy, self.rpc._freqtrade)
        with self.strategy.risk_request(ref), self.assertRaises(EntryRiskError):
            self.exchange.create_order(pair=trade.pair, side='buy', reduceOnly=False)
        original.assert_not_called()

    def test_future_tp_edit_after_json_write_survives_restart_before_native_plan_write(self):
        from unittest.mock import patch

        trade = self.prepare()
        state = {
            'stage': 2,
            'tp_filled': {'1': 3, '2': 3.5},
            'protected_stop': 110,
            'tp2_completed_at': 123,
        }
        self.strategy._save_execution(trade, state)
        snap = self.management.snapshot(self.bot, trade.id)
        original = self.strategy._plan
        with patch.object(self.strategy, '_plan', side_effect=OSError('crash after JSON save')):
            with self.assertRaises(OSError):
                self.management.edit_plan(self.bot, trade.id, snap['revision'], {'tp3': 140})
        self.assertEqual(original(trade)['exit_points'], [110, 120, 140])
        self.assertEqual(self.strategy._execution(trade), state)

    def test_list_has_paging_scope_and_empty_state(self):
        from freqtrade.rpc.order_panel import order_list

        self.prepare(pending=True)
        for _ in range(6):
            self.position()
        first = order_list(self.bot, 'all', 0)
        second = order_list(self.bot, 'all', 1)
        self.assertEqual(
            (len(first['rows']), len(second['rows']), first['pages'], first['total']), (6, 1, 2, 7)
        )
        self.assertEqual(len({r['id'] for r in first['rows'] + second['rows']}), 7)
        self.assertEqual(order_list(self.bot, 'pending', 0)['total'], 1)
        self.assertEqual(order_list(self.bot, 'position', 0)['total'], 6)
        self.Trade.session.query(self.Trade).update({'is_open': False})
        self.Trade.commit()
        empty = order_list(self.bot, 'all', 0)
        self.assertEqual(
            (empty['rows'], empty['page'], empty['pages'], empty['total']), ([], 0, 1, 0)
        )
        self.assertEqual(
            (empty['positions'], empty['profit_total'], empty['profit_missing']), (0, 0, 0)
        )

    def test_deadline_is_rechecked_after_native_leverage_setup(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import entry_risk

        trade = self.prepare(pending=True)
        ref = 'f' * 24
        self.strategy.entry_requests.reserve(ref, dict(self.request, entry_deadline=100))
        self.strategy.entry_requests.set_status(ref, 'submitting', {})
        now = [90]
        api_create = Mock(return_value={'id': 'should-not-exist'})
        self.exchange._api = SimpleNamespace(create_order=api_create)

        def native_create(**kwargs):
            now[0] = 110  # Native _lev_prep has network round trips.
            return self.exchange._api.create_order('BTC/USDT:USDT', 'limit', 'buy', 3, 99)

        self.exchange.create_order = native_create
        entry_risk.install_risk_guard(self.strategy, self.rpc._freqtrade)
        with self.strategy.risk_request(ref), patch.object(
            entry_risk.time, 'time', side_effect=lambda: now[0]
        ):
            with self.assertRaises(entry_risk.EntryRiskError):
                self.exchange.create_order(pair=trade.pair, side='buy', reduceOnly=False)
        api_create.assert_not_called()
