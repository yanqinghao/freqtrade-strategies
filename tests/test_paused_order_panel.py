import asyncio
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import test_telegram_orders as orders
from test_order_panel import update


class PausedOrderTests(unittest.TestCase):
    setUp = orders.TelegramOrderTests.setUp
    position = orders.TelegramOrderTests.position

    def pause(self, ref='a' * 24, phase='paused', **changes):
        from entry_pause import EntryPause

        self.strategy.entry_pause = EntryPause(self.strategy, self.rpc._freqtrade, None)
        meta = dict(
            ref=ref,
            attempt_ref=ref,
            phase=phase,
            decision='waiting_continuity',
            created=1800000000,
            deadline=1800086400,
            quiet_after=1800000000,
            request=dict(self.request, entry_price=100, ttl_seconds=86400),
            attempts=0,
        )
        meta.update(changes)
        self.strategy.entry_requests.reserve(ref, meta['request'])
        self.strategy.entry_requests.set_status(ref, 'submitted', {'pause_ref': ref})
        self.strategy.entry_requests.put_pause(meta)
        return meta

    def panel_module(self):
        from freqtrade.rpc import order_panel

        return order_panel

    def test_paused_plan_is_visible_without_trade_or_pnl_and_filters_share_totals(self):
        self.pause()
        panel = self.panel_module()
        for scope in ('all', 'pending', 'paused'):
            data = panel.order_list(self.bot, scope, 0)
            self.assertEqual(data['total'], 1)
            row = data['rows'][0]
            self.assertEqual(row['pause_ref'], 'a' * 24)
            self.assertIn('暂停', row['label'])
            self.assertIsNone(row['profit_abs'])
            self.assertEqual((data['positions'], data['profit_total']), (0, 0))
        self.assertEqual(panel.order_list(self.bot, 'position', 0)['total'], 0)

    def test_paused_paging_does_not_duplicate_linked_trade_or_change_position_profit(self):
        from freqtrade.rpc import order_management as management

        trade = self.position()
        from freqtrade.persistence import Order

        trade.orders.append(
            Order(
                order_id='linked-entry',
                ft_order_side=trade.entry_side,
                ft_pair=trade.pair,
                symbol=trade.pair,
                side=trade.entry_side,
                ft_is_open=False,
                status='closed',
                order_type='limit',
                amount=10,
                filled=10,
                remaining=0,
                price=100,
                ft_amount=10,
                ft_price=100,
                ft_order_tag='manual_long|req=' + 'a' * 24,
            )
        )
        self.Trade.commit()
        self.exchange.get_rate = lambda *a, **kw: 105
        self.pause(trade_id=trade.id, order_id='linked-entry')
        for i in range(1, 8):
            self.pause(f'{i:024x}')
        panel = self.panel_module()
        one, two = panel.order_list(self.bot, 'all', 0), panel.order_list(self.bot, 'all', 1)
        self.assertEqual((one['total'], len(one['rows']), len(two['rows'])), (8, 6, 2))
        self.assertEqual(len({r['id'] for r in one['rows'] + two['rows']}), 8)
        for data in (one, two, panel.order_list(self.bot, 'paused', 0)):
            self.assertEqual(data['positions'], 1)
            self.assertAlmostEqual(data['profit_total'], trade.calculate_profit(105).profit_abs)
        self.assertEqual(management.snapshot(self.bot, trade.id)['pause_ref'], 'a' * 24)

    def test_cancel_is_durable_and_does_not_submit_close_or_delete_plan_parameters(self):
        from freqtrade.rpc import order_management as management
        from entry_requests import EntryRequests

        self.pause()
        data = management.pause_snapshot(self.bot, 'a' * 24)
        original = deepcopy(data['meta']['request'])
        result = management.cancel_pause(self.bot, 'a' * 24, data['revision'])
        self.assertIn('取消', result)
        restored = EntryRequests(self.strategy.entry_requests.path).get_pause('a' * 24)
        self.assertEqual(restored['decision'], 'cancelled')
        self.assertEqual(restored['request'], original)
        self.assertEqual(self.panel_module().order_list(self.bot, 'all', 0)['total'], 0)
        self.assertEqual(self.Trade.get_open_trade_count(), 0)

    def test_market_diagnostic_updates_do_not_invalidate_cancel_but_resume_does(self):
        from freqtrade.rpc import order_management as management

        meta = self.pause()
        snap = management.pause_snapshot(self.bot, meta['ref'])
        meta.update(
            decision='waiting_data',
            checked_until=1800000090,
            check_error=dict(reason='缺少当前行情', at=1800000090, stage='恢复条件'),
        )
        self.strategy.entry_requests.put_pause(meta)
        self.assertEqual(
            management.pause_snapshot(self.bot, meta['ref'])['revision'], snap['revision']
        )
        meta.update(phase='watching', order_id='new-order', trade_id=987)
        self.strategy.entry_requests.put_pause(meta)
        with self.assertRaises(ValueError):
            management.cancel_pause(self.bot, meta['ref'], snap['revision'])
        self.assertNotEqual(
            self.strategy.entry_requests.get_pause(meta['ref'])['decision'], 'cancelled'
        )

    def test_unknown_receipt_cancel_only_stops_future_restoration_and_remains_visible(self):
        from freqtrade.rpc import order_management as management

        meta = self.pause(phase='unknown')
        self.strategy.entry_requests.set_status(meta['ref'], 'unknown', {'message': '回执未知'})
        snap = management.pause_snapshot(self.bot, meta['ref'])
        self.assertTrue(snap['uncertain'])
        result = management.cancel_pause(self.bot, meta['ref'], snap['revision'])
        self.assertIn('核对', result)
        self.assertNotIn('已撤单', result)
        self.assertEqual(
            self.strategy.entry_requests.get_pause(meta['ref'])['decision'], 'cancelled'
        )
        self.assertEqual(self.panel_module().order_list(self.bot, 'all', 0)['total'], 1)

    def test_detail_cancel_requires_confirmation_and_consumes_token_once(self):
        self.pause()
        self.bot._app = SimpleNamespace(
            bot=SimpleNamespace(send_message=AsyncMock(return_value=SimpleNamespace(message_id=99)))
        )
        self.bot._await_command = AsyncMock(side_effect=lambda fn, *args: fn(*args))
        self.bot._manual_order_lock = asyncio.Lock()
        u = update(query=True)
        asyncio.run(self.bot._panel_pause_detail(u, 'a' * 24))
        view = u.callback_query.edit_message_text.call_args.kwargs
        self.assertIn('100', view['text'])
        self.assertIn('未挂', view['text'])
        button = next(
            b for row in view['reply_markup'].inline_keyboard for b in row if b.text == '取消计划'
        )
        u.callback_query.data = button.callback_data
        asyncio.run(self.bot._panel_callback.__wrapped__(self.bot, u, None))
        self.assertNotEqual(
            self.strategy.entry_requests.get_pause('a' * 24)['decision'], 'cancelled'
        )
        view = u.callback_query.edit_message_text.call_args.kwargs
        confirm = next(
            b for row in view['reply_markup'].inline_keyboard for b in row if b.text == '确认取消'
        )
        u.callback_query.data = confirm.callback_data
        asyncio.run(self.bot._panel_callback.__wrapped__(self.bot, u, None))
        asyncio.run(self.bot._panel_callback.__wrapped__(self.bot, u, None))
        self.assertEqual(self.strategy.entry_requests.get_pause('a' * 24)['decision'], 'cancelled')
        cancellations = [
            call
            for call in self.bot._await_command.call_args_list
            if call.args[0].__name__ == 'cancel_pause'
        ]
        self.assertEqual(len(cancellations), 1)

    def test_observed_fill_pending_accounting_invalidates_cancel_confirmation(self):
        import test_order_management as native_tests
        from freqtrade.rpc import order_management as management

        trade = native_tests.OrderManagementTests.prepare(self, pending=True)
        meta = self.pause(phase='canceling', trade_id=trade.id, order_id='entry-1')
        snap = management.pause_snapshot(self.bot, meta['ref'])
        self.strategy._save_execution(trade, dict(entry_observed={'entry-1': 1}))
        with self.assertRaises(ValueError):
            management.cancel_pause(self.bot, meta['ref'], snap['revision'])
        self.assertNotEqual(
            self.strategy.entry_requests.get_pause(meta['ref'])['decision'], 'cancelled'
        )

    def test_reused_trade_id_cannot_hide_pause_or_link_an_unrelated_position(self):
        from freqtrade.rpc import order_management as management

        for short in (False, True):
            with self.subTest(short=short):
                if short:
                    self.doCleanups()
                    self.setUp()
                old = self.position()
                old_id = old.id
                meta = self.pause(trade_id=old_id, order_id='deleted-entry')
                self.Trade.session.delete(old)
                self.Trade.commit()
                other = self.position(short)
                self.assertEqual(other.id, old_id, 'fixture must exercise actual SQLite id reuse')
                self.exchange.get_rate = lambda *a, **kw: 105
                data = self.panel_module().order_list(self.bot, 'all', 0)
                self.assertEqual(data['total'], 2)
                self.assertIsNone(management.pause_snapshot(self.bot, meta['ref'])['trade_id'])
                self.assertIsNone(management.snapshot(self.bot, other.id)['pause_ref'])
                snap = management.pause_snapshot(self.bot, meta['ref'])
                management.cancel_pause(self.bot, meta['ref'], snap['revision'])
                self.assertEqual(self.Trade.get_open_trade_count(), 1)
                self.assertEqual(other.amount, 10)
