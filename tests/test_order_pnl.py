import asyncio
import math
import unittest

import test_order_management as order_tests
import test_order_panel as panel_tests


class OrderPnlTests(unittest.TestCase):
    setUp = order_tests.OrderManagementTests.setUp
    position = order_tests.OrderManagementTests.position
    prepare = order_tests.OrderManagementTests.prepare

    def listing(self, scope='all', page=0):
        from freqtrade.rpc.order_panel import order_list

        return order_list(self.bot, scope, page)

    def test_native_long_short_fee_aware_amount_and_ratio_exclude_realized(self):
        from freqtrade.rpc import order_management

        for short in (False, True):
            trade = self.prepare(short=short)
            trade.realized_profit = 123
            rate = 105 if not short else 95
            self.exchange.get_rate.return_value = rate
            expected = trade.calculate_profit(rate)
            row = self.listing()['rows'][0]
            self.assertAlmostEqual(row['profit_abs'], expected.profit_abs)
            self.assertAlmostEqual(row['profit'], expected.profit_ratio * 100)
            self.assertNotEqual(row['profit_abs'], expected.total_profit)
            detail = order_management.snapshot(self.bot, trade.id)
            self.assertEqual(detail['profit_abs'], row['profit_abs'])
            self.assertEqual(detail['profit'], row['profit'])
            trade.is_open = False
            self.Trade.commit()

    def test_total_covers_all_pages_and_does_not_change_with_filter(self):
        self.prepare(pending=True)
        expected = 0
        for i in range(7):
            trade = self.position(is_short=i % 2 == 1)
            trade.amount = i + 1
            trade.recalc_open_trade_value()
            expected += trade.calculate_profit(102).profit_abs
        self.Trade.commit()
        self.exchange.get_rate.return_value = 102
        first, second, pending = self.listing(), self.listing(page=1), self.listing('pending')
        self.assertEqual((len(first['rows']), len(second['rows'])), (6, 2))
        for data in (first, second, pending):
            self.assertAlmostEqual(data['profit_total'], expected)
            self.assertEqual(data['profit_missing'], 0)
            self.assertEqual(data['positions'], 7)
        row = pending['rows'][0]
        self.assertIsNone(row['profit'])
        self.assertIsNone(row['profit_abs'])
        self.assertEqual(row['pnl_status'], '未成交')
        self.assertEqual(self.exchange.get_rate.call_count, 21)

    def test_unavailable_and_nonfinite_quotes_never_become_zero(self):
        self.prepare()
        self.position(is_short=True)
        self.exchange.get_rate.side_effect = [101, OSError('timeout')]
        data = self.listing()
        self.assertEqual(data['profit_missing'], 1)
        self.assertEqual(sum(r['profit_abs'] is None for r in data['rows']), 1)
        self.assertNotEqual(data['profit_total'], 0)
        for value in (math.nan, math.inf, 0):
            self.exchange.get_rate.side_effect = None
            self.exchange.get_rate.return_value = value
            data = self.listing()
            self.assertEqual(data['profit_missing'], 2)
            self.assertTrue(all(r['profit_abs'] is None for r in data['rows']))

    def test_pending_does_not_fetch_quote_and_partial_uses_filled_position(self):
        trade = self.prepare(pending=True)
        self.assertEqual(self.listing()['positions'], 0)
        self.exchange.get_rate.assert_not_called()
        trade.amount = 0.5
        trade.recalc_open_trade_value()
        self.Trade.commit()
        self.exchange.get_rate.return_value = 105
        data = self.listing()
        self.assertEqual(data['positions'], 1)
        self.assertIn('部分成交', data['rows'][0]['label'])
        self.assertEqual(data['profit_total'], trade.calculate_profit(105).profit_abs)

    def test_observed_unaccounted_fill_is_not_classified_as_unfilled(self):
        trade = self.prepare(pending=True)
        self.strategy._save_execution(trade, {'entry_observed': {'entry-1': 0.2}})
        data = self.listing()
        self.assertEqual(data['positions'], 1)
        self.assertEqual(data['profit_missing'], 1)
        self.assertEqual(data['rows'][0]['pnl_status'], '成交待记账')
        self.assertIsNone(data['rows'][0]['profit_abs'])


class OrderPnlUiTests(unittest.TestCase):
    setUp = panel_tests.OrderPanelTests.setUp

    def test_rows_total_timestamp_and_failure_are_visible_with_short_buttons(self):
        self.bot._await_command.return_value = dict(
            rows=[
                dict(id=1, label='#1 BTC 多 · 持仓', profit=5.25, profit_abs=2.3456, pnl_status='可用')
            ],
            page=0,
            pages=2,
            total=7,
            profit_total=-4.321,
            profit_missing=0,
            positions=7,
            checked='11-06 12:00:00 UTC',
        )
        asyncio.run(self.bot._panel_list(panel_tests.update()))
        message = self.bot._app.bot.send_message.call_args.kwargs
        self.assertIn('+2.3456 USDT', message['text'])
        self.assertIn('+5.25%', message['text'])
        self.assertIn('全部持仓浮盈亏：-4.3210 USDT', message['text'])
        self.assertIn('跨页', message['text'])
        self.assertIn('12:00:00 UTC', message['text'])
        labels = [b.text for row in message['reply_markup'].inline_keyboard for b in row]
        self.assertIn('#1 BTC 多 · 持仓', labels)
        self.assertIn('下一页', labels)
        self.bot._await_command.return_value['profit_missing'] = 1
        asyncio.run(self.bot._panel_list(panel_tests.update()))
        text = self.bot._app.bot.send_message.call_args.kwargs['text']
        self.assertIn('汇总不完整', text)
        self.assertIn('1 笔', text)
        self.assertNotIn('全部持仓浮盈亏：', text)


class OrderPnlPrecisionTests(unittest.TestCase):
    setUp = OrderPnlTests.setUp
    position = OrderPnlTests.position
    prepare = OrderPnlTests.prepare

    def test_display_accepts_rounding_noise_but_flags_real_unaccounted_fill(self):
        from freqtrade.rpc.order_management import position_profit

        trade = self.prepare(pending=True)
        trade.amount = 0.3
        trade.recalc_open_trade_value()
        self.strategy._save_execution(trade, {'entry_observed': {'entry-1': 0.30000000000000004}})
        self.assertEqual(position_profit(trade, 105)['pnl_status'], '可用')
        self.strategy._save_execution(trade, {'entry_observed': {'entry-1': 0.31}})
        self.assertEqual(position_profit(trade, 105)['pnl_status'], '成交待记账')

    def test_invalid_native_results_and_percentage_overflow_are_unavailable(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from freqtrade.rpc.order_management import position_profit

        trade = self.prepare()
        for amount, ratio in ((math.nan, 0.1), (1, math.inf), (1, 1e308)):
            with patch.object(
                trade,
                'calculate_profit',
                return_value=SimpleNamespace(profit_abs=amount, profit_ratio=ratio),
            ):
                data = position_profit(trade, 105)
                self.assertIsNone(data['profit_abs'])
                self.assertIsNone(data['profit'])
