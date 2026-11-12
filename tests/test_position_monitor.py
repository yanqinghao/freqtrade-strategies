import unittest
from copy import deepcopy
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Event
from types import SimpleNamespace

import test_telegram_orders as order_tests


class ImmediateExecutor:
    def __init__(self):
        self.calls = []

    def submit(self, fn, *args):
        self.calls.append(args)
        future = Future()
        future.set_result(fn(*args))
        return future

    def shutdown(self, **kwargs):
        pass


class PositionMonitorTests(unittest.TestCase):
    setUp = order_tests.TelegramOrderTests.setUp
    position = order_tests.TelegramOrderTests.position

    def prepare(self, short=False, pair='XRP/USDT:USDT'):
        import position_monitor

        self.module = position_monitor
        self.now = datetime(2026, 11, 12, 12, tzinfo=timezone.utc)
        trade = self.position(is_short=short)
        trade.pair = pair
        trade.recalc_open_trade_value()
        plan = dict(
            direction='short' if short else 'long',
            trade_id=trade.id,
            stop_loss=110 if short else 90,
            exit_points=[90, 80, 70] if short else [110, 120, 130],
        )
        self.strategy.set_manual_plan(pair, plan)
        trade.set_custom_data('assistant_plan', plan)
        self.strategy._save_execution(
            trade,
            dict(
                stage=0,
                protection=dict(
                    status='confirmed',
                    stop_price=plan['stop_loss'],
                    covered_amount=10,
                    checked_at=self.now.timestamp(),
                ),
            ),
        )
        self.Trade.commit()
        return trade

    def market(self, direction='偏多', bid=105, ask=106):
        widths = {'1d': 86400, '4h': 14400, '1h': 3600, '15m': 900}
        return dict(
            frames={
                tf: dict(
                    trend=direction,
                    close=104,
                    ema20=103,
                    ema50=102,
                    support=95,
                    resistance=115,
                    closed_at=datetime.fromtimestamp(
                        int(self.now.timestamp()) // width * width, timezone.utc
                    ),
                )
                for tf, width in widths.items()
            },
            quote=dict(bid=bid, ask=ask, observed_at=self.now.timestamp()),
        )

    def render(self, trade, market=None):
        return self.module.build_position_report(
            self.strategy, trade, self.market() if market is None else market, self.now
        )

    def candles(self, tf):
        width = {'1d': 86400, '4h': 14400, '1h': 3600, '15m': 900}[tf]
        end = int(self.now.timestamp()) // width * width
        data = []
        for i in range(121):
            price = 100 + (i - 119) * 0.01
            data.append(
                [(end - (120 - i) * width) * 1000, price, price + 0.02, price - 0.02, price, 10]
            )
        data[-1][1:5] = [10000, 10001, 9999, 10000]
        return data

    def monitor(self, executor=None, fetch=None):
        monitor = self.module.PositionMonitor(
            fetch or (lambda pair, tf: self.candles(tf)),
            clock=lambda: self.now.timestamp(),
            executor=executor or ImmediateExecutor(),
        )
        monitor._client = SimpleNamespace(
            fetch_order_book=lambda *a, **kw: dict(bids=[[105, 1]], asks=[[106, 1]])
        )
        self.addCleanup(monitor.close)
        return monitor

    def test_remaining_native_pnl_uses_correct_exit_book_side_and_excludes_realized(self):
        for short, bid, ask, expected in [
            (False, 105, 106, '+48.9750 USDT'),
            (True, 94, 95, '+49.0250 USDT'),
        ]:
            with self.subTest(short=short):
                trade = self.prepare(
                    short=short, pair='DOGE/USDT:USDT' if short else 'XRP/USDT:USDT'
                )
                trade.realized_profit = 123
                state = deepcopy(self.strategy._execution(trade))
                plan = deepcopy(self.strategy.manual_open)
                text = self.render(trade, self.market('偏空' if short else '偏多', bid, ask))
                self.assertIn(expected, text)
                self.assertIn('收益率', text)
                self.assertIn(trade.pair, text)
                self.assertIn('持有', text)
                self.assertEqual(self.strategy._execution(trade), state)
                self.assertEqual(self.strategy.manual_open, plan)

    def test_opposite_trends_reduce_advice_and_favorable_large_trend_describes_pullback(self):
        trade = self.prepare()
        market = self.market('偏空')
        self.assertIn('减仓', self.render(trade, market))
        market['frames']['1d']['trend'] = market['frames']['4h']['trend'] = '偏多'
        self.assertIn('回调', self.render(trade, market))
        for frame in market['frames'].values():
            frame['trend'] = '震荡'
        self.assertIn('震荡', self.render(trade, market))
        trade.is_short = True
        plan = self.strategy.manual_open[trade.pair]
        plan.update(direction='short', stop_loss=110, exit_points=[90, 80, 70])
        trade.set_custom_data('assistant_plan', plan)
        self.strategy._save_execution(
            trade,
            dict(
                stage=0,
                protection=dict(
                    status='confirmed',
                    stop_price=110,
                    covered_amount=10,
                    checked_at=self.now.timestamp(),
                ),
            ),
        )
        self.assertIn('减仓', self.render(trade, self.market()))

    def test_original_exit_progress_stop_and_tp_take_priority_over_direction(self):
        trade = self.prepare()
        state = self.strategy._execution(trade)
        state['full_exit'] = {'reason': 'already exiting'}
        self.strategy._save_execution(trade, state)
        self.assertIn('退出', self.render(trade))
        self.assertNotIn('建议：顺势持有', self.render(trade))
        state.pop('full_exit')
        self.strategy._save_execution(trade, state)
        self.assertIn('TP1', self.render(trade, self.market(bid=111, ask=112)))
        self.assertIn('核对', self.render(trade, self.market(bid=89, ask=90)))
        text = self.render(trade, self.market(bid=94, ask=95))
        self.assertIn('盘中', text)
        self.assertNotIn('建议：顺势持有', text)

    def test_missing_stale_data_unconfirmed_protection_and_pending_plan_withhold_directional_advice(
        self,
    ):
        trade = self.prepare()
        for change in (
            'missing_quote',
            'stale_quote',
            'missing_frame',
            'stale_frame',
            'invalid_book',
        ):
            market = self.market()
            if change == 'missing_quote':
                market.pop('quote')
            if change == 'stale_quote':
                market['quote']['observed_at'] -= 31
            if change == 'missing_frame':
                market['frames'].pop('4h')
            if change == 'stale_frame':
                market['frames']['15m']['closed_at'] -= timedelta(minutes=15)
            if change == 'invalid_book':
                market['quote']['bid'] = float('nan')
            with self.subTest(change=change):
                text = self.render(trade, market)
                self.assertNotIn('建议：顺势持有', text)
                if change in ('missing_quote', 'stale_quote', 'invalid_book'):
                    self.assertNotIn('+48.9750 USDT', text)
                    self.assertIn('不可用', text)
        self.strategy._save_execution(trade, {'stage': 0, 'protection': {'status': 'unconfirmed'}})
        self.assertIn('核对', self.render(trade))
        self.assertNotIn('建议：顺势持有', self.render(trade))
        self.strategy.manual_open[trade.pair]['exit_points'] = [111, 121, 131]
        self.assertIn('待应用', self.render(trade))

    def test_observed_fill_pending_accounting_never_reports_zero_or_directional_action(self):
        trade = self.prepare()
        trade.amount = 0
        self.strategy._save_execution(trade, dict(stage=0, entry_observed={'pending': 1}))
        # Use an actual order so observed quantity comes through the shared accounting helper.
        import test_entry_risk as risk_tests

        risk_tests.EntryRiskTests.order(self, trade, amount=2, filled=0)
        self.strategy._save_execution(
            trade, dict(stage=0, entry_observed={str(trade.orders[0].order_id): 1})
        )
        text = self.render(trade)
        self.assertIn('成交待记账', text)
        self.assertIn('已观察数量 1', text)
        self.assertNotIn('未成交', text)
        self.assertNotIn('+0.0000 USDT', text)
        self.assertNotIn('建议：顺势持有', text)

    def test_public_collection_ignores_forming_candles_and_validates_closed_history(self):
        trade = self.prepare()
        monitor = self.monitor()
        data = monitor.collect(((trade.id, trade.pair),), int(self.now.timestamp()) // 3600)
        market = data['markets'][trade.pair]
        self.assertEqual(market['frames']['4h']['close'], 100)
        self.assertEqual(market['frames']['4h']['trend'], '偏多')
        self.assertLess(market['frames']['4h']['resistance'], 1000)

        def gap(pair, tf):
            rows = self.candles(tf)
            return rows[:50] + rows[51:]

        broken = self.monitor(fetch=gap).collect(
            ((trade.id, trade.pair),), int(self.now.timestamp()) // 3600
        )
        self.assertEqual(broken['markets'][trade.pair]['frames'], {})

    def test_one_symbol_failure_preserves_other_position_report(self):
        trade = self.prepare()

        def fetch(pair, tf):
            if pair.startswith('BAD/'):
                raise TimeoutError('unavailable')
            return self.candles(tf)

        monitor = self.monitor(fetch=fetch)
        data = monitor.collect(
            ((trade.id, trade.pair), (99, 'BAD/USDT:USDT')), int(self.now.timestamp()) // 3600
        )
        self.assertEqual(len(data['markets'][trade.pair]['frames']), 4)
        self.assertEqual(data['markets']['BAD/USDT:USDT']['frames'], {})

    def test_hourly_delivery_is_deduplicated_across_restart_and_expires(self):
        trade = self.prepare()
        monitor = self.monitor()
        monitor.tick(self.now, self.strategy, [trade])
        monitor.tick(self.now, self.strategy, [trade])
        pending = self.strategy.alert_queue.pending(now=self.now.timestamp())
        self.assertEqual(len(pending), 1)
        self.assertIn('持仓简报', pending[0]['message'])
        self.strategy.alert_queue.acknowledge(pending[0])
        restarted = self.monitor()
        restarted.tick(self.now, self.strategy, [trade])
        restarted.tick(self.now, self.strategy, [trade])
        self.assertEqual(self.strategy.alert_queue.pending(now=self.now.timestamp()), [])
        self.now += timedelta(hours=1)
        monitor.tick(self.now, self.strategy, [trade])
        monitor.tick(self.now, self.strategy, [trade])
        self.assertEqual(len(self.strategy.alert_queue.pending(now=self.now.timestamp())), 1)
        self.assertEqual(self.strategy.alert_queue.pending(now=self.now.timestamp() + 301), [])

    def test_empty_unfilled_and_closed_positions_do_not_schedule_public_io(self):
        trade = self.prepare()
        executor = ImmediateExecutor()
        monitor = self.monitor(executor=executor)
        trade.amount = 0
        monitor.tick(self.now, self.strategy, [trade])
        monitor.tick(self.now, self.strategy, [])
        trade.amount = 10
        trade.is_open = False
        monitor.tick(self.now, self.strategy, [trade])
        self.assertEqual(executor.calls, [])

    def test_slow_worker_never_blocks_loop_and_closed_trade_is_filtered_on_return(self):
        trade = self.prepare()
        started, release = Event(), Event()

        def fetch(pair, tf):
            started.set()
            release.wait(2)
            return self.candles(tf)

        executor = ThreadPoolExecutor(max_workers=1)
        monitor = self.monitor(executor=executor, fetch=fetch)
        try:
            monitor.tick(self.now, self.strategy, [trade])
            self.assertTrue(started.wait(1))
            monitor.tick(self.now, self.strategy, [trade])
            self.assertFalse(monitor._future.done())
            trade.is_open = False
            release.set()
            monitor._future.result(timeout=2)
            replacement = self.position()
            replacement.pair = trade.pair
            monitor.tick(self.now, self.strategy, [replacement])
            self.assertEqual(self.strategy.alert_queue.pending(now=self.now.timestamp()), [])
        finally:
            release.set()
            executor.shutdown(wait=True)

    def test_position_scheduled_before_hour_boundary_is_not_delivered_after_it(self):
        trade = self.prepare()
        monitor = self.monitor()
        monitor.tick(self.now, self.strategy, [trade])
        self.now += timedelta(hours=1)
        monitor.tick(self.now, self.strategy, [trade])
        self.assertEqual(self.strategy.alert_queue.pending(now=self.now.timestamp()), [])

    def test_strategy_loop_drives_reporter_without_mutating_position_or_plan(self):
        trade = self.prepare()
        monitor = self.monitor()
        self.strategy.position_monitor = monitor
        original = (
            trade.amount,
            deepcopy(self.strategy.manual_open),
            self.strategy._execution(trade),
        )
        self.strategy.bot_loop_start(self.now)
        self.strategy.bot_loop_start(self.now)
        rows = self.strategy.alert_queue.pending(now=self.now.timestamp())
        self.assertEqual(len(rows), 1, 'strategy loop did not publish held-position report')
        self.assertIn(trade.pair, rows[0]['message'])
        self.assertEqual(
            (trade.amount, self.strategy.manual_open, self.strategy._execution(trade)), original
        )

    def test_each_report_uses_current_time_after_previous_enqueue(self):
        first = self.prepare()
        second = self.prepare(pair='SOL/USDT:USDT')
        monitor = self.monitor()
        monitor.tick(self.now, self.strategy, [first, second])
        put = self.strategy.alert_queue.put_market_once

        def slow_enqueue(*args, **kwargs):
            result = put(*args, **kwargs)
            self.now += timedelta(seconds=31)
            return result

        self.strategy.alert_queue.put_market_once = slow_enqueue
        monitor.tick(self.now, self.strategy, [first, second])
        rows = self.strategy.alert_queue.pending(now=self.now.timestamp())
        self.assertEqual(len(rows), 2)
        second_message = next(row['message'] for row in rows if 'SOL/USDT:USDT' in row['message'])
        self.assertIn('浮盈亏不可用', second_message)
        self.assertNotIn('建议：顺势持有', second_message)

    def test_extension_advice_uses_active_tp3_while_preserving_original_targets(self):
        for short, stop, target, between, beyond in [
            (False, 120, 150, 135, 151),
            (True, 80, 50, 65, 49),
        ]:
            with self.subTest(short=short):
                trade = self.prepare(
                    short=short, pair='DOGE/USDT:USDT' if short else 'XRP/USDT:USDT'
                )
                state = self.strategy._execution(trade)
                state.update(stage=2, protected_stop=stop, extension=dict(stop=stop, target=target))
                state['protection']['stop_price'] = stop
                self.strategy._save_execution(trade, state)
                direction = '偏空' if short else '偏多'
                market = self.market(
                    direction,
                    bid=between - 1 if short else between,
                    ask=between if short else between + 1,
                )
                text = self.render(trade, market)
                self.assertIn('延伸已生效', text)
                self.assertNotIn('已到', text.split('建议：')[-1])
                market = self.market(
                    direction,
                    bid=beyond - 1 if short else beyond,
                    ask=beyond if short else beyond + 1,
                )
                self.assertIn('当前TP3', self.render(trade, market))
                self.assertEqual(self.strategy._execution(trade), state)
