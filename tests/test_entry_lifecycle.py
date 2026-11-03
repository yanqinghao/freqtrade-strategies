import importlib.util
import sys
import tempfile
import unittest
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

from test_exit_policy import RuleExchange

ROOT = Path(__file__).resolve().parents[1]
sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != ROOT]
sys.path.insert(0, str(ROOT / 'user_data' / 'strategies'))


class EntryLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('entry_lifecycle'), 'entry lifecycle missing')
        import entry_lifecycle
        from freqtrade.persistence import Trade, Order, init_db
        from freqtrade.freqtradebot import FreqtradeBot
        from freqtrade.enums import RunMode, TradingMode
        from KamaFama_Dynamic import KamaFama_Dynamic

        self.module = entry_lifecycle
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        init_db('sqlite:///' + str(Path(self.temp.name) / 'test.sqlite'))
        engine = Trade.session.get_bind()

        def cleanup():
            from freqtrade.persistence.custom_data import _CustomData

            _CustomData.session.remove()
            Trade.session.remove()
            engine.dispose()

        self.addCleanup(cleanup)
        self.Trade, self.Order = Trade, Order
        self.now = datetime(2026, 11, 3, 12, tzinfo=timezone.utc)
        self.strategy = KamaFama_Dynamic(
            {
                'runmode': RunMode.DRY_RUN,
                'user_data_dir': Path(self.temp.name),
                'stake_currency': 'USDT',
                'exchange': {'name': 'binance'},
            }
        )
        self.messages = []
        self.exchange = RuleExchange(minimum=0.001, step='.001', cost=0.0, pair='BTC/USDT:USDT')
        self.strategy.dp = SimpleNamespace(
            _exchange=self.exchange, send_msg=lambda text, **kw: self.messages.append(text)
        )
        self.trade = Trade(
            pair='BTC/USDT:USDT',
            is_short=False,
            is_open=True,
            open_rate=100.0,
            amount=0.0,
            stake_amount=1000.0 / 3,
            leverage=3.0,
            exchange='binance',
            open_date=self.now - timedelta(hours=1),
            enter_tag='manual_long',
            fee_open=0.0005,
            fee_close=0.0005,
            amount_precision=0.001,
            price_precision=0.1,
            precision_mode=4,
            contract_size=1.0,
            trading_mode=TradingMode.FUTURES,
        )
        Trade.session.add(self.trade)
        Trade.commit()
        self.order = self.add_order(
            'entry-1', self.now - timedelta(hours=1), 'manual_long|ttl=14400'
        )
        self.strategy.set_manual_plan(
            self.trade.pair,
            {
                'trade_id': self.trade.id,
                'direction': 'long',
                'stop_loss': 90.0,
                'exit_points': [110.0, 120.0, 130.0],
            },
        )
        self.raw = {
            'id': self.order.order_id,
            'symbol': self.trade.pair,
            'side': 'buy',
            'type': 'limit',
            'status': 'open',
            'amount': 10.0,
            'filled': 0.0,
            'remaining': 10.0,
            'price': 100.0,
            'average': 100.0,
            'cost': 0.0,
            'timestamp': int(self.order.order_date_utc.timestamp() * 1000),
            'info': {},
        }
        self.cancelled = {**deepcopy(self.raw), 'status': 'canceled'}
        self.queries = []
        self.cancels = []
        self.exchange.fetch_order = lambda ident, pair: self.queries.append(ident) or deepcopy(
            self.raw
        )
        self.exchange.cancel_order_with_result = lambda ident, pair, amount: self.cancels.append(
            ident
        ) or deepcopy(self.cancelled)
        self.exchange.check_order_canceled_empty = (
            lambda row: row.get('status') in ('canceled', 'closed', 'expired')
            and row.get('filled') == 0.0
        )
        self.bot = object.__new__(FreqtradeBot)
        self.bot.exchange = self.exchange
        self.bot.strategy = self.strategy
        self.bot._exit_lock = Lock()
        self.bot.wallets = SimpleNamespace(update=lambda: None)
        self.bot.handle_order_fee = lambda *args: None
        self.bot._update_trade_after_fill = (
            lambda trade, order, send: self.strategy.order_filled(
                trade.pair, trade, order, self.now
            )
            or trade
        )
        self.bot.order_close_notify = lambda *args: None
        self.bot._notify_enter_cancel = lambda *args, **kw: None

    def add_order(self, ident, date, tag, side='buy', filled=0.0, status='open'):
        order = self.Order(
            order_id=ident,
            ft_order_side=side,
            ft_is_open=status == 'open',
            ft_pair=self.trade.pair,
            order_type='limit',
            status=status,
            symbol=self.trade.pair,
            side=side,
            amount=10.0,
            filled=filled,
            remaining=10.0 - filled,
            price=100.0,
            average=100.0,
            cost=100.0 * filled,
            ft_amount=10.0,
            ft_price=100.0,
            order_date=date,
            ft_order_tag=tag,
        )
        self.trade.orders.append(order)
        self.Trade.commit()
        return order

    def run_check(self):
        self.Trade.commit()  # Model the native, already-persisted order snapshot.
        self.module.process_entry_orders(self.strategy, self.bot, self.trade, self.now)

    def partial(self, filled=1.0):
        self.order.filled = filled
        self.raw.update(filled=filled, remaining=10.0 - filled, cost=100.0 * filled)
        self.cancelled.update(filled=filled, remaining=10.0 - filled, cost=100.0 * filled)
        self.Trade.commit()

    def test_duration_and_tag_are_bounded_and_per_order(self):
        from test_telegram_assistant import load_telegram

        parser = load_telegram().ttl_seconds
        for text, want in (('1m', 60), ('4h', 14400), ('1d', 86400), ('5d', 432000)):
            self.assertEqual(parser(text), want)
        for text in ('0h', '6d', '-1h', '1.5h', 'nan', '1w'):
            with self.assertRaises(ValueError):
                parser(text)
        self.assertEqual(self.module.entry_ttl(self.order), 14400)
        self.order.ft_order_tag = 'manual_long'
        self.assertIsNone(self.module.entry_ttl(self.order))

    def test_unexpired_order_and_legacy_order_are_not_cancelled(self):
        self.run_check()
        self.assertEqual(self.queries, [])
        self.assertEqual(self.cancels, [])
        self.order.order_date = self.now - timedelta(hours=5)
        self.order.ft_order_tag = 'manual_long'
        self.run_check()
        self.assertEqual(self.cancels, [])

    def test_expiry_cancels_zero_fill_and_cleans_only_owned_plan(self):
        self.order.order_date = self.now - timedelta(hours=5)
        self.run_check()
        self.assertEqual(self.cancels, ['entry-1'])
        self.assertEqual(self.Trade.get_open_trade_count(), 0)
        self.assertNotIn(self.trade.pair, self.strategy.manual_open)
        self.assertTrue(any('到期' in text or '有效期' in text for text in self.messages))

    def test_zero_fill_cancel_does_not_delete_new_plan_for_same_pair(self):
        self.strategy.set_manual_plan(
            self.trade.pair, {'trade_id': 999, 'direction': 'long', 'stop_loss': 80.0}
        )
        self.order.order_date = self.now - timedelta(hours=5)
        self.run_check()
        self.assertEqual(self.strategy.manual_open[self.trade.pair]['trade_id'], 999)

    def test_partial_before_expiry_uses_native_accounting_and_preserves_position(self):
        self.partial()
        self.run_check()
        self.assertEqual(self.cancels, ['entry-1'])
        self.assertEqual(self.trade.amount, 1.0)
        self.assertTrue(self.trade.is_open)
        self.assertFalse(self.order.ft_is_open)
        self.assertIn(self.trade.pair, self.strategy.manual_open)
        self.assertEqual(self.strategy._execution(self.trade)['stage'], 0)

    def test_tiny_partial_is_not_left_growing_or_misclassified_as_empty(self):
        self.partial(0.001)
        self.exchange.remaining_floor = 700.0
        self.run_check()
        self.assertEqual(self.cancels, ['entry-1'])
        self.assertAlmostEqual(self.trade.amount, 0.001)
        self.assertTrue(self.trade.is_open)
        self.assertEqual(self.Trade.get_open_trade_count(), 1)

    def test_native_reconciliation_failure_preserves_receipt_and_retry_intent(self):
        self.partial()
        original = self.bot._update_trade_after_fill
        self.bot._update_trade_after_fill = lambda *args: (_ for _ in ()).throw(
            RuntimeError('failed after amount update')
        )
        self.run_check()
        self.Trade.session.expire_all()
        self.assertFalse(self.order.ft_is_open, 'native persists the real terminal receipt first')
        self.assertIn('entry-1', self.strategy._execution(self.trade)['entry_reconcile'])
        # Native update_trade commits the new quantity before its post-fill hooks.
        self.assertEqual(self.trade.amount, 1.0)
        ident = self.trade.id
        self.Trade.session.remove()
        self.trade = self.Trade.get_trades([self.Trade.id == ident]).first()
        self.strategy = type(self.strategy)(self.strategy.config)
        self.strategy.dp = SimpleNamespace(
            _exchange=self.exchange, send_msg=lambda text, **kw: self.messages.append(text)
        )
        self.bot.strategy = self.strategy
        self.bot._update_trade_after_fill = original
        self.raw = deepcopy(self.cancelled)
        self.run_check()
        self.assertEqual(self.trade.amount, 1.0)
        self.assertFalse(self.trade.select_order_by_order_id('entry-1').ft_is_open)
        self.assertNotIn('entry-1', self.strategy._execution(self.trade)['entry_reconcile'])

    def test_fill_during_cancel_is_reconciled_without_claiming_cancel_success(self):
        self.order.order_date = self.now - timedelta(hours=5)
        self.cancelled.update(status='closed', filled=10.0, remaining=0.0, cost=1000.0)
        self.run_check()
        self.assertEqual(self.trade.amount, 10.0)
        self.assertEqual(self.order.status, 'closed')
        self.assertFalse(any('已撤' in text for text in self.messages))

    def test_unknown_cancel_keeps_records_and_deduplicates_alerts(self):
        self.order.order_date = self.now - timedelta(hours=5)

        def fail(ident, pair, amount):
            self.cancels.append(ident)
            raise TimeoutError('SECRET_SIGNATURE')

        self.exchange.cancel_order_with_result = fail
        self.run_check()
        count = len(self.messages)
        self.run_check()
        self.assertEqual(self.Trade.get_open_trade_count(), 1)
        self.assertTrue(self.order.ft_is_open)
        self.assertEqual(len(self.messages), count)
        self.assertNotIn('SECRET_SIGNATURE', ' '.join(self.messages))
        self.assertEqual(self.queries, ['entry-1', 'entry-1'])

    def test_invalid_receipt_does_not_delete_or_reconcile_trade(self):
        self.order.order_date = self.now - timedelta(hours=5)
        for change in (
            {'id': 'wrong'},
            {'filled': None},
            {'filled': float('nan')},
            {'symbol': 'ETH/USDT:USDT'},
        ):
            with self.subTest(change=change):
                self.cancelled.update(change)
                self.run_check()
                self.assertTrue(self.order.ft_is_open)
                self.assertEqual(self.Trade.get_open_trade_count(), 1)
                self.cancelled = {**deepcopy(self.raw), 'status': 'canceled'}

    def test_stop_and_exit_orders_are_never_cancelled_by_entry_policy(self):
        self.order.ft_order_side = 'stoploss'
        self.order.order_date = self.now - timedelta(hours=5)
        self.run_check()
        self.assertEqual(self.queries, [])
        self.order.ft_order_side = 'sell'
        self.run_check()
        self.assertEqual(self.queries, [])

    def test_restart_and_plan_edit_do_not_extend_deadline(self):
        self.order.order_date = self.now - timedelta(hours=5)
        self.Trade.commit()
        ident = self.trade.id
        self.Trade.session.remove()
        self.trade = self.Trade.get_trades([self.Trade.id == ident]).first()
        self.strategy.set_manual_plan(
            self.trade.pair, {'trade_id': ident, 'direction': 'long', 'stop_loss': 95.0}
        )
        self.run_check()
        self.assertEqual(self.cancels, ['entry-1'])

    def test_cancelling_unfilled_add_keeps_existing_position_and_plan(self):
        self.add_order(
            'previous', self.now - timedelta(hours=8), 'manual_long', filled=10.0, status='closed'
        )
        self.trade.amount = 10.0
        self.Trade.commit()
        self.order.order_date = self.now - timedelta(hours=5)
        self.run_check()
        self.assertEqual(self.Trade.get_open_trade_count(), 1)
        self.assertEqual(self.trade.amount, 10.0)
        self.assertIn(self.trade.pair, self.strategy.manual_open)
