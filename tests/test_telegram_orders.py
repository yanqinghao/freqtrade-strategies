import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from threading import Lock

from test_telegram_assistant import load_telegram
from test_exit_policy import RuleExchange

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'user_data' / 'strategies'))


class TelegramOrderTests(unittest.TestCase):
    def setUp(self):
        self.module = load_telegram()
        from freqtrade.persistence import init_db, Trade
        from freqtrade.enums import RunMode
        from KamaFama_Dynamic import KamaFama_Dynamic

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        init_db('sqlite:///' + str(Path(self.temp.name) / 'test.sqlite'))
        engine = Trade.session.get_bind()

        def cleanup_database():
            from freqtrade.persistence.custom_data import _CustomData

            _CustomData.session.remove()
            Trade.session.remove()
            engine.dispose()

        self.addCleanup(cleanup_database)
        self.Trade = Trade
        self.strategy = KamaFama_Dynamic(
            {
                'runmode': RunMode.DRY_RUN,
                'user_data_dir': Path(self.temp.name),
                'stake_currency': 'USDT',
            }
        )
        self.bot = object.__new__(self.module.Telegram)
        self.rpc = SimpleNamespace(
            _freqtrade=SimpleNamespace(
                strategy=self.strategy,
                exchange=SimpleNamespace(get_rate=lambda *a, **k: 100.0),
                _exit_lock=Lock(),
            )
        )
        self.bot._rpc = self.rpc
        self.exchange = RuleExchange(minimum=0.001, step='.001', cost=0.0, pair='BTC/USDT:USDT')
        self.exchange._api = SimpleNamespace(fetch_open_orders=lambda *a, **kw: [])
        self.rpc._freqtrade.exchange = self.exchange
        self.strategy.dp = SimpleNamespace(_exchange=self.exchange, send_msg=lambda *a, **kw: None)
        self.request = self.module.parse_entry(
            ['BTC', '100', '3', 'sl=90', 'tp=110,120,130'], 'long'
        )

    def position(self, is_short=False):
        trade = self.Trade(
            pair='BTC/USDT:USDT',
            is_short=is_short,
            is_open=True,
            open_rate=100.0,
            amount=10.0,
            stake_amount=100.0,
            leverage=3.0,
            open_date=datetime.now(timezone.utc),
            exchange='binance',
            enter_tag='manual_long',
            fee_open=0.0005,
            fee_close=0.0005,
        )
        self.Trade.session.add(trade)
        self.Trade.commit()
        return trade

    def test_rejected_submission_restores_previous_plan(self):
        from freqtrade.rpc import RPCException

        prior = {'direction': 'long', 'stop_loss': 80, 'exit_points': [150]}
        self.strategy.set_manual_plan(self.request['pair'], prior)

        def reject(*args, **kwargs):
            raise RPCException('rejected')

        self.rpc._rpc_force_entry = reject
        with self.assertRaises(RPCException):
            self.bot._submit_entry(self.request)
        self.assertEqual(self.strategy.manual_open[self.request['pair']], prior)

    def test_live_external_position_rejects_before_plan_or_order_mutation(self):
        self.strategy.config['dry_run'] = False
        prior = {'direction': 'long', 'stop_loss': 80, 'exit_points': [150]}
        self.strategy.set_manual_plan(self.request['pair'], prior)
        self.exchange.fetch_positions = lambda pair: [
            {
                'symbol': pair,
                'contracts': 1.0,
                'side': 'long',
                'contractSize': 1.0,
                'hedged': False,
                'info': {'positionSide': 'BOTH'},
            }
        ]
        calls = []
        self.rpc._rpc_force_entry = lambda *a, **k: calls.append(a) or self.position()
        with self.assertRaisesRegex(ValueError, '未接管'):
            self.bot._submit_entry(self.request)
        self.assertEqual(calls, [])
        self.assertEqual(self.strategy.manual_open[self.request['pair']], prior)

    def test_external_entry_order_blocks_before_plan_and_submission(self):
        self.strategy.config['dry_run'] = False
        self.exchange.fetch_positions = lambda pair: []
        self.exchange._api.fetch_open_orders = lambda pair, **kw: [
            {
                'id': 'outside',
                'symbol': pair,
                'status': 'open',
                'side': 'buy',
                'type': 'limit',
                'reduceOnly': False,
                'info': {'reduceOnly': False},
            }
        ]
        calls = []
        self.rpc._rpc_force_entry = lambda *a, **kw: calls.append(a) or self.position()
        with self.assertRaisesRegex(ValueError, '挂单'):
            self.bot._submit_entry(self.request)
        self.assertEqual(calls, [])
        self.assertEqual(self.strategy.manual_open, {})

    def test_live_position_query_failure_does_not_submit_order(self):
        self.strategy.config['dry_run'] = False

        def fail(pair):
            raise TimeoutError('private network failure')

        self.exchange.fetch_positions = fail
        calls = []
        self.rpc._rpc_force_entry = lambda *a, **k: calls.append(a) or self.position()
        with self.assertRaisesRegex(ValueError, '无法核对'):
            self.bot._submit_entry(self.request)
        self.assertEqual(calls, [])
        self.assertEqual(self.strategy.manual_open, {})

    def test_live_exit_is_not_blocked_by_unavailable_position_query(self):
        self.strategy.config['dry_run'] = False
        trade = self.position()

        def fail(pair):
            raise TimeoutError('position endpoint unavailable')

        self.exchange.fetch_positions = fail
        calls = []
        self.rpc._rpc_force_exit = lambda ident, **kw: calls.append((ident, kw)) or {'result': 'ok'}
        self.bot._close_position(str(trade.id), 100)
        self.assertEqual(len(calls), 1)

    def test_live_add_without_confirmed_stop_is_rejected_before_plan_change(self):
        self.strategy.config['dry_run'] = False
        trade = self.position()
        prior = {
            'trade_id': trade.id,
            'direction': 'long',
            'stop_loss': 90.0,
            'exit_points': [110.0, 120.0, 130.0],
        }
        self.strategy.set_manual_plan(trade.pair, prior)
        self.exchange.fetch_positions = lambda pair: [
            {
                'symbol': pair,
                'contracts': 10.0,
                'side': 'long',
                'contractSize': 1.0,
                'hedged': False,
                'info': {'positionSide': 'BOTH'},
            }
        ]
        calls = []
        self.rpc._rpc_force_entry = lambda *a, **kw: calls.append(a) or trade
        with self.assertRaisesRegex(ValueError, '止损'):
            self.bot._submit_entry(self.request)
        self.assertEqual(calls, [])
        self.assertEqual(self.strategy.manual_open[trade.pair], prior)

    def test_uncertain_submission_preserves_protection_plan(self):
        def unknown(*args, **kwargs):
            raise TimeoutError('exchange acknowledgement unknown')

        self.rpc._rpc_force_entry = unknown
        with self.assertRaises(TimeoutError):
            self.bot._submit_entry(self.request)
        self.assertEqual(self.strategy.manual_open[self.request['pair']]['stop_loss'], 90)

    def test_reverse_request_never_changes_plan_or_creates_order(self):
        self.position(is_short=True)
        self.strategy.set_manual_plan(
            self.request['pair'], {'direction': 'short', 'stop_loss': 120}
        )
        with self.assertRaisesRegex(ValueError, '单方向'):
            self.bot._submit_entry(self.request)
        self.assertEqual(self.strategy.manual_open[self.request['pair']]['direction'], 'short')

    def test_add_requires_existing_leverage(self):
        trade = self.position()
        trade.leverage = 10
        self.Trade.commit()
        with self.assertRaisesRegex(ValueError, '原有|现有'):
            self.bot._submit_entry(self.request)

    def test_close_percentage_uses_actual_base_quantity(self):
        trade = self.position()
        calls = []
        self.rpc._rpc_force_exit = lambda ident, **kw: calls.append((ident, kw)) or {'result': 'ok'}
        self.bot._close_position(str(trade.id), 40)
        self.assertEqual(calls[0][1]['amount'], 4.0)

    def test_close_percentage_with_tiny_remainder_requests_native_full_close(self):
        trade = self.position()
        self.exchange.remaining_floor = 700.0
        calls = []
        self.rpc._rpc_force_exit = lambda ident, **kw: calls.append((ident, kw)) or {'result': 'ok'}
        self.bot._close_position(str(trade.id), 40)
        self.assertIsNone(calls[0][1].get('amount'))
        self.assertTrue(trade.get_custom_data('assistant_execution')['full_exit'])

    def test_native_rpc_accepts_fallback_without_remaining_amount_rejection(self):
        from freqtrade.rpc import RPC
        from freqtrade.enums import State

        trade = self.position()
        self.exchange.remaining_floor = 700.0
        runtime = self.rpc._freqtrade
        runtime.state = State.RUNNING
        runtime.wallets = SimpleNamespace(update=lambda: None)
        submitted = []
        runtime.execute_trade_exit = lambda *a, **kw: submitted.append(kw) or True
        native_rpc = object.__new__(RPC)
        native_rpc._freqtrade = runtime
        self.bot._rpc = native_rpc
        self.bot._close_position(str(trade.id), 40)
        self.assertEqual(len(submitted), 1)
        self.assertIsNone(submitted[0]['sub_trade_amt'])

    def test_full_fallback_survives_reload_and_rejects_add(self):
        trade = self.position()
        self.strategy.set_manual_plan(
            trade.pair,
            {
                'trade_id': trade.id,
                'direction': 'long',
                'stop_loss': 90,
                'exit_points': [110, 120, 130],
            },
        )
        self.strategy._plan(trade)
        self.exchange.remaining_floor = 700.0
        self.rpc._rpc_force_exit = lambda *a, **kw: {'result': 'ok'}
        self.bot._close_position(str(trade.id), 40)
        self.Trade.session.remove()
        reopened = self.Trade.get_open_trades()[0]
        restored = type(self.strategy)(self.strategy.config)
        restored.dp = self.strategy.dp
        self.assertTrue(restored._execution(reopened)['full_exit'])
        self.assertIsNone(
            restored.custom_exit(reopened.pair, reopened, datetime.now(timezone.utc), 105.0, 0.5)
        )
        with self.assertRaisesRegex(ValueError, '全平'):
            self.bot._submit_entry(self.request)

    def test_plan_edit_does_not_change_position_direction(self):
        trade = self.position()
        with self.assertRaisesRegex(ValueError, '方向'):
            self.bot._update_plan(str(trade.id), ['short', 'sl=120', 'tp=90'])
        self.assertFalse(self.Trade.get_trades([self.Trade.id == trade.id]).first().is_short)

    def test_new_entry_discards_stale_trade_binding(self):
        self.strategy.set_manual_plan(
            self.request['pair'], {'direction': 'long', 'stop_loss': 90, 'trade_id': 999}
        )

        def create(pair, price, **kwargs):
            self.assertNotIn('trade_id', self.strategy.manual_open[pair])
            return self.position()

        self.rpc._rpc_force_entry = create
        self.bot._submit_entry(self.request)

    def test_limit_ttl_is_sent_as_order_tag_not_position_plan(self):
        request = self.module.parse_entry(
            ['BTC', '100', '3', 'sl=90', 'price=100', 'ttl=1h'], 'long'
        )
        calls = []
        self.rpc._rpc_force_entry = lambda *args, **kwargs: calls.append(kwargs) or self.position()
        self.bot._submit_entry(request)
        self.assertEqual(calls[0]['enter_tag'], 'manual_long|ttl=3600')
        self.assertNotIn('ttl_seconds', self.strategy.manual_open[request['pair']])

    def test_take_profit_edit_rejected_while_order_is_open(self):
        from freqtrade.persistence import Order

        trade = self.position()
        self.strategy.set_manual_plan(
            trade.pair, {'direction': 'long', 'stop_loss': 90, 'exit_points': [110, 120, 130]}
        )
        order = Order(
            order_id='pending',
            ft_order_side=trade.exit_side,
            ft_is_open=True,
            ft_pair=trade.pair,
            order_type='limit',
            status='open',
            symbol=trade.pair,
            side=trade.exit_side,
            amount=3,
            filled=0,
            remaining=3,
            price=110,
            ft_amount=3,
            ft_price=110,
        )
        trade.orders.append(order)
        self.Trade.commit()
        with self.assertRaisesRegex(ValueError, '挂单|订单'):
            self.bot._update_plan(str(trade.id), ['tp=120,130,140'])
        self.assertEqual(self.strategy.manual_open[trade.pair]['exit_points'], [110, 120, 130])
        self.bot._update_plan(str(trade.id), ['sl=95'])
        self.assertEqual(self.strategy.manual_open[trade.pair]['stop_loss'], 95)

    def test_plan_edit_holds_native_exit_lock(self):
        trade = self.position()
        original = self.strategy.set_manual_plan

        def verify_locked(*args, **kwargs):
            self.assertTrue(self.rpc._freqtrade._exit_lock.locked())
            return original(*args, **kwargs)

        self.strategy.set_manual_plan = verify_locked
        self.bot._update_plan(str(trade.id), ['sl=90'])


if __name__ == '__main__':
    unittest.main()
