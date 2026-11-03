import asyncio
import importlib.util
import unittest
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import test_entry_risk as risk_tests
import test_telegram_orders as order_tests


class TradeOverviewTests(unittest.TestCase):
    setUp = order_tests.TelegramOrderTests.setUp
    position = order_tests.TelegramOrderTests.position
    order = risk_tests.EntryRiskTests.order

    def prepare(self):
        self.assertIsNotNone(
            importlib.util.find_spec('trade_overview'), 'read-only overview missing'
        )
        import trade_overview

        self.view = trade_overview.overview_messages
        self.now = datetime(2026, 11, 3, 12, tzinfo=timezone.utc)
        self.messages = []

        def no_network(*a, **kw):
            raise AssertionError('overview queried exchange')

        self.exchange.get_rate = no_network
        self.exchange.fetch_stoploss_order = no_network
        self.exchange.get_fee = no_network
        self.strategy.dp.send_msg = lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError('side effect')
        )

    def render(self, trades):
        return self.view(self.strategy, trades, 'RUNNING', self.now)

    def test_empty_overview_is_explicit_and_does_not_create_journal_or_plan(self):
        self.prepare()
        text = '\n'.join(self.render([]))
        self.assertIn('RUNNING', text)
        self.assertIn('暂无', text)
        self.assertIn('未启用', text)
        self.assertFalse(self.strategy.entry_requests.path.exists())
        self.assertFalse(Path(self.strategy.state_file).exists())

    def test_pending_order_reports_expiry_legacy_and_expired_without_claiming_cancel(self):
        self.prepare()
        trade = self.position()
        trade.amount = 0
        self.order(trade, amount=2, filled=0)
        row = trade.orders[0]
        row.order_date = self.now - timedelta(hours=23)
        text = '\n'.join(self.render([trade]))
        self.assertIn('1h', text)
        self.assertIn(row.order_id, text)
        row.ft_order_tag = 'manual_long'
        self.assertIn('原生超时', '\n'.join(self.render([trade])))
        row.ft_order_tag = 'manual_long|ttl=86400'
        row.order_date = self.now - timedelta(hours=25)
        text = '\n'.join(self.render([trade]))
        self.assertIn('已到期', text)
        self.assertNotIn('已撤单', text)
        self.assertTrue(row.ft_is_open)

    def test_fresh_stop_snapshot_stale_and_insufficient_coverage_are_distinguished(self):
        self.prepare()
        trade = self.position()
        trade.amount = 2
        self.strategy.set_manual_plan(
            trade.pair,
            {
                'direction': 'long',
                'trade_id': trade.id,
                'stop_loss': 90,
                'exit_points': [110, 120, 130],
            },
        )
        state = {
            'stage': 0,
            'protected_stop': 90,
            'protection': {
                'status': 'confirmed',
                'stop_price': 90,
                'covered_amount': 2,
                'checked_at': self.now.timestamp() - 10,
            },
        }
        self.strategy._save_execution(trade, state)
        text = '\n'.join(self.render([trade]))
        self.assertIn('核验快照', text)
        self.assertIn('90', text)
        state['protection']['checked_at'] = self.now.timestamp() - 120
        self.strategy._save_execution(trade, state)
        self.assertIn('过期', '\n'.join(self.render([trade])))
        state['protection'].update(checked_at=self.now.timestamp(), covered_amount=1)
        self.strategy._save_execution(trade, state)
        self.assertIn('覆盖不足', '\n'.join(self.render([trade])))
        state['protection'].update(covered_amount=2, stop_price=89)
        self.strategy._save_execution(trade, state)
        self.assertIn('落后', '\n'.join(self.render([trade])))

    def test_partial_fill_not_yet_native_accounted_is_visible(self):
        self.prepare()
        trade = self.position()
        trade.amount = 0
        self.order(trade, amount=2, filled=0.5)
        text = '\n'.join(self.render([trade]))
        self.assertIn('尚未完成记账', text)
        self.assertIn('0.5', text)
        self.assertIn('1.5', text)

    def test_stage_plan_and_protection_reads_never_mutate_native_state(self):
        self.prepare()
        trade = self.position()
        trade.amount = 2
        old = {
            'direction': 'long',
            'trade_id': trade.id,
            'stop_loss': 90,
            'exit_points': [110, 120],
        }
        trade.set_custom_data('assistant_plan', old)
        new = {**old, 'stop_loss': 95, 'exit_points': [115, 125]}
        self.strategy.set_manual_plan(trade.pair, new)
        state = {'stage': 1, 'protected_stop': 100.1, 'full_exit': {'status': 'blocked'}}
        self.strategy._save_execution(trade, state)
        before = Path(self.strategy.state_file).read_bytes()
        text = '\n'.join(self.render([trade]))
        self.assertIn('阶段 1', text)
        self.assertIn('待应用', text)
        self.assertIn('全平', text)
        self.assertEqual(self.strategy._execution(trade), state)
        self.assertEqual(trade.get_custom_data('assistant_plan'), old)
        self.assertEqual(Path(self.strategy.state_file).read_bytes(), before)

    def test_bad_one_trade_does_not_hide_other_trade_and_invalid_config_is_visible(self):
        self.prepare()
        bad = SimpleNamespace(
            id=999,
            pair='BAD/USDT:USDT',
            is_open=True,
            is_short=False,
            amount=1.0,
            leverage=20.0,
            open_rate=float('nan'),
            orders=[],
            get_custom_data=lambda *a, **kw: {},
        )
        good = self.position()
        good.amount = 1
        self.strategy.risk_limits_file.write_text('{bad')
        rows = self.render([bad, good])
        text = '\n'.join(rows)
        self.assertIn('不可用', text)
        self.assertIn('BTC/USDT:USDT', text)
        self.assertIn('风险配置', text)
        self.assertEqual(len(rows), 3)

    def test_unknown_request_is_listed_without_reclassifying_or_resending(self):
        self.prepare()
        store = self.strategy.entry_requests
        store.reserve(
            'a' * 24, {'pair': 'BTC/USDT:USDT', 'direction': 'long', 'size': 10, 'leverage': 20}
        )
        store.set_status('a' * 24, 'submitting', {})
        before = deepcopy(store.get('a' * 24))
        text = '\n'.join(self.render([]))
        self.assertIn('a' * 24, text)
        self.assertIn('待核实', text)
        self.assertEqual(store.get('a' * 24), before)

    def test_no_fresh_confirmation_claim_for_simulated_or_unfilled_trade(self):
        self.prepare()
        trade = self.position()
        self.strategy._save_execution(
            trade, {'protection': {'status': 'simulated', 'checked_at': self.now.timestamp()}}
        )
        self.assertIn('模拟', '\n'.join(self.render([trade])))
        trade.amount = 0
        self.assertIn('未成交', '\n'.join(self.render([trade])))

    def test_telegram_overview_sends_separate_plain_text_snapshots_without_orders(self):
        self.prepare()
        self.position()
        sent = []

        async def send(text, **kwargs):
            sent.append((text, kwargs))

        async def execute(function, *args):
            return function(*args)

        self.bot._send_msg, self.bot._await_command = send, execute
        asyncio.run(self.bot._overview.__wrapped__(self.bot, None, SimpleNamespace(args=[])))
        self.assertEqual(len(sent), 2)
        self.assertTrue(all(kw.get('parse_mode') is None for _, kw in sent))
        self.assertFalse(self.strategy.entry_requests.path.exists())
