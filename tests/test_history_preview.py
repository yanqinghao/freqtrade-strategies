import asyncio
import json
import re
import unittest
from copy import deepcopy
from html import unescape
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import test_entry_risk as risk_tests
import test_telegram_orders as order_tests


class HistoryPreviewTests(unittest.TestCase):
    position = order_tests.TelegramOrderTests.position

    def setUp(self):
        order_tests.TelegramOrderTests.setUp(self)
        risk_tests.EntryRiskTests.prepare(self)
        self.ref = 'a' * 24
        self.store = self.strategy.entry_requests
        self.store.reserve(self.ref, self.request)
        self.store.set_status(self.ref, 'rejected', {'message': 'old result'})
        self.sent = []

        async def send(text, **kwargs):
            self.sent.append((text, kwargs))

        async def execute(function, *args):
            return function(*args)

        self.bot._send_msg = send
        self.bot._await_command = execute
        self.rpc._rpc_force_entry = lambda *a, **kw: self.fail('preview submitted an order')

    def preview(self, args):
        asyncio.run(self.bot._preview.__wrapped__(self.bot, None, SimpleNamespace(args=args)))
        return unescape('\n'.join(text for text, _ in self.sent))

    def test_history_preview_and_edited_command_are_read_only_and_use_current_limits(self):
        before = self.store.get(self.ref)
        message = self.preview([self.ref])
        self.assertIn('6.197', message)
        self.assertIn('<code>/preview long BTC/USDT:USDT 10 20 sl=97</code>', message)
        self.assertIn('<code>/forcelong BTC/USDT:USDT 10 20 sl=97</code>', message)
        self.assertEqual(self.store.get(self.ref), before)
        self.assertFalse(Path(self.strategy.state_file).exists())
        self.assertEqual(self.Trade.get_open_trade_count(), 0)
        self.path.write_text(json.dumps({**self.limits, 'max_margin_usdt': 5}))
        self.sent.clear()
        message = self.preview(['long', 'BTC/USDT:USDT', '10', '20', 'sl=97'])
        self.assertIn('当前会被拦截', message)
        self.assertIn('<code>/preview ', message)
        self.assertNotIn('<code>/forcelong ', message)

    def test_history_list_and_detail_offer_nontrading_buttons(self):
        asyncio.run(self.bot._request.__wrapped__(self.bot, None, SimpleNamespace(args=[])))
        keyboard = self.sent[-1][1].get('keyboard')
        self.assertTrue(keyboard, 'history needs a preview action')
        self.assertEqual(keyboard[0][0].callback_data, 'preview_request:' + self.ref)
        self.assertIn('预览', keyboard[0][0].text)
        self.sent.clear()
        asyncio.run(self.bot._request.__wrapped__(self.bot, None, SimpleNamespace(args=[self.ref])))
        self.assertEqual(
            self.sent[-1][1]['keyboard'][0][0].callback_data, 'preview_request:' + self.ref
        )

    def test_callback_repeats_do_not_resolve_unknown_request_or_change_plan(self):
        self.assertTrue(hasattr(self.bot, '_preview_history'))
        self.store.set_status(self.ref, 'unknown', {'message': 'still unknown'})
        before = deepcopy(self.store.get(self.ref))
        query = SimpleNamespace(data='preview_request:' + self.ref, answer=AsyncMock())
        update = SimpleNamespace(callback_query=query)
        for _ in range(2):
            asyncio.run(self.bot._preview_history.__wrapped__(self.bot, update, None))
        self.assertEqual(self.store.get(self.ref), before)
        self.assertEqual(self.Trade.get_open_trade_count(), 0)
        self.assertFalse(Path(self.strategy.state_file).exists())
        self.assertTrue(all('/forcelong ' not in text for text, _ in self.sent))
        self.assertIn('待核实', self.sent[-1][0])

    def test_unauthorized_chat_topic_and_user_cannot_preview(self):
        self.assertTrue(hasattr(self.bot, '_preview_history'))
        self.bot._config = {
            'telegram': {'chat_id': '12', 'topic_id': '7', 'authorized_users': ['5']}
        }
        for chat, topic, user in ((99, 7, 5), (12, 8, 5), (12, 7, 6)):
            query = SimpleNamespace(
                data='preview_request:' + self.ref,
                answer=AsyncMock(),
                message=SimpleNamespace(chat_id=chat, message_thread_id=topic),
            )
            update = SimpleNamespace(callback_query=query, effective_user=SimpleNamespace(id=user))
            asyncio.run(self.bot._preview_history(update, None))
        self.assertEqual(self.sent, [])

    def test_authorized_callback_uses_existing_position_without_mutating_plan_or_tp(self):
        self.assertTrue(hasattr(self.bot, '_preview_history'))
        trade = self.position()
        trade.amount = 2
        trade.leverage = 20
        self.strategy.set_manual_plan(
            trade.pair,
            {
                'direction': 'long',
                'trade_id': trade.id,
                'stop_loss': 97,
                'exit_points': [110, 120],
            },
        )
        self.strategy._save_execution(trade, {'stage': 0, 'seen': {'x': 1}})
        self.Trade.commit()
        before = Path(self.strategy.state_file).read_bytes()
        execution = deepcopy(self.strategy._execution(trade))
        self.bot._config = {
            'telegram': {'chat_id': '12', 'topic_id': '7', 'authorized_users': ['5']}
        }
        query = SimpleNamespace(
            data='preview_request:' + self.ref,
            answer=AsyncMock(),
            message=SimpleNamespace(chat_id=12, message_thread_id=7),
        )
        update = SimpleNamespace(callback_query=query, effective_user=SimpleNamespace(id=5))
        asyncio.run(self.bot._preview_history(update, None))
        self.assertIn('该币种加仓后', self.sent[-1][0])
        self.assertEqual(Path(self.strategy.state_file).read_bytes(), before)
        current = self.Trade.get_trades([self.Trade.id == trade.id]).first()
        self.assertEqual(self.strategy._execution(current), execution)
        self.assertEqual(self.Trade.get_open_trade_count(), 1)

    def test_chunked_history_attaches_preview_buttons_only_to_final_message(self):
        for i in range(5):
            self.store.reserve(f'{i:024x}', {**self.request, 'pair': 'LONG' * 80 + '/USDT:USDT'})
        asyncio.run(self.bot._request.__wrapped__(self.bot, None, SimpleNamespace(args=[])))
        self.assertGreater(len(self.sent), 1)
        self.assertTrue(all(not kwargs.get('keyboard') for _, kwargs in self.sent[:-1]))
        buttons = self.sent[-1][1]['keyboard']
        self.assertEqual(len(buttons), 5)
        for row in buttons:
            self.assertLessEqual(len(row[0].callback_data.encode()), 64)
            self.assertLess(len(row[0].text), 40)

    def test_missing_stop_error_keeps_editable_preview_without_inventing_stop(self):
        request = {k: v for k, v in self.request.items() if k != 'stop_loss'}
        ref = 'b' * 24
        self.store.reserve(ref, request)
        message = self.preview([ref])
        self.assertIn('开仓必须提供', message)
        self.assertIn('<code>/preview long BTC/USDT:USDT 10 20</code>', message)
        self.assertNotIn('<code>/forcelong ', message)

    def test_short_limit_command_roundtrip_preserves_price_tp_and_ttl(self):
        request = self.module.parse_entry(
            [
                'BTC',
                '5.123456789',
                '3',
                'price=100.123456789',
                'sl=103',
                'tp=98.123456789,95',
                'ttl=90m',
            ],
            'short',
        )
        ref = 'c' * 24
        self.store.reserve(ref, request)
        message = self.preview([ref])
        draft = re.search(r'<code>(/preview [^<]+)</code>', message)
        self.assertIsNotNone(draft)
        parts = draft.group(1).split()
        self.assertEqual(self.module.parse_entry(parts[2:], parts[1]), request)
        self.assertIn('<code>/forceshort ', message)

    def test_missing_ref_is_visible_and_does_not_create_request(self):
        before = self.store.recent()
        message = self.preview(['f' * 24])
        self.assertIn('未找到', message)
        self.assertEqual(self.store.recent(), before)

    def test_candle_rules_survive_history_preview_and_remain_read_only(self):
        request = self.module.parse_entry(
            [
                'BTC',
                '5',
                '3',
                'sl=103',
                'price=100',
                'tp=98,96,94',
                'soft_sl=102',
                'soft_tf=4h',
                'extend_trigger=95',
                'extend_tp3=92',
                'extend_tf=4h',
            ],
            'short',
        )
        ref = 'd' * 24
        self.store.reserve(ref, request)
        before = deepcopy(self.store.get(ref))
        message = self.preview([ref])
        draft = re.search(r'<code>(/preview [^<]+)</code>', message)
        parts = draft.group(1).split()
        self.assertEqual(self.module.parse_entry(parts[2:], parts[1]), request)
        self.assertIn('软SL', message)
        self.assertIn('待延伸', message)
        self.assertIn('风险估算仍按硬SL', message)
        self.assertEqual(self.store.get(ref), before)
        self.assertFalse(Path(self.strategy.state_file).exists())

    def test_button_handler_is_nonblocking_and_matches_only_preview_ids(self):
        handlers = self.bot._assistant_handlers()
        matching = [
            h
            for h in handlers
            if getattr(h, 'pattern', None) and h.pattern.match('preview_request:' + self.ref)
        ]
        self.assertEqual(len(matching), 1)
        self.assertFalse(matching[0].block)
        self.assertFalse(matching[0].pattern.match('preview_request:' + self.ref + ':execute'))
