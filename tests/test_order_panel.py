import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from test_telegram_assistant import load_telegram


def update(user=7, message=10, query=False, reply=None):
    msg = SimpleNamespace(
        chat_id=123,
        message_id=message,
        message_thread_id=None,
        reply_to_message=SimpleNamespace(message_id=reply) if reply else None,
        text='123',
    )
    return SimpleNamespace(
        effective_user=SimpleNamespace(id=user),
        effective_chat=SimpleNamespace(id=123),
        effective_message=msg,
        message=None if query else msg,
        callback_query=SimpleNamespace(
            message=msg, answer=AsyncMock(), edit_message_text=AsyncMock()
        )
        if query
        else None,
    )


class OrderPanelTests(unittest.TestCase):
    def setUp(self):
        self.module = load_telegram()
        self.bot = object.__new__(self.module.Telegram)
        self.bot._manual_order_lock = asyncio.Lock()
        self.bot._app = SimpleNamespace(
            bot=SimpleNamespace(send_message=AsyncMock(return_value=SimpleNamespace(message_id=99)))
        )
        self.bot._rpc = SimpleNamespace(_freqtrade=SimpleNamespace(strategy=SimpleNamespace()))
        self.bot._await_command = AsyncMock()

    def test_callbacks_are_user_and_topic_bound_and_expire(self):
        token = self.bot._panel_token(update(), {'action': 'list'})
        self.assertEqual(self.bot._panel_take(update(), token)['action'], 'list')
        with self.assertRaises(ValueError):
            self.bot._panel_take(update(user=8), token)
        other = update()
        other.effective_message.message_thread_id = 4
        with self.assertRaises(ValueError):
            self.bot._panel_take(other, token)
        self.bot._panel_tokens[token]['expires'] = 0
        with self.assertRaises(ValueError):
            self.bot._panel_take(update(), token)

    def test_confirmation_token_is_consumed_before_command_await(self):
        token = self.bot._panel_token(
            update(), dict(action='do', id=1, revision='r', operation='close')
        )
        u = update(query=True)
        u.callback_query.data = 'om:' + token
        self.bot._await_command.return_value = '已提交全平请求'
        self.bot._panel_detail = AsyncMock()
        for _ in range(2):
            asyncio.run(self.bot._panel_callback.__wrapped__(self.bot, u, None))
        self.assertEqual(self.bot._await_command.call_count, 1)

    def test_unrelated_and_wrong_user_replies_do_not_edit(self):
        self.bot._panel_prompt = AsyncMock()
        self.bot._panel_edits = {
            (123, 7, None): dict(prompt=99, expires=10**20, id=1, field='tp1')
        }
        self.bot._panel_send = AsyncMock()
        for u in (update(reply=98), update(user=8, reply=99)):
            asyncio.run(self.bot._panel_reply.__wrapped__(self.bot, u, None))
        self.bot._await_command.assert_not_called()

    def test_detail_has_refresh_fields_and_confirm_close_without_trading(self):
        data = dict(
            id=1,
            pair='BTC/USDT:USDT',
            short=False,
            kind='position',
            amount=1,
            entry=100,
            leverage=3,
            plan=dict(stop_loss=90, exit_points=[110, 120, 130]),
            orders=[],
            stage=0,
            extension=None,
            full_exit=None,
            tp_filled={},
            final_tp_started=False,
            stop=90,
            revision='r',
            rate=105,
            fresh=True,
            error='',
            checked='now',
            profit=15,
            profit_abs=5,
            pnl_status='可用',
            deadline=None,
        )
        self.bot._await_command.return_value = data
        asyncio.run(self.bot._panel_detail(update(), 1))
        kwargs = self.bot._app.bot.send_message.call_args.kwargs
        labels = [b.text for row in kwargs['reply_markup'].inline_keyboard for b in row]
        self.assertIn('快速全平', labels)
        self.assertIn('TP1', labels)
        self.assertIn('刷新', labels)
        self.assertNotIn('开仓价', labels)
        self.assertIn('105', kwargs['text'])
        data['fresh'] = False
        data['error'] = '订单回执暂未核实'
        asyncio.run(self.bot._panel_detail(update(), 1))
        labels = [
            b.text
            for row in self.bot._app.bot.send_message.call_args.kwargs[
                'reply_markup'
            ].inline_keyboard
            for b in row
        ]
        self.assertNotIn('快速全平', labels)
        self.assertNotIn('TP1', labels)

    def test_status_table_and_orders_share_panel(self):
        self.bot._panel_list = AsyncMock()
        asyncio.run(
            self.bot._status_table.__wrapped__(self.bot, update(), SimpleNamespace(args=['table']))
        )
        self.bot._panel_list.assert_awaited_once()

    def test_render_binds_callbacks_to_message_and_cancels_old_mutations(self):
        u = update(query=True)
        old = self.bot._panel_button(u, '确认全平', action='do', operation='close')
        asyncio.run(self.bot._panel_send(u, '确认', [[old]]))
        token = old.callback_data[3:]
        with self.assertRaises(ValueError):
            self.bot._panel_take(update(message=11), token)
        back = self.bot._panel_button(u, '返回列表', action='list')
        asyncio.run(self.bot._panel_send(u, '已取消', [[back]]))
        with self.assertRaises(ValueError):
            self.bot._panel_take(u, token)

    def test_opening_panel_publishes_persistent_shortcuts_before_inline_controls(self):
        from telegram import InlineKeyboardMarkup, ReplyKeyboardMarkup

        u = update()
        button = self.bot._panel_button(u, '返回列表', action='list')
        asyncio.run(self.bot._panel_send(u, '订单管理', [[button]]))
        calls = self.bot._app.bot.send_message.call_args_list
        keyboard = calls[0].kwargs['reply_markup']
        self.assertIsInstance(keyboard, ReplyKeyboardMarkup)
        self.assertTrue(keyboard.is_persistent)
        self.assertFalse(keyboard.one_time_keyboard)
        self.assertIn('/watch', [b.text for row in keyboard.keyboard for b in row])
        self.assertIsInstance(calls[-1].kwargs['reply_markup'], InlineKeyboardMarkup)
        self.assertEqual(
            self.bot._panel_take(update(message=99), button.callback_data[3:]), {'action': 'list'}
        )

    def test_leaving_force_reply_restores_shortcuts_once_in_the_same_topic(self):
        from telegram import ForceReply, ReplyKeyboardMarkup

        u = update(query=True)
        u.effective_message.message_thread_id = 42
        asyncio.run(self.bot._panel_send(u, '输入价格', prompt=True))
        self.assertIsInstance(
            self.bot._app.bot.send_message.call_args.kwargs['reply_markup'], ForceReply
        )
        before = self.bot._app.bot.send_message.call_count
        asyncio.run(
            self.bot._panel_send(u, '已取消', [[self.bot._panel_button(u, '返回', action='list')]])
        )
        self.assertEqual(self.bot._app.bot.send_message.call_count, before + 1)
        sent = self.bot._app.bot.send_message.call_args.kwargs
        self.assertIsInstance(sent['reply_markup'], ReplyKeyboardMarkup)
        self.assertEqual(sent['message_thread_id'], 42)
        self.assertTrue(sent['disable_notification'])
        asyncio.run(
            self.bot._panel_send(u, '再次刷新', [[self.bot._panel_button(u, '返回', action='list')]])
        )
        self.assertEqual(self.bot._app.bot.send_message.call_count, before + 1)

    def test_reply_saves_single_existing_rule_value_and_duplicate_reply_is_ignored(self):
        self.bot._panel_edits = {
            (123, 7, None): dict(
                prompt=99, expires=10**20, id=1, revision='r', field='soft_sl', existing=True
            )
        }
        self.bot._panel_detail = AsyncMock()
        self.bot._await_command.return_value = '已保存'
        u = update(reply=99)
        for _ in range(2):
            asyncio.run(self.bot._panel_reply.__wrapped__(self.bot, u, None))
        self.bot._await_command.assert_awaited_once()
        self.assertEqual(self.bot._await_command.call_args.args[-1], {'soft_sl': 123})

    def test_invalid_price_retry_keeps_input_without_restoring_menu_between_prompts(self):
        from telegram import ForceReply, ReplyKeyboardMarkup

        draft = dict(id=1, revision='r', field='stop_loss', current=90, existing=True)
        asyncio.run(self.bot._panel_prompt_value(update(query=True), draft))
        before = self.bot._app.bot.send_message.call_count
        u = update(reply=99)
        u.message.text = 'nan'
        asyncio.run(self.bot._panel_reply.__wrapped__(self.bot, u, None))
        sent = self.bot._app.bot.send_message.call_args_list[before:]
        self.assertFalse(
            any(isinstance(c.kwargs['reply_markup'], ReplyKeyboardMarkup) for c in sent)
        )
        self.assertIsInstance(sent[-1].kwargs['reply_markup'], ForceReply)
        self.bot._await_command.assert_not_called()

    def test_extension_wizard_saves_only_after_complete_timeframe_selection(self):
        self.bot._panel_edits = {
            (123, 7, None): dict(
                prompt=99,
                expires=10**20,
                id=1,
                revision='r',
                field='extend_trigger',
                existing=False,
            )
        }
        u = update(reply=99)
        u.message.text = '125'
        asyncio.run(self.bot._panel_reply.__wrapped__(self.bot, u, None))
        self.bot._await_command.assert_not_called()
        u.message.text = '150'
        asyncio.run(self.bot._panel_reply.__wrapped__(self.bot, u, None))
        self.bot._await_command.assert_not_called()
        kwargs = self.bot._app.bot.send_message.call_args.kwargs
        button = kwargs['reply_markup'].inline_keyboard[1][0]
        callback = update(message=99, query=True)
        callback.callback_query.data = button.callback_data
        self.bot._panel_detail = AsyncMock()
        asyncio.run(self.bot._panel_callback.__wrapped__(self.bot, callback, None))
        self.assertEqual(
            self.bot._await_command.call_args.args[-1],
            {'extend_trigger': 125, 'extend_tp3': 150, 'extend_tf': '4h'},
        )
