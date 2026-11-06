import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from test_telegram_assistant import load_telegram
from test_order_panel import update


class WatchPanelTests(unittest.TestCase):
    def setUp(self):
        self.module = load_telegram()
        self.assertTrue(hasattr(self.module.Telegram, '_watch'), 'watch entry missing')
        from assistant_alerts import AlertQueue
        from condition_watch import WatchStore

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = WatchStore(AlertQueue(Path(self.tmp.name) / 'alerts.sqlite'))
        self.bot = object.__new__(self.module.Telegram)
        self.bot._rpc = SimpleNamespace(
            _freqtrade=SimpleNamespace(strategy=SimpleNamespace(watches=self.store))
        )
        self.sent = []
        self.message_id = 100

        async def send(**kw):
            self.message_id += 1
            self.sent.append(dict(kw, message_id=self.message_id))
            return SimpleNamespace(message_id=self.message_id)

        self.bot._app = SimpleNamespace(bot=SimpleNamespace(send_message=send))

    def click(self, text, user=7):
        view = next(
            v for v in reversed(self.sent) if hasattr(v.get('reply_markup'), 'inline_keyboard')
        )
        buttons = [b for row in view['reply_markup'].inline_keyboard for b in row]
        button = next(b for b in buttons if text in b.text)
        u = update(user=user, message=view['message_id'], query=True)
        u.callback_query.data = button.callback_data

        async def edit(**kw):
            self.sent.append(dict(kw, message_id=view['message_id']))
            return SimpleNamespace(message_id=view['message_id'])

        u.callback_query.edit_message_text = edit
        asyncio.run(self.bot._watch_callback.__wrapped__(self.bot, u, None))
        return u

    def reply(self, text, user=7, prompt=None):
        prompt = prompt or self.sent[-1]['message_id']
        u = update(user=user, message=prompt + 1000, reply=prompt)
        u.message.text = text
        asyncio.run(self.bot._panel_reply.__wrapped__(self.bot, u, None))

    def start(self, mode):
        asyncio.run(self.bot._watch.__wrapped__(self.bot, update(), None))
        self.click('新建监控')
        self.click(mode)
        self.click('BTC')
        self.click('关注做空')
        if mode != '到价':
            self.click('1h')

    def finish(self, mode):
        if mode in ('突破后', '假突破'):
            self.reply('101')
        if mode in ('到价', '反弹受阻', '突破后'):
            self.reply('100')
            self.reply('102')
        if mode != '到价':
            self.reply('99')
        if mode in ('反弹受阻', '突破后', '假突破'):
            self.click('3根')
        self.click('不设价格失效')
        self.click('24小时')

    def test_all_five_wizards_confirm_once_and_copy_cancel(self):
        for mode in ('到价', '收盘站上', '反弹受阻', '突破后', '假突破'):
            self.start(mode)
            self.finish(mode)
            prior = len(self.store.list([123, 7, None]))
            text = self.sent[-1]['text']
            self.assertIn('仅提醒', text)
            self.assertIn('确认创建', text)
            u = self.click('确认创建')
            self.assertEqual(len(self.store.list([123, 7, None])), prior + 1)
            asyncio.run(self.bot._watch_callback.__wrapped__(self.bot, u, None))
            self.assertEqual(len(self.store.list([123, 7, None])), prior + 1)
        self.click('返回监控')
        self.click('BTC')
        self.click('取消监控')
        self.assertEqual(sum(w['state'] == 'cancelled' for w in self.store.list([123, 7, None])), 1)
        self.click('复制新建')
        self.assertIn('确认创建', self.sent[-1]['text'])
        self.click('确认创建')
        self.assertEqual(len(self.store.list([123, 7, None])), 6)

    def test_wrong_owner_unrelated_reply_and_invalid_input_preserve_draft(self):
        self.start('反弹受阻')
        prompt = self.sent[-1]['message_id']
        count = len(self.sent)
        self.reply('100', user=8, prompt=prompt)
        self.reply('100', prompt=prompt - 1)
        self.assertEqual(len(self.sent), count)
        self.reply('nan', prompt=prompt)
        self.assertIn('大于0', '\n'.join(v['text'] for v in self.sent[-2:]))
        self.reply('100')
        self.reply('98')
        self.assertIn('下限', '\n'.join(v['text'] for v in self.sent[-2:]))
        self.reply('102')
        self.reply('99')
        self.click('3根')
        self.click('不设价格失效')
        self.click('24小时')
        self.click('返回修改')
        self.assertIn('有效期', self.sent[-1]['text'])
        self.click('取消创建')
        self.assertEqual(self.store.list([123, 7, None]), [])
        self.reply('99', prompt=prompt)
        self.assertEqual(self.store.list([123, 7, None]), [])

    def test_registry_keyboard_and_expired_token_do_not_create(self):
        self.bot._init_keyboard()
        self.assertIn('/watch', sum(self.bot._keyboard, []))
        handlers = self.bot._assistant_handlers()
        self.assertTrue(any('watch' in getattr(h, 'commands', ()) for h in handlers))
        self.start('到价')
        self.finish('到价')
        for row in self.bot._panel_tokens.values():
            row['expires'] = 0
        self.click('确认创建')
        self.assertEqual(self.store.list([123, 7, None]), [])
        self.assertIn('过期', self.sent[-1]['text'])

    def test_review_can_edit_price_directly_without_losing_other_values(self):
        self.start('反弹受阻')
        self.finish('反弹受阻')
        self.click('修改收盘确认价')
        self.reply('98')
        self.assertIn('确认创建', self.sent[-1]['text'])
        self.assertIn('98', self.sent[-1]['text'])
        self.click('确认创建')
        w = self.store.list([123, 7, None])[0]
        self.assertEqual(w['spec']['confirm'], 98)
        self.assertEqual(w['spec']['high'], 102)

    def test_price_entry_returns_to_shortcuts_without_waiting_for_a_report(self):
        from telegram import ForceReply, ReplyKeyboardMarkup

        self.start('到价')
        self.assertIsInstance(self.sent[-1]['reply_markup'], ForceReply)
        before = sum(isinstance(v.get('reply_markup'), ReplyKeyboardMarkup) for v in self.sent)
        self.reply('100')
        self.assertIsInstance(self.sent[-1]['reply_markup'], ForceReply)
        self.assertEqual(
            sum(isinstance(v.get('reply_markup'), ReplyKeyboardMarkup) for v in self.sent), before
        )
        self.reply('102')
        self.click('不设价格失效')
        self.assertTrue(
            any(isinstance(v.get('reply_markup'), ReplyKeyboardMarkup) for v in self.sent[-2:])
        )
        self.assertIn('有效期', self.sent[-1]['text'])

    def test_cancelling_price_prompt_restores_keyboard_and_ignores_old_reply(self):
        from telegram import ReplyKeyboardMarkup

        self.start('到价')
        prompt = self.sent[-1]['message_id']
        before = len(self.sent)
        self.click('取消创建')
        self.assertTrue(
            any(isinstance(v.get('reply_markup'), ReplyKeyboardMarkup) for v in self.sent[before:])
        )
        count = len(self.sent)
        self.reply('100', prompt=prompt)
        self.assertEqual(len(self.sent), count)
        self.assertEqual(self.store.list([123, 7, None]), [])


if __name__ == '__main__':
    unittest.main()
