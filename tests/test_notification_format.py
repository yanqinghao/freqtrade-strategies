import sys
import unittest
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'user_data' / 'strategies'))


class NotificationFormatTests(unittest.TestCase):
    def test_card_keeps_order_precision_and_one_beijing_timestamp(self):
        from notification_format import notice, with_time

        now = datetime(2026, 11, 12, 4, 5, tzinfo=timezone.utc)
        text = notice(
            '持仓简报',
            'SKYAI/USDT:USDT #18 · 多',
            summary=['建议：持有观察'],
            sections=[('盈亏', ['+1.2345 USDT', '均价 0.0361']), ('', [])],
            action='用 /overview 核对',
            at=now,
        )
        self.assertLess(text.index('建议'), text.index('盈亏'))
        self.assertIn('均价 0.0361', text)
        self.assertIn('12:05:00 北京时间', text)
        self.assertEqual(text.count('时间：'), 1)
        self.assertEqual(with_time(text, now), text)
        self.assertNotIn('\n\n\n', text)

    def test_plain_text_preserves_symbols_and_unknown_status(self):
        from notification_format import notice

        text = notice(
            '入场', 'A_B/USDT #1', summary=['结果未确认 <timeout>'], sections=[('订单', ['编号 123', None])]
        )
        self.assertIn('A_B/USDT', text)
        self.assertIn('<timeout>', text)
        self.assertNotIn('None', text)
        self.assertNotIn('成功', text)


class NativeNotificationTests(unittest.TestCase):
    def setUp(self):
        from test_telegram_assistant import load_telegram
        from freqtrade.enums import RPCMessageType

        self.types = RPCMessageType
        self.bot = object.__new__(load_telegram().Telegram)

    def message(self, kind, **extra):
        now = datetime(2026, 11, 12, 4, tzinfo=timezone.utc)
        return dict(
            type=kind,
            pair='A_B/USDT:USDT',
            trade_id=18,
            direction='Long',
            sub_trade=False,
            amount=2770,
            leverage=3,
            open_rate=0.0361,
            current_rate=0.037,
            order_rate=0.037,
            close_rate=0.037,
            quote_currency='USDT',
            stake_currency='USDT',
            stake_amount=33.3,
            profit_amount=2.4,
            profit_ratio=0.024,
            cumulative_profit=7.4,
            final_profit_ratio=0.074,
            exit_reason='manual_exit',
            open_date=now,
            close_date=now,
            **extra
        )

    def native_context(self):
        from types import SimpleNamespace

        self.bot._config = {'dry_run': False, 'telegram': {'notification_settings': {}}}
        self.bot._rpc = SimpleNamespace(_fiat_converter=None)

    def test_running_and_trade_notifications_keep_upstream_format(self):
        self.native_context()
        status = self.bot.compose_message(dict(type=self.types.STATUS, status='running'))
        self.assertEqual(status, '*Status:* `running`')
        for kind, wording in (
            (self.types.ENTRY, 'New Trade'),
            (self.types.ENTRY_FILL, 'New Trade filled'),
            (self.types.EXIT, 'Exiting'),
            (self.types.EXIT_FILL, 'Exited'),
        ):
            with self.subTest(kind=kind):
                msg = self.message(kind)
                msg.update(exchange='Binance', gain='profit', pair='BTC/USDT:USDT')
                text = self.bot.compose_message(msg)
                self.assertIn(wording, text)
                self.assertIn('*Direction:* `Long', text)
                self.assertNotIn('【', text)
        msg = self.message(self.types.ENTRY_CANCEL, reason='timeout')
        msg['exchange'] = 'Binance'
        self.assertIn('Cancelling enter Order', self.bot.compose_message(msg))

    def test_market_text_is_escaped_and_exception_uses_custom_notice(self):
        text = self.bot.compose_message(dict(type=self.types.STRATEGY_MSG, msg='A_B [待确认]'))
        self.assertEqual(text, r'A\_B \[待确认]')
        text = self.bot.compose_message(dict(type=self.types.EXCEPTION, status='worker_failed'))
        self.assertIn('【运行异常】', text)
        self.assertIn(r'worker\_failed', text)
