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

    def test_submit_and_fill_have_different_states(self):
        submitted = self.bot.compose_message(self.message(self.types.ENTRY))
        filled = self.bot.compose_message(self.message(self.types.ENTRY_FILL))
        self.assertIn('已提交', submitted)
        self.assertNotIn('已成交', submitted)
        self.assertIn('成交回执', filled)
        self.assertIn('0.0361', filled)
        self.assertIn(r'A\_B/USDT', filled)

    def test_exit_estimate_actual_and_final_totals_keep_different_basis(self):
        pending = self.bot.compose_message(self.message(self.types.EXIT))
        final = self.bot.compose_message(self.message(self.types.EXIT_FILL, is_final_exit=True))
        self.assertIn('预估盈亏（待成交回执）', pending)
        self.assertNotIn('本次已实现', pending)
        self.assertIn('本次已实现', final)
        self.assertIn('+2.4000 USDT', final)
        self.assertIn('整笔累计', final)
        self.assertIn('+7.4000 USDT', final)
        self.assertIn('+7.40%', final)

    def test_cancel_does_not_claim_zero_fill_or_full_exit(self):
        text = self.bot.compose_message(self.message(self.types.ENTRY_CANCEL, reason='timeout'))
        self.assertIn('撤单状态', text)
        self.assertIn('timeout', text)
        self.assertNotIn('零成交', text)
        self.assertNotIn('已全平', text)

    def test_increase_reports_total_margin_and_strategy_text_is_escaped(self):
        msg = self.message(self.types.ENTRY_FILL)
        msg.update(sub_trade=True, stake_amount=100)
        text = self.bot.compose_message(msg)
        self.assertIn('加仓成交回执', text)
        self.assertIn('保证金总额：100 USDT', text)
        self.assertNotIn('新增保证金', text)
        text = self.bot.compose_message(dict(type=self.types.STRATEGY_MSG, msg='A_B [待确认]'))
        self.assertEqual(text, r'A\_B \[待确认]')

    def test_exit_cancel_kept_open_is_not_presented_as_cancelled(self):
        from freqtrade.constants import CANCEL_REASON

        text = self.bot.compose_message(
            self.message(self.types.EXIT_CANCEL, reason=CANCEL_REASON['PARTIALLY_FILLED_KEEP_OPEN'])
        )
        self.assertIn('尚未撤销', text)
        self.assertIn('订单仍开放', text)
        self.assertNotIn('已撤单', text)
