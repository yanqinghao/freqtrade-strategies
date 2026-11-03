import asyncio
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != ROOT]


def load_telegram():
    # Docker keeps the official Telegram implementation under this private name.
    try:
        import freqtrade.rpc._telegram_base as upstream
    except ModuleNotFoundError:
        import freqtrade.rpc.telegram as upstream
    sys.modules['freqtrade.rpc._telegram_base'] = upstream
    spec = importlib.util.spec_from_file_location('assistant_telegram', ROOT / 'telegram.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TelegramAssistantTests(unittest.TestCase):
    def setUp(self):
        self.module = load_telegram()

    def test_entry_parser_accepts_only_explicit_single_order(self):
        self.assertTrue(hasattr(self.module, 'parse_entry'), 'simple manual order parser missing')
        plan = self.module.parse_entry(
            ['BTC', '100', '3', 'price=100', 'sl=90', 'tp=110,120,130'], 'long'
        )
        self.assertEqual(plan['pair'], 'BTC/USDT:USDT')
        self.assertEqual(plan['size'], 100.0)
        self.assertEqual(plan['exit_points'], [110.0, 120.0, 130.0])
        for args in (
            ['BTC', '100', '3', 'sl=nan'],
            ['BTC', '-1', '3', 'sl=90'],
            ['BTC', '100', '3', 'entries=100,95', 'sl=90'],
            ['BTC', '100', '3', '110', '120', '130', '90', '100,95'],
            ['BTC', '100', '3', 'sl=90', 'tp=110,120,130,140'],
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.module.parse_entry(args, 'long')

    def test_handler_registry_has_manual_operations_but_no_retired_commands(self):
        cls = self.module.Telegram
        self.assertTrue(hasattr(cls, '_assistant_handlers'), 'explicit command registry missing')
        bot = object.__new__(cls)
        handlers = bot._assistant_handlers()
        commands = set().union(*(h.commands for h in handlers if hasattr(h, 'commands')))
        self.assertTrue(
            {'forcelong', 'forceshort', 'forceexit', 'setmanual', 'manual', 'monitor', 'close'}
            <= commands
        )
        self.assertFalse(
            commands
            & {
                'ai',
                'analysis',
                'prompt',
                'promptjson',
                'setpairstrategy',
                'setpairstrategyauto',
                'hedge',
                'hglong',
                'hgshort',
                'rtd',
                'restoremanual',
            }
        )
        monitor = next(h for h in handlers if 'monitor' in getattr(h, 'commands', ()))
        self.assertFalse(monitor.block, 'slow reports must not block urgent Telegram commands')

    def test_monitor_sends_three_plain_text_reports_without_creating_orders(self):
        self.assertTrue(
            hasattr(self.module.Telegram, '_monitor'), 'advisory report command missing'
        )
        bot = object.__new__(self.module.Telegram)
        sent = []

        async def send(message, **kwargs):
            sent.append((message, kwargs))

        bot._send_msg = send
        bot._rpc = SimpleNamespace(
            _freqtrade=SimpleNamespace(
                strategy=SimpleNamespace(
                    get_market_reports=lambda: ['BTC report', 'ETH report', 'SOL report']
                )
            )
        )
        asyncio.run(bot._monitor.__wrapped__(bot, None, None))
        self.assertEqual([s[0] for s in sent], ['BTC report', 'ETH report', 'SOL report'])
        self.assertTrue(all(s[1].get('parse_mode') is None for s in sent))


if __name__ == '__main__':
    unittest.main()
