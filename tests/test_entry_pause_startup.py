"""Exercise the resolver's temporary import path in a fresh interpreter."""
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class EntryPauseStartupTests(unittest.TestCase):
    def test_native_resolver_then_telegram_and_monitor_start_without_strategy_on_sys_path(self):
        script = textwrap.dedent(
            '''
            import importlib.util
            import sys
            from pathlib import Path
            from types import SimpleNamespace
            from unittest.mock import patch
            from freqtrade.resolvers import StrategyResolver
            from freqtrade.enums import RunMode, State

            root = Path(sys.argv[1])
            config = dict(strategy='KamaFama_Dynamic',
                          strategy_path=str(root / 'user_data/strategies'),
                          user_data_dir=Path.cwd(), runmode=RunMode.DRY_RUN,
                          trading_mode='futures', stake_currency='USDT')
            strategy = StrategyResolver.load_strategy(config)
            assert str(root / 'user_data/strategies') not in sys.path
            try:
                import freqtrade.rpc._telegram_base as upstream
            except ModuleNotFoundError:
                import freqtrade.rpc.telegram as upstream
                sys.modules['freqtrade.rpc._telegram_base'] = upstream
            for name in ('order_management', 'order_panel', 'watch_panel', 'notification_format'):
                qualified = 'freqtrade.rpc.' + name
                spec = importlib.util.spec_from_file_location(
                    qualified, root / 'user_data/strategies' / (name + '.py'))
                module = importlib.util.module_from_spec(spec)
                sys.modules[qualified] = module
                spec.loader.exec_module(module)
            spec = importlib.util.spec_from_file_location('startup_telegram', root / 'telegram.py')
            telegram = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(telegram)
            def forbidden(*args, **kwargs):
                raise AssertionError('startup test must not access accounts or submit orders')
            exchange = SimpleNamespace(fetch_order=forbidden, cancel_order_with_result=forbidden,
                                       create_order=forbidden)
            bot = SimpleNamespace(strategy=strategy, exchange=exchange,
                                  state=State.RUNNING, last_process=None)
            rpc = SimpleNamespace(_freqtrade=bot)
            # Only suppress Telegram's external startup thread. Keep our constructor real.
            with patch.object(upstream.Telegram, '__init__', return_value=None):
                telegram.Telegram(rpc, config)
            strategy.bot_start()
            try:
                assert strategy.entry_pause is not None
                assert strategy.entry_pause.monitor is not None
                assert strategy.position_monitor is not None
                assert exchange._assistant_risk_guard
                assert exchange._assistant_entry_receipts
            finally:
                strategy._health_publisher.close()
                for monitor in (strategy.market_monitor, strategy.extreme_volume_monitor,
                                strategy.condition_monitor, strategy.entry_pause.monitor,
                                strategy.position_monitor):
                    monitor.close()
        '''
        )
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, '-c', script, str(ROOT)],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=30,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
