import asyncio
import importlib.util
import tempfile
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from test_telegram_assistant import load_telegram

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'user_data' / 'strategies'))


class AlertDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('assistant_alerts'))
        from assistant_alerts import AlertQueue

        self.cls = AlertQueue
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'alerts.sqlite'
        self.queue = AlertQueue(self.path)

    def test_restart_coalescing_and_ack_cannot_delete_newer_event(self):
        self.queue.put('stop:1', '需要处理', now=100)
        first = self.queue.pending(now=100)[0]
        self.queue.put('stop:1', '已恢复', now=101)
        self.queue.acknowledge(first)
        restored = self.cls(self.path)
        self.assertEqual([r['message'] for r in restored.pending(now=101)], ['已恢复'])
        restored.acknowledge(restored.pending(now=101)[0])
        self.assertEqual(restored.pending(now=102), [])

    def test_actual_send_failure_survives_restart_and_retries_after_delay(self):
        module = load_telegram()
        bot = object.__new__(module.Telegram)
        bot._rpc = SimpleNamespace(
            _freqtrade=SimpleNamespace(strategy=SimpleNamespace(alert_queue=self.queue))
        )
        bot._config = {'telegram': {'chat_id': 123, 'topic_id': 7}}
        sent = []

        async def fail(*args, **kwargs):
            raise TimeoutError('PRIVATE_TOKEN')

        async def success(*args, **kwargs):
            sent.append((args, kwargs))

        bot._app = SimpleNamespace(bot=SimpleNamespace(send_message=fail))
        self.queue.put('exit:1', '需要处理：退出失败', now=100)
        asyncio.run(bot._deliver_alerts_once(now=100))
        self.assertEqual(self.cls(self.path).pending(now=159), [])
        self.assertEqual(len(self.cls(self.path).pending(now=160)), 1)
        bot._app.bot.send_message = success
        asyncio.run(bot._deliver_alerts_once(now=160))
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0][1]['message_thread_id'], 7)
        self.assertIn('退出失败', sent[0][1]['text'])
        self.assertEqual(self.queue.pending(now=161), [])

    def test_delivery_is_bounded_and_corrupt_storage_is_not_erased(self):
        for i in range(6):
            self.queue.put(str(i), str(i), now=100)
        self.assertEqual(len(self.queue.pending(limit=3, now=100)), 3)
        bad = Path(self.temp.name) / 'bad.sqlite'
        bad.write_bytes(b'broken')
        with self.assertRaises(Exception):
            self.cls(bad).put('x', 'x')
        self.assertEqual(bad.read_bytes(), b'broken')

    def test_cancelled_or_replaced_watch_is_rechecked_before_starting_send(self):
        from condition_watch import WatchStore, evaluate

        module = load_telegram()
        bot = object.__new__(module.Telegram)
        bot._rpc = SimpleNamespace(
            _freqtrade=SimpleNamespace(strategy=SimpleNamespace(alert_queue=self.queue))
        )
        bot._config = {'telegram': {'chat_id': 123}}
        store = WatchStore(self.queue)
        spec = dict(
            mode='reject',
            side='short',
            pair='BTC/USDT:USDT',
            tf='1h',
            low=100,
            high=102,
            confirm=99,
            invalid=None,
            window=3,
            ttl=86400,
        )
        w = store.create(spec, [123, 7, None], 'create', now=100)
        new, message = evaluate(w, [], dict(price=101, time=101), 101)
        store.apply(w, new, message, now=101)
        self.queue.put('critical:first', 'urgent', now=102)
        sent = []

        async def send(*args, **kwargs):
            sent.append(kwargs['text'])
            if len(sent) == 1:
                store.cancel(w['id'], [123, 7, None], now=103)

        bot._app = SimpleNamespace(bot=SimpleNamespace(send_message=send))
        asyncio.run(bot._deliver_alerts_once(now=104))
        self.assertEqual(len(sent), 1)
        self.assertEqual(self.queue.pending(now=104), [])
