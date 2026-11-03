import importlib.util
import json
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'user_data' / 'strategies'))


class ExecutionHealthTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            importlib.util.find_spec('execution_health'), 'health publisher missing'
        )
        from execution_health import HeartbeatPublisher

        self.cls = HeartbeatPublisher
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'health.json'
        self.bot = SimpleNamespace(
            state=SimpleNamespace(name='RUNNING'),
            last_process=datetime.fromtimestamp(900, timezone.utc),
        )

    def test_live_publisher_does_not_fabricate_native_loop_progress(self):
        publisher = self.cls(self.bot, self.path)
        publisher.publish(1000.0)
        first = json.loads(self.path.read_text())
        publisher.publish(1300.0)
        second = json.loads(self.path.read_text())
        self.assertEqual(first['last_process'], 900.0)
        self.assertEqual(second['last_process'], 900.0)
        self.assertEqual(second['observed_at'], 1300.0)
        self.assertEqual(second['state'], 'RUNNING')

    def test_paused_stopped_and_initializing_native_states_are_preserved(self):
        publisher = self.cls(self.bot, self.path)
        for state in ('PAUSED', 'STOPPED', 'RELOAD_CONFIG'):
            self.bot.state = SimpleNamespace(name=state)
            self.bot.last_process = None
            publisher.publish(1000.0)
            data = json.loads(self.path.read_text())
            self.assertEqual(data['state'], state)
            self.assertIsNone(data['last_process'])

    def test_background_thread_publishes_and_closes_cleanly(self):
        publisher = self.cls(self.bot, self.path, interval=0.01)
        self.addCleanup(publisher.close)
        publisher.start()
        deadline = time.monotonic() + 2
        while not self.path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(self.path.exists())
        self.bot.last_process = datetime.fromtimestamp(999, timezone.utc)
        while (
            json.loads(self.path.read_text())['last_process'] != 999.0
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        self.assertEqual(json.loads(self.path.read_text())['last_process'], 999.0)
        publisher.close()
        self.assertFalse(publisher._thread.is_alive())
