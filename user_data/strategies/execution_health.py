"""Publish read-only native execution progress independently of the trading loop."""
import logging
import time
from pathlib import Path
from threading import Event, Thread

from assistant_state import update_state

logger = logging.getLogger(__name__)


class HeartbeatPublisher:
    def __init__(self, bot, path, *, interval=15):
        self.bot = bot
        self.path = Path(path)
        self.interval = interval
        self._stop = Event()
        self._thread = None

    def _state_name(self):
        state = self.bot.state
        return getattr(state, 'name', str(state)).upper()

    def publish(self, now=None):
        last = self.bot.last_process
        snapshot = {
            'observed_at': time.time() if now is None else now,
            'last_process': last.timestamp() if last is not None else None,
            'state': self._state_name(),
        }
        update_state(self.path, lambda document: document.update(snapshot))

    def _publish_safely(self):
        try:
            self.publish()
        except Exception as exc:
            logger.warning('Health snapshot publication failed (%s)', type(exc).__name__)

    def _run(self):
        while not self._stop.is_set():
            self._publish_safely()
            self._stop.wait(self.interval)

    def start(self):
        if self._thread is None:
            self._thread = Thread(target=self._run, name='execution-health', daemon=True)
            self._thread.start()
        return self

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._state_name() in ('STOPPED', 'PAUSED'):
            self._publish_safely()
