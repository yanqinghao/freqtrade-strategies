"""Public candles/book on a dedicated worker; no order or account access."""
import logging
import time
from math import isfinite
from queue import SimpleQueue, Empty
from market_monitor import HourlyMarketMonitor
from entry_volume import checked_rows, analyze_volume
from entry_pause import PAIRS

logger = logging.getLogger(__name__)


class EntryPauseMonitor(HourlyMarketMonitor):
    def __init__(self, *, clock=time.time, **kwargs):
        super().__init__(**kwargs)
        self.clock = clock
        self._last = {}
        self._continuous = {}
        self._scheduled = 0
        self._updates = SimpleQueue()
        self._signals = {}

    def collect(self, needs):
        volumes, results = {}, {}
        # Publish valid volume immediately, before historical backfill or book I/O.
        # Those can block restoration, but must not discard known cancellation evidence.
        for pair in needs:
            try:
                observed = self.clock()
                volume = (
                    self._fetch_public(pair, '5m')
                    if self._client is None
                    else self._client.fetch_ohlcv(pair, timeframe='5m', limit=28)
                )
                current = self.clock()
                fresh = {
                    direction: analyze_volume(volume, observed, current, direction)
                    for direction in ('long', 'short')
                }
                signals = dict(self._signals.get(pair, {}))
                for direction, signal in fresh.items():
                    if signal and signal['at'] >= signals.get(direction, {}).get('at', 0):
                        signals[direction] = signal
                self._signals[pair] = signals
                if observed - self._last.get(pair, 0) > 30:
                    self._continuous[pair] = observed
                self._last[pair] = observed
                snapshot = dict(
                    observed_at=observed,
                    volume=volume,
                    signals=signals,
                    continuous_since=self._continuous[pair],
                )
                volumes[pair] = snapshot
                self._updates.put((pair, snapshot))
            except Exception as error:
                logger.warning(
                    'Public entry volume unavailable for %s (%s)', pair, type(error).__name__
                )
        for pair, snapshot in volumes.items():
            try:
                observed = snapshot['observed_at']
                minute = []
                cursor = int(needs[pair]) // 60 * 60000
                while cursor <= int(observed) // 60 * 60000:
                    batch = self._client.fetch_ohlcv(pair, timeframe='1m', since=cursor, limit=1000)
                    if not batch or batch[-1][0] < cursor:
                        raise ValueError('历史行情不可用')
                    minute.extend(r for r in batch if r[0] >= cursor)
                    cursor = batch[-1][0] + 60000
                book = self._client.fetch_order_book(pair, limit=5)
                bid, ask = book['bids'][0][0], book['asks'][0][0]
                checked_rows(minute, 60, observed, self.clock(), 2)
                if not all(isfinite(v) and v > 0 for v in (bid, ask)) or bid > ask:
                    raise ValueError('盘口无效')
                results[pair] = {**snapshot, 'minute': minute, 'bid': bid, 'ask': ask}
                self._updates.put((pair, results[pair]))
            except Exception as error:
                logger.warning(
                    'Public entry history/book unavailable for %s (%s)', pair, type(error).__name__
                )
        return results

    def drain(self):
        results = {}
        while True:
            try:
                pair, snapshot = self._updates.get_nowait()
                results[pair] = snapshot
            except Empty:
                return results

    def tick(self, now, plans):
        if self._future is not None:
            if not self._future.done():
                return self.drain()
            try:
                self._future.result()
            except Exception as error:
                logger.warning('Entry market worker unavailable (%s)', type(error).__name__)
            self._future = None
        results = self.drain()
        stamp = now.timestamp()
        if stamp - self._scheduled >= 10:
            needs = {pair: stamp - 120 for pair in PAIRS}
            for meta in plans:
                if meta['phase'] not in ('filled', 'done') and meta['request']['pair'] in needs:
                    pair = meta['request']['pair']
                    needs[pair] = min(needs[pair], meta.get('checked_until', meta['created']))
            self._scheduled = stamp
            self._future = self._executor.submit(self.collect, needs)
        return results
