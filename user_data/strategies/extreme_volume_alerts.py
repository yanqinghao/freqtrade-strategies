"""Public intrabar volume notices, independent of trading plans and orders."""
import logging
from datetime import datetime, timezone
from math import isfinite

from market_monitor import MONITOR_PAIRS, HourlyMarketMonitor
from notification_format import notice, local_time

logger = logging.getLogger(__name__)
WIDTH = 300000


def extreme_candles(rows, now):
    """Compare actual accumulated base volume to 24 complete 5m candles."""
    now_ms = int(now.timestamp() * 1000)
    candles = sorted(
        {int(r[0]): list(r[:6]) for r in rows if len(r) >= 6 and r[0] <= now_ms}.values(),
        key=lambda r: r[0],
    )
    events = []
    for i in range(max(24, len(candles) - 2), len(candles)):
        current = candles[i]
        candle = int(current[0])
        if now_ms - candle >= WIDTH + 30000:
            continue
        window = candles[i - 24 : i + 1]
        if any(int(r[0]) % WIDTH for r in window) or any(
            int(b[0]) - int(a[0]) != WIDTH for a, b in zip(window, window[1:])
        ):
            continue
        try:
            volumes = [float(r[5]) for r in window]
            open_, high, low, close = (float(v) for v in current[1:5])
        except (TypeError, ValueError, OverflowError):
            continue
        if not all(isfinite(v) and v >= 0 for v in volumes):
            continue
        if not all(isfinite(v) and v > 0 for v in (open_, high, low, close)):
            continue
        if not low <= min(open_, close) <= max(open_, close) <= high:
            continue
        average = sum(volumes[:-1]) / 24
        if not isfinite(average) or average <= 0:
            continue
        ratio = volumes[-1] / average
        if not isfinite(ratio) or ratio < 10:
            continue
        events.append(
            dict(
                candle=candle,
                close=close,
                change=(close / open_ - 1) * 100,
                volume=volumes[-1],
                average=average,
                ratio=ratio,
                forming=now_ms < candle + WIDTH,
            )
        )
    return events


class ExtremeVolumeAlerts:
    def __init__(self, queue, clock=None):
        self.queue = queue
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def check(self, fetch):
        for pair in MONITOR_PAIRS:
            try:
                rows = fetch(pair, '5m')
                now = self._clock()
                for event in extreme_candles(rows, now):
                    coin = pair.split('/')[0]
                    direction = (
                        '上涨' if event['change'] > 0 else '下跌' if event['change'] < 0 else '震荡'
                    )
                    phase = '盘中·未收盘' if event['forming'] else '刚收盘'
                    opened = datetime.fromtimestamp(event['candle'] / 1000, timezone.utc)
                    message = notice(
                        '极端放量',
                        f'{coin} · {direction} · 5m{phase}',
                        summary=[f"量比：{event['ratio']:.2f}倍 / 本根涨跌 {event['change']:+.2f}%"],
                        sections=[
                            (
                                '行情',
                                [
                                    f"最新价：{event['close']:.8g}",
                                    f"本根累计：{event['volume']:.8g} {coin}",
                                    f"此前24根5m均量：{event['average']:.8g} {coin}",
                                    '本根开始：' + local_time(opened),
                                ],
                            )
                        ],
                        action='留意持仓与挂单；量能提示不代表趋势确认。',
                        at=now,
                    )
                    self.queue.put_market_once(
                        f"market-volume:{coin}:{event['candle']}",
                        message,
                        group=f'{coin}:{direction}',
                        now=now.timestamp(),
                    )
            except Exception as error:
                logger.warning(
                    'Extreme volume check unavailable for %s (%s)', pair, type(error).__name__
                )


class ExtremeVolumeMonitor(HourlyMarketMonitor):
    """Reuse public-data access and cleanup, with a separate worker and client."""

    def __init__(self, queue, **kwargs):
        super().__init__(alerts=ExtremeVolumeAlerts(queue), **kwargs)
        self._slot = None

    def tick(self, now):
        if self._future is not None:
            if not self._future.done():
                return
            try:
                self._future.result()
            except Exception as error:
                logger.warning('Extreme volume worker failed (%s)', type(error).__name__)
            self._future = None
        slot = int(now.timestamp()) // 10
        if self._slot != slot:
            self._slot = slot
            self._future = self._executor.submit(self._alerts.check, self._fetch)
