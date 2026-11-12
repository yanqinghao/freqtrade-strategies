"""Advisory reports and closed-candle alerts, with no trading-plan/order access."""
import logging
import time
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from math import isfinite
from threading import Lock

import pandas as pd
from notification_format import notice, local_time

logger = logging.getLogger(__name__)
MONITOR_PAIRS = ('BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT')
TIMEFRAMES = {'1d': 86400000, '4h': 14400000, '1h': 3600000, '15m': 900000}


def _closed_frame(rows, timeframe, now, minimum=50):
    """Use the latest 120 closed candles; never use a forming candle."""
    width = TIMEFRAMES[timeframe]
    now_ms = int(now.timestamp() * 1000)
    closed = [r for r in rows if len(r) >= 6 and r[0] + width <= now_ms]
    closed = sorted({r[0]: r for r in closed}.values(), key=lambda r: r[0])[-120:]
    if len(closed) < minimum:
        raise ValueError(f'至少需要{minimum}根已收盘K线')
    if any(not all(isfinite(float(v)) and float(v) > 0 for v in r[1:5]) for r in closed):
        raise ValueError('K线价格无效')
    if now_ms - (closed[-1][0] + width) >= width:
        raise ValueError('行情已过期')
    return pd.DataFrame(closed, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])


def analyze_trend(rows, timeframe, now):
    frame = _closed_frame(rows, timeframe, now)
    close = float(frame.close.iloc[-1])
    ema20 = float(frame.close.ewm(span=20, adjust=False).mean().iloc[-1])
    ema50 = float(frame.close.ewm(span=50, adjust=False).mean().iloc[-1])
    trend = '偏多' if close > ema20 > ema50 else '偏空' if close < ema20 < ema50 else '震荡'
    return {
        'close': close,
        'trend': trend,
        'ema20': ema20,
        'ema50': ema50,
        'closed_at': datetime.fromtimestamp(
            (frame.timestamp.iloc[-1] + TIMEFRAMES[timeframe]) / 1000, timezone.utc
        ),
    }


def analyze_candles(rows, timeframe, now):
    if timeframe not in ('1d', '4h'):
        raise ValueError('参考点位仅允许日线与4小时数据')
    frame = _closed_frame(rows, timeframe, now)
    result = analyze_trend(rows, timeframe, now)
    close = result['close']
    recent = frame.tail(60).reset_index(drop=True)
    supports, resistances = [], []
    for i in range(2, len(recent) - 2):
        if recent.low[i] == recent.low.iloc[i - 2 : i + 3].min() and recent.low[i] < close:
            supports.append(float(recent.low[i]))
        if recent.high[i] == recent.high.iloc[i - 2 : i + 3].max() and recent.high[i] > close:
            resistances.append(float(recent.high[i]))
    low, high = float(recent.low.min()), float(recent.high.max())
    return {
        **result,
        'support': max(supports) if supports else low if low < close else None,
        'resistance': min(resistances) if resistances else high if high > close else None,
    }


def classify_regime(trends):
    if len(trends) < 4:
        return '部分周期缺少有效数据，暂不分区'
    daily, h4, h1, m15 = (trends[t] for t in ('1d', '4h', '1h', '15m'))
    if daily == h4 == '偏多':
        return (
            '大周期偏多，短线回调区' if h1 == m15 == '偏空' else '多头顺势区' if h1 == m15 == '偏多' else '大周期偏多，短线整理区'
        )
    if daily == h4 == '偏空':
        return (
            '大周期偏空，短线反弹区' if h1 == m15 == '偏多' else '空头顺势区' if h1 == m15 == '偏空' else '大周期偏空，短线整理区'
        )
    return '多周期震荡区' if daily == h4 == h1 == m15 == '震荡' else '大周期分歧区，短线方向不能替代日线与4小时判断'


def observation(trends):
    if len(trends) < 4:
        return '数据不完整，暂不判断；等待行情恢复后再观察。'
    daily, h4 = trends['1d'], trends['4h']
    if daily != h4 or daily == '震荡':
        return '大周期未同向；先观察4小时区间边界能否收盘突破，短线波动不足以确认方向。'
    if daily == '偏多':
        return '回踩4小时支撑后能否守住、短线能否转强；跌破支撑则重新评估偏多判断。'
    return '反弹4小时阻力后能否受阻、短线能否转弱；站上阻力则重新评估偏空判断。'


def level_distance(level, close, *, support):
    if level is None:
        return '暂无'
    if close is None:
        return f'{level:.8g}（距离不可用）'
    distance = (close - level) / close * 100
    if not isfinite(distance):
        return f'{level:.8g}（距离不可用）'
    relation = (
        ('距支撑' if distance >= 0 else '已低于支撑') if support else ('距阻力' if distance <= 0 else '已高于阻力')
    )
    return f'{level:.8g}（{relation} {abs(distance):.2f}%）'


def build_reports(fetch, now, *, compact=False):
    """One automatic digest, or detailed per-coin reports on explicit request."""
    reports = []
    summaries = []
    for pair in MONITOR_PAIRS:
        lines = []
        trends = {}
        results = {}
        volume_note = ''
        for timeframe, label in (('1d', '日线'), ('4h', '4小时'), ('1h', '1小时'), ('15m', '15分钟')):
            try:
                rows = fetch(pair, timeframe)
                result = (
                    analyze_candles(rows, timeframe, now)
                    if timeframe in ('1d', '4h')
                    else analyze_trend(rows, timeframe, now)
                )
                trends[timeframe] = result['trend']
                results[timeframe] = result
                if timeframe == '15m':
                    frame = _closed_frame(rows, timeframe, now)
                    try:
                        volumes = [float(value) for value in frame.volume.iloc[-21:]]
                    except (ValueError, TypeError, OverflowError):
                        volumes = []
                    if volumes and all(isfinite(value) and value >= 0 for value in volumes):
                        mean = sum(volumes[:-1]) / 20
                        if isfinite(mean) and mean > 0 and volumes[-1] / mean >= 2:
                            volume_note = f'；15m放量 {volumes[-1] / mean:.2f}倍（单独放量不确认方向）'
                lines.extend(
                    [
                        f"\n{label}：{result['trend']}（收盘 {local_time(result['closed_at'])}）",
                        f"收盘参考价 {result['close']:.8g} | "
                        f"EMA20 {result['ema20']:.8g} | EMA50 {result['ema50']:.8g}",
                    ]
                )
                if timeframe in ('1d', '4h'):
                    support = (
                        f"{result['support']:.8g}" if result['support'] is not None else '区间内暂无'
                    )
                    resistance = (
                        f"{result['resistance']:.8g}"
                        if result['resistance'] is not None
                        else '区间内暂无'
                    )
                    lines.append(f"参考支撑 {support} | 参考阻力 {resistance}")
            except Exception:
                logger.warning(
                    'Market report unavailable for %s %s', pair, timeframe, exc_info=True
                )
                lines.append(f'\n{label}：行情暂不可用。')
        reports.append(
            notice(
                '行情详情',
                pair.split('/')[0],
                summary=[classify_regime(trends)],
                sections=[('周期与点位', lines)],
                action=observation(trends) + ' 参考已收盘K线，不触发下单。',
                at=now,
            )
        )
        summary = [
            pair.split('/')[0] + ' · ' + classify_regime(trends),
            '日线 ' + trends.get('1d', '不可用') + ' / 4h ' + trends.get('4h', '不可用'),
            '1h ' + trends.get('1h', '不可用') + ' / 15m ' + trends.get('15m', '不可用'),
        ]
        reference = results.get('15m')
        if '15m' in results:
            summary.append(
                f"15m收盘 {reference['close']:.8g}"
                f"（{local_time(reference['closed_at'])}）{volume_note}"
            )
        else:
            summary.append('15m收盘价不可用，暂不计算距离')
        result = results.get('4h')
        if result:
            close = reference['close'] if reference else None
            support = level_distance(result['support'], close, support=True)
            resistance = level_distance(result['resistance'], close, support=False)
            summary.extend([f'4h支撑：{support}', f'4h阻力：{resistance}'])
        else:
            summary.append('4h行情暂不可用')
        summary.append('观察：' + observation(trends))
        summaries.append('\n'.join(summary))
    if compact:
        return [
            notice(
                '行情简报',
                'BTC / ETH / SOL',
                sections=[('', [part]) for part in summaries],
                action='/monitor 查看周期详情；仅为观察条件，不是开仓信号。',
                at=now,
            )
        ]
    return reports


class SharedVolumeCandles:
    """Coalesce concurrent 5m reads without changing the original sample time."""

    def __init__(self, clock=time.time):
        self._clock = clock
        self._lock = Lock()
        self._pair_locks = {}
        self._samples = {}

    def get(self, pair, fetch, *, refresh=False):
        if refresh:
            # Live plans never wait behind advisory I/O, even for the same coin.
            observed = self._clock()
            rows = fetch()
            self._publish(pair, observed, rows)
            return observed, deepcopy(rows)
        with self._lock:
            pair_lock = self._pair_locks.setdefault(pair, Lock())
        with pair_lock:
            now = self._clock()
            observed, rows = self._samples.get(pair, (0, None))
            if rows is None or not (0 <= now - observed < 10 and now // 300 == observed // 300):
                rows = fetch()
                observed = now  # Time before I/O, so a slow request cannot extend freshness.
                self._publish(pair, observed, rows)
            return observed, deepcopy(rows)

    def _publish(self, pair, observed, rows):
        with self._lock:
            if observed >= self._samples.get(pair, (0, None))[0]:
                self._samples[pair] = (observed, rows)


class HourlyMarketMonitor:
    """Fetch public data on a dedicated worker, drain reports on the bot thread."""

    def __init__(
        self,
        fetch=None,
        *,
        exchange_name='binance',
        futures=True,
        executor=None,
        alerts=None,
        volume_candles=None,
    ):
        self._client = None
        self._exchange_name = exchange_name
        self._futures = futures
        self._fetch = fetch or self._fetch_public
        self._executor = executor or ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='market-monitor'
        )
        self._future = None
        self._hour = None
        self._minute = None
        self._lock = Lock()
        self._alerts = alerts
        self._volume_candles = volume_candles
        self._volume_observed_at = None

    def _get_client(self):
        if self._client is None:
            import ccxt

            self._client = getattr(ccxt, self._exchange_name)(
                {
                    'enableRateLimit': True,
                    'timeout': 10000,
                    'options': {'defaultType': 'future' if self._futures else 'spot'},
                }
            )
        return self._client

    def _fetch_public(self, pair, timeframe, *, refresh=False):
        if timeframe == '5m' and self._volume_candles is not None:
            observed, rows = self._volume_candles.get(
                pair, lambda: self._fetch_uncached(pair, timeframe), refresh=refresh
            )
            self._volume_observed_at = observed
            return rows
        return self._fetch_uncached(pair, timeframe)

    def _fetch_uncached(self, pair, timeframe):
        symbol = pair if self._futures else pair.split(':')[0]
        return self._get_client().fetch_ohlcv(symbol, timeframe=timeframe, limit=121)

    def reports(self, now=None):
        with self._lock:
            return build_reports(self._fetch, now or datetime.now(timezone.utc))

    def _run_cycle(self, now, hourly):
        with self._lock:
            cache = {}

            def fetch(pair, timeframe):
                key = pair, timeframe
                if key not in cache:
                    cache[key] = self._fetch(pair, timeframe)
                return cache[key]

            messages = build_reports(fetch, now, compact=True) if hourly else []
            if self._alerts is not None:
                try:
                    alerts = self._alerts.check(fetch, now)
                    if alerts:
                        messages.append('\n\n'.join(alerts))
                except Exception:
                    logger.exception('Closed-candle monitoring failed; hourly reports retained')
            return messages

    def tick(self, now):
        messages = []
        if self._future is not None:
            if not self._future.done():
                return messages
            try:
                messages = self._future.result()
            except Exception:
                logger.exception('Market monitoring failed')
            self._future = None
        hour = int(now.timestamp()) // 3600
        minute = int(now.timestamp()) // 60
        hourly = self._hour != hour
        if hourly or (self._alerts is not None and self._minute != minute):
            self._hour = hour
            self._minute = minute
            self._future = self._executor.submit(self._run_cycle, now, hourly)
        return messages

    def close(self):
        self._executor.shutdown(wait=False, cancel_futures=True)
