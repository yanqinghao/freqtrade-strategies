"""Advisory daily/4h reports. This module has no access to trading plans or orders."""
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from math import isfinite
from threading import Lock

import pandas as pd

logger = logging.getLogger(__name__)
MONITOR_PAIRS = ('BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT')
TIMEFRAMES = {'1d': 86400000, '4h': 14400000, '1h': 3600000, '15m': 900000}


def _closed_frame(rows, timeframe, now):
    """Use the latest 120 closed candles; never use a forming candle."""
    width = TIMEFRAMES[timeframe]
    now_ms = int(now.timestamp() * 1000)
    closed = [r for r in rows if len(r) >= 6 and r[0] + width <= now_ms]
    closed = sorted({r[0]: r for r in closed}.values(), key=lambda r: r[0])[-120:]
    if len(closed) < 50:
        raise ValueError('至少需要50根已收盘K线')
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


def build_reports(fetch, now):
    """Generate exactly one report per monitored coin, isolating data failures."""
    reports = []
    for pair in MONITOR_PAIRS:
        lines = [f"{pair.split('/')[0]} 日线 / 4小时监控", f"生成时间：{now:%Y-%m-%d %H:%M} UTC"]
        trends = {}
        for timeframe, label in (('1d', '日线'), ('4h', '4小时'), ('1h', '1小时'), ('15m', '15分钟')):
            try:
                rows = fetch(pair, timeframe)
                result = (
                    analyze_candles(rows, timeframe, now)
                    if timeframe in ('1d', '4h')
                    else analyze_trend(rows, timeframe, now)
                )
                trends[timeframe] = result['trend']
                lines.extend(
                    [
                        f"\n{label}：{result['trend']}（收盘 {result['closed_at']:%m-%d %H:%M} UTC）",
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
        lines.append('\n走势分区：' + classify_regime(trends))
        lines.append('\n点位仅来自已收盘日线与4小时K线；1小时、15分钟仅判断当前走势。不触发下单。')
        reports.append('\n'.join(lines))
    return reports


class HourlyMarketMonitor:
    """Fetch public data on a dedicated worker, drain reports on the bot thread."""

    def __init__(self, fetch=None, *, exchange_name='binance', futures=True, executor=None):
        self._client = None
        self._exchange_name = exchange_name
        self._futures = futures
        self._fetch = fetch or self._fetch_public
        self._executor = executor or ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='market-monitor'
        )
        self._future = None
        self._hour = None
        self._lock = Lock()

    def _fetch_public(self, pair, timeframe):
        if self._client is None:
            import ccxt

            self._client = getattr(ccxt, self._exchange_name)(
                {
                    'enableRateLimit': True,
                    'timeout': 10000,
                    'options': {'defaultType': 'future' if self._futures else 'spot'},
                }
            )
        symbol = pair if self._futures else pair.split(':')[0]
        return self._client.fetch_ohlcv(symbol, timeframe=timeframe, limit=121)

    def reports(self, now=None):
        with self._lock:
            return build_reports(self._fetch, now or datetime.now(timezone.utc))

    def tick(self, now):
        messages = []
        if self._future is not None:
            if not self._future.done():
                return messages
            try:
                messages = self._future.result()
            except Exception:
                logger.exception('Hourly monitoring failed')
            self._future = None
        hour = int(now.timestamp()) // 3600
        if self._hour != hour:
            self._hour = hour
            self._future = self._executor.submit(self.reports, now)
        return messages

    def close(self):
        self._executor.shutdown(wait=False, cancel_futures=True)
