"""Closed-candle public-market alerts; no trading plans, credentials or orders."""
import logging
from copy import deepcopy
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path

from assistant_state import read_state, update_state
from market_monitor import MONITOR_PAIRS, TIMEFRAMES, _closed_frame, analyze_candles

logger = logging.getLogger(__name__)


def _events(frame, levels):
    previous, latest = frame.iloc[-2], frame.iloc[-1]
    close, prior = float(latest.close), float(previous.close)
    ratio = None
    try:
        volumes = [float(v) for v in frame.volume.iloc[-21:]]
        if all(isfinite(v) and v >= 0 for v in volumes):
            average = sum(volumes[:-1]) / 20
            if isfinite(average) and average > 0:
                ratio = volumes[-1] / average
                if not isfinite(ratio):
                    ratio = None
    except (TypeError, ValueError, OverflowError):
        pass  # Unusable volume must not suppress an otherwise valid price cross.
    events = []
    if ratio is not None and ratio >= 2:
        events.append(('volume', f'放量：此前20根均量的 {ratio:.2f} 倍'))
    for label, reference in levels:
        resistance, support = reference['resistance'], reference['support']
        if resistance is not None and prior <= resistance < close:
            events.append(('breakout', f'突破：{label}阻力 {resistance:.8g}'))
        if support is not None and prior >= support > close:
            events.append(('breakdown', f'跌破：{label}支撑 {support:.8g}'))
    return events, close, (close / prior - 1) * 100, ratio


class MarketAlerts:
    """Record a candle before publishing; restart establishes a fresh baseline."""

    def __init__(self, state_file):
        self.state_file = Path(state_file)
        self._pairs = None
        self._primed = set()

    def check(self, fetch, now):
        if self._pairs is None:
            self._pairs = read_state(self.state_file).get('pairs', {})
            if not isinstance(self._pairs, dict):
                self._pairs = None
                raise ValueError('异动监控状态无效')
        pairs = deepcopy(self._pairs)
        primed = set(self._primed)
        messages = []
        for pair in MONITOR_PAIRS:
            try:
                frame = _closed_frame(fetch(pair, '15m'), '15m', now, minimum=21)
                candle = int(frame.timestamp.iloc[-1])
                if any(
                    int(d) != TIMEFRAMES['15m'] for d in frame.timestamp.tail(21).diff().dropna()
                ):
                    raise ValueError('15分钟行情存在缺失K线')
                record = pairs.setdefault(pair, {'last_candle': candle, 'cooldowns': {}})
                if pair not in primed:
                    record['last_candle'] = max(candle, record['last_candle'])
                    primed.add(pair)
                    continue
                if candle <= record['last_candle']:
                    continue
                # Freeze references at the event candle's OPEN, before its move.
                cutoff = datetime.fromtimestamp(candle / 1000, timezone.utc)
                levels, unavailable = [], []
                for timeframe, label in (('4h', '4小时'), ('1d', '日线')):
                    try:
                        levels.append(
                            (label, analyze_candles(fetch(pair, timeframe), timeframe, cutoff))
                        )
                    except Exception:
                        unavailable.append(label)
                        logger.warning(
                            'Alert reference unavailable for %s %s', pair, timeframe, exc_info=True
                        )
                events, close, change, ratio = _events(frame, levels)
                closed_at = (candle + TIMEFRAMES['15m']) / 1000
                cooldowns = record['cooldowns']
                eligible = {
                    kind
                    for kind, _ in events
                    if kind != 'volume' and closed_at - cooldowns.get(kind, 0) >= 3600
                }
                lines = [
                    text
                    for kind, text in events
                    if kind in eligible or (kind == 'volume' and eligible)
                ]
                record['last_candle'] = candle
                if lines:
                    if unavailable:
                        lines.append('参考周期暂不可用：' + '、'.join(unavailable))
                    cooldowns.update({kind: closed_at for kind in eligible})
                    stamp = datetime.fromtimestamp(closed_at, timezone.utc)
                    volume = f'量比 {ratio:.2f}' if ratio is not None else '量比不可用'
                    messages.append(
                        '\n'.join(
                            [
                                f"{pair.split('/')[0]} 15分钟收盘异动",
                                f'收盘时间：{stamp:%Y-%m-%d %H:%M} UTC',
                                f'收盘价 {close:.8g} | 本根涨跌幅 {change:+.2f}% | {volume}',
                                *lines,
                                '下一步观察：'
                                + (
                                    '上下方向信号冲突，等待后续收盘确认。'
                                    if {'breakout', 'breakdown'} <= eligible
                                    else '后续15m收盘能否守住上述阻力位；回落其下则突破未站稳。'
                                    if 'breakout' in eligible
                                    else '后续15m收盘能否收回上述支撑位；收回则跌破未延续。'
                                ),
                                '参考位来自本根开始前已收盘的4小时/日线，仅提醒，不触发下单。',
                            ]
                        )
                    )
            except Exception:
                logger.warning('Market alert unavailable for %s', pair, exc_info=True)
        if pairs != self._pairs:
            update_state(self.state_file, lambda state: state.update(pairs=pairs))
            self._pairs = pairs
        self._primed = primed
        return messages
