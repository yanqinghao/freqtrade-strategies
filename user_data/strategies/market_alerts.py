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
        events.append(('volume', f'放量：此前20根均量的 {ratio:.2f} 倍', None))
    for label, reference in levels:
        resistance, support = reference['resistance'], reference['support']
        if resistance is not None and prior <= resistance < close:
            events.append(
                (
                    'breakout',
                    f'突破观察：{label}阻力 {resistance:.8g}'
                    f'（高于 {(close / resistance - 1) * 100:.2f}%）',
                    {'kind': 'breakout', 'label': label, 'price': resistance},
                )
            )
        if support is not None and prior >= support > close:
            events.append(
                (
                    'breakdown',
                    f'跌破观察：{label}支撑 {support:.8g}' f'（低于 {(1 - close / support) * 100:.2f}%）',
                    {'kind': 'breakdown', 'label': label, 'price': support},
                )
            )
    return events, close, (close / prior - 1) * 100, ratio


def _followup(record, pair, candle, close):
    pending = record.get('followup')
    if not pending:
        return None
    target = pending['candle'] + TIMEFRAMES['15m']
    if candle < target:
        return None
    record.pop('followup')
    if candle > target:
        return None  # Do not substitute a later close or replay stale market advice.
    stamp = datetime.fromtimestamp((candle + TIMEFRAMES['15m']) / 1000, timezone.utc)
    origin = datetime.fromtimestamp((pending['candle'] + TIMEFRAMES['15m']) / 1000, timezone.utc)
    lines = [
        f"{pair.split('/')[0]} 异动后续 · 原事件 {origin:%m-%d %H:%M} UTC",
        f'下一根15m收盘 {close:.8g}（{stamp:%m-%d %H:%M} UTC）',
    ]
    for reference in pending['levels']:
        price, kind = reference['price'], reference['kind']
        if kind == 'breakout':
            label = '阻力'
            outcome = '仍在位上' if close > price else '未守住，回到位下' if close < price else '未守住，收于原位'
        else:
            label = '支撑'
            outcome = '仍在位下' if close < price else '已收回' if close > price else '回到原位，未继续位下'
        lines.append(f"原{reference['label']}{label} {price:.8g}：{outcome}")
    lines.append('仅核对这一根收盘，不代表趋势确认；本次跟踪结束。')
    return '\n'.join(lines)


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
                followup = _followup(record, pair, candle, float(frame.close.iloc[-1]))
                if followup:
                    messages.append(followup)
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
                    for kind, _, _ in events
                    if kind != 'volume' and closed_at - cooldowns.get(kind, 0) >= 3600
                }
                lines = [
                    text
                    for kind, text, _ in events
                    if kind in eligible or (kind == 'volume' and eligible)
                ]
                record['last_candle'] = candle
                if lines:
                    if unavailable:
                        lines.append('参考周期暂不可用：' + '、'.join(unavailable))
                    cooldowns.update({kind: closed_at for kind in eligible})
                    record['followup'] = {
                        'candle': candle,
                        'levels': [level for kind, _, level in events if kind in eligible],
                    }
                    stamp = datetime.fromtimestamp(closed_at, timezone.utc)
                    next_check = datetime.fromtimestamp(closed_at + 900, timezone.utc)
                    volume = (
                        f'量比 {ratio:.2f}（' + ('达到2倍放量阈值' if ratio >= 2 else '未达2倍放量阈值') + '）'
                        if ratio is not None
                        else '量比不可用，无法判断量能'
                    )
                    message = '\n'.join(
                        [
                            f"{pair.split('/')[0]} 15分钟收盘异动 · 越线待观察",
                            f'收盘时间：{stamp:%Y-%m-%d %H:%M} UTC',
                            f'收盘价 {close:.8g} | 本根涨跌幅 {change:+.2f}% | {volume}',
                            *lines,
                            f'下一步：核对 {next_check:%m-%d %H:%M} UTC 的15m收盘' '与上述原位的关系，仅反馈一次；错过则不补发。',
                            '参考位来自本根开始前已收盘的4小时/日线，仅提醒，不触发下单。',
                        ]
                    )
                    if followup:
                        messages[-1] += '\n\n' + message
                    else:
                        messages.append(message)
            except Exception:
                logger.warning('Market alert unavailable for %s', pair, exc_info=True)
        if pairs != self._pairs:
            update_state(self.state_file, lambda state: state.update(pairs=pairs))
            self._pairs = pairs
        self._primed = primed
        return messages
