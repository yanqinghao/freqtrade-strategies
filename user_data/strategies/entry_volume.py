"""Causal volume and price checks for temporarily withdrawn initial limit entries."""
from copy import deepcopy
from datetime import datetime, timezone
from math import isfinite

COOLDOWN = 900
MAX_AGE = 30


class EntryDataError(ValueError):
    """A controlled validation reason, safe to display without network payloads."""


PAUSE_REASONS = {
    'paused': '等待安静观察满15分钟（从最近逆势放量或初始暂缓起算）',
    'waiting_data': '等待新鲜、完整的行情',
    'waiting_history': '历史不足，继续等待',
    'waiting_range': '当前价越过原价超过¼R，等待回到允许恢复区域',
    'waiting_reclaim': '等待完整1m收回：挂回等待侧需收回原价，可能立即成交侧需收回¼R区域',
    'waiting_trend': '15m趋势与入场方向相反，暂不恢复可能立即成交的限价单',
    'waiting_trend_data': '等待新鲜、完整的已收盘15m趋势数据',
    'waiting_continuity': '等待连续有效观察满15分钟',
    'invalidated': '触及原硬SL，永久失效',
    'expired': '原有效期到期',
    'cancelled': '用户已取消',
    'replaced': '已被新计划替换',
    'ready': '行情恢复条件已满足，仍需通过重试时间和最终风控检查',
}


def data_error_reason(error):
    if isinstance(error, EntryDataError):
        return str(error)
    if isinstance(error, KeyError):
        return '行情字段缺失，等待完整采样'
    return f'行情读取或校验失败（{type(error).__name__}）'


def checked_rows(rows, width, observed_at, now, minimum=1):
    if not 0 <= now - observed_at <= MAX_AGE or len(rows) < minimum:
        raise EntryDataError('入场行情过期或不足')
    last = int(observed_at) // width * width
    if int(now) // width != int(observed_at) // width:
        raise EntryDataError('等待新周期行情')
    previous = None
    for row in rows:
        if len(row) < 6 or not all(isfinite(float(v)) for v in row[:6]):
            raise EntryDataError('入场行情数值无效')
        stamp, op, high, low, close, volume = row[:6]
        if (
            stamp % (width * 1000)
            or min(op, high, low, close) <= 0
            or volume < 0
            or low > min(op, close)
            or high < max(op, close)
            or low > high
            or (previous is not None and stamp - previous != width * 1000)
        ):
            raise EntryDataError('入场行情缺口或价格无效')
        previous = stamp
    if previous != last * 1000:
        raise EntryDataError('缺少当前行情')
    return rows


def analyze_volume(rows, observed_at, now, direction):
    """Actual base volume, never extrapolated; also retain recent closed shocks."""
    checked_rows(rows, 300, observed_at, now, 28)
    latest = None
    for i in range(len(rows) - 4, len(rows)):
        stamp, op, _, _, close, volume = rows[i][:6]
        baseline = sum(r[5] for r in rows[i - 24 : i]) / 24
        if baseline <= 0:
            raise EntryDataError('入场均量不可用')
        adverse = close < op if direction == 'long' else close > op
        at = observed_at if i == len(rows) - 1 else stamp / 1000 + 300
        if adverse and volume / baseline >= 4 and now - at < COOLDOWN:
            latest = {'at': at, 'ratio': volume / baseline}
    return latest


def analyze_recovery_trend(rows, observed_at, now):
    """Validate current public data, then reuse the advisory closed-candle EMA rule."""
    from market_monitor import analyze_trend

    checked_rows(rows, 900, observed_at, now, 51)
    result = analyze_trend(rows, '15m', datetime.fromtimestamp(observed_at, timezone.utc))
    return dict(
        direction=result['trend'],
        observed_at=observed_at,
        closed_at=result['closed_at'].timestamp(),
    )


def advance_pause(meta, rows, signal, observed_at, now, *, quote=None, trend=None):
    """Update only from observations already available; preserve the original deadline."""
    result = deepcopy(meta)
    if result.get('decision') in ('invalidated', 'expired'):
        return result
    if now >= result['deadline']:
        result['decision'] = 'expired'
        return result
    checked_rows(rows, 60, observed_at, now)
    start = result.get('checked_until', result['created'])
    if rows[0][0] / 1000 > int(start) // 60 * 60:
        raise EntryDataError('缺少等待期间价格历史，继续暂停')
    request = result['request']
    short = request['direction'] == 'short'
    entry, stop = request['entry_price'], request['stop_loss']
    bound = entry + (0.25 if short else -0.25) * abs(entry - stop)
    result.setdefault('crossed', False)
    for stamp_ms, op, high, low, close, *_ in rows:
        stamp = stamp_ms / 1000
        if stamp + 60 <= start:
            continue
        # Extremes in the creation minute can precede the request. Never invent
        # their ordering; hold an ambiguous plan instead of declaring a stop hit.
        stop_hit = high >= stop if short else low <= stop
        if stamp < result['created'] and stop_hit:
            result['history_uncertain'] = True
        elif stop_hit:
            result['decision'] = 'invalidated'
        result['low_water'] = min(result.get('low_water', low), low)
        result['high_water'] = max(result.get('high_water', high), high)
        if high > entry if short else low < entry:
            result['crossed'] = True
            result['crossed_at'] = max(result.get('crossed_at', 0), stamp)
        if high > bound if short else low < bound:
            result['range_crossed_at'] = max(result.get('range_crossed_at', 0), stamp)
    quote = rows[-1][4] if quote is None else quote
    if not isfinite(quote) or quote <= 0:
        raise EntryDataError('当前报价无效')
    if quote >= stop if short else quote <= stop:
        result['decision'] = 'invalidated'
    if quote > entry if short else quote < entry:
        result['crossed'] = True
        result['crossed_at'] = observed_at
    if quote > bound if short else quote < bound:
        result['range_crossed_at'] = observed_at
    result['checked_until'] = observed_at
    if result.get('decision') == 'invalidated':
        return result
    if signal:
        result['last_adverse'] = max(result.get('last_adverse', 0), signal['at'])
    if result.get('history_uncertain'):
        result['decision'] = 'waiting_history'
    elif now - result.get('last_adverse', result.get('quiet_after', now)) < COOLDOWN:
        result['decision'] = 'paused'
    elif quote >= entry if short else quote <= entry:
        # A marketable original limit needs bounded price/close and a trend veto.
        closed = [r for r in rows if r[0] / 1000 + 60 <= observed_at]
        if quote > bound if short else quote < bound:
            result['decision'] = 'waiting_range'
        elif (
            not closed
            or (closed[-1][4] > bound if short else closed[-1][4] < bound)
            or closed[-1][0] / 1000 + 60 <= result.get('range_crossed_at', 0)
        ):
            result['decision'] = 'waiting_reclaim'
        else:
            try:
                valid_trend = (
                    trend['direction'] in ('偏多', '偏空', '震荡')
                    and isfinite(trend['observed_at'])
                    and 0 <= now - trend['observed_at'] <= MAX_AGE
                    and trend['closed_at'] == int(now) // 900 * 900
                    and trend['closed_at'] <= trend['observed_at']
                )
            except (KeyError, TypeError, ValueError):
                valid_trend = False
            if not valid_trend:
                result['decision'] = 'waiting_trend_data'
            elif trend['direction'] == ('偏多' if short else '偏空'):
                result['decision'] = 'waiting_trend'
            else:
                result['decision'] = 'ready'
    elif result['crossed']:
        closed = [r for r in rows if r[0] / 1000 + 60 <= observed_at]
        reclaimed = bool(closed and closed[-1][0] / 1000 + 60 > result['crossed_at'])
        reclaimed = reclaimed and (
            max(closed[-1][4], quote) <= entry if short else min(closed[-1][4], quote) >= entry
        )
        result['decision'] = 'ready' if reclaimed else 'waiting_reclaim'
    else:
        result['decision'] = 'ready'
    return result
