"""Shared validation and closed-candle data for optional manual exit rules."""

from math import isfinite

import pandas as pd

TIMEFRAMES = {'15m': 900, '1h': 3600, '4h': 14400, '1d': 86400}
RULE_GROUPS = {
    'soft_sl': ('soft_sl', 'soft_tf'),
    'extend_trigger': ('extend_trigger', 'extend_tp3', 'extend_tf'),
}
RULE_FIELDS = tuple(field for group in RULE_GROUPS.values() for field in group)


def validate_rules(plan):
    for group in RULE_GROUPS.values():
        values = [plan.get(field) for field in group]
        if all(value is None for value in values):
            continue
        if any(value is None for value in values):
            raise ValueError('收盘规则参数须成组填写：' + '、'.join(group))
        if values[-1] not in TIMEFRAMES:
            raise ValueError('收盘周期仅支持 15m、1h、4h、1d')
        if any(
            isinstance(value, bool) or not isfinite(float(value)) or float(value) <= 0
            for value in values[:-1]
        ):
            raise ValueError('收盘规则价格必须是有限正数')
    if plan.get('extend_trigger') is not None:
        targets = sorted(
            set(plan.get('exit_points') or []), reverse=plan.get('direction') == 'short'
        )
        if len(targets) != 3:
            raise ValueError('延伸止盈需要三个不同的 TP')
        second, third = targets[1:]
        trigger, extended = plan['extend_trigger'], plan['extend_tp3']
        valid = (
            second > trigger > third > extended
            if plan.get('direction') == 'short'
            else second < trigger < third < extended
        )
        if not valid:
            raise ValueError('延伸触发价须在 TP2 与 TP3 之间，延伸 TP3 须比原 TP3 更远')


def closed_candle(frame, timeframe, now, after):
    """Latest fresh closed candle, strictly newer than activation; no forming-bar fallback."""
    if frame is None or frame.empty:
        return None
    width = TIMEFRAMES[timeframe]
    dates = pd.to_datetime(frame['date'], utc=True, errors='coerce')
    if dates.isna().any() or dates.duplicated().any():
        return None
    ends = dates + pd.Timedelta(seconds=width)
    eligible = frame.loc[ends <= now].copy()
    if eligible.empty:
        return None
    eligible['closed_at'] = ends.loc[eligible.index]
    row = eligible.sort_values('closed_at').iloc[-1]
    stamp = row['closed_at'].timestamp()
    if stamp <= after or not 0 <= now.timestamp() - stamp < width:
        return None
    prices = [float(row[key]) for key in ('open', 'high', 'low', 'close')]
    if not all(isfinite(price) and price > 0 for price in prices):
        return None
    opened, high, low, close = prices
    if not low <= min(opened, close) <= max(opened, close) <= high:
        return None
    return {'close': close, 'high': high, 'low': low, 'closed_at': stamp}


def describe_rules(plan, state=None):
    state = state or {}
    lines = []
    short = plan.get('direction') == 'short'
    if plan.get('soft_sl') is not None:
        lines.append(
            f"软SL：{plan['soft_tf']} 收盘 {'≥' if short else '≤'} " f"{plan['soft_sl']:g} 全平（硬SL仍有效）。"
        )
    extension = state.get('extension')
    if extension:
        lines.append(
            f"延伸已生效：TP3 {extension['target']:g}，" f"保护至少推进至 TP2 {extension['stop']:g}（触价退出，只收紧）。"
        )
    elif plan.get('extend_trigger') is not None:
        lines.append(
            f"待延伸：TP2成交后，{plan['extend_tf']} 收盘 {'≤' if short else '≥'} "
            f"{plan['extend_trigger']:g} → TP3 {plan['extend_tp3']:g}，保护推进至 TP2。"
        )
    if (state.get('full_exit') or {}).get('exit_tag') == 'manual_soft_sl':
        lines.append('软SL已触发，等待确认全平。')
    return '\n'.join(lines)
