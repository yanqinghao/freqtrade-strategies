"""Shared native Telegram notification layout. Formatting only; no trading state."""
from datetime import datetime, timedelta, timezone

BEIJING = timezone(timedelta(hours=8))


def local_time(value):
    if not isinstance(value, datetime):
        value = datetime.fromtimestamp(value, timezone.utc)
    return value.astimezone(BEIJING).strftime('%m-%d %H:%M:%S') + ' 北京时间'


def with_time(message, at):
    if any(line.startswith('时间：') and line.endswith('北京时间') for line in message.splitlines()):
        return message
    return message + '\n\n时间：' + local_time(at)


def notice(title, subject='', *, summary=(), sections=(), action=None, at=None):
    parts = [f'【{title}】' + subject]
    if summary:
        parts.append('\n'.join(str(line) for line in summary if line))
    for heading, lines in sections:
        body = '\n'.join(str(line) for line in lines if line)
        if body:
            parts.append((heading + '\n' if heading else '') + body)
    if action:
        parts.append('下一步：' + action)
    message = '\n\n'.join(parts)
    return with_time(message, at) if at is not None else message
