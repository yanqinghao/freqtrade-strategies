"""Independent notification-only watchdog. No exchange credentials or trading imports."""
import argparse
import json
import logging
import signal
import sys
import time
from copy import deepcopy
from math import isfinite
from pathlib import Path
from threading import Event
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'user_data' / 'strategies'))
from assistant_state import read_state, update_state
from notification_format import notice, local_time

logger = logging.getLogger(__name__)
PROBE_FAILURE_GRACE = 180
PROBES = {
    'spot': 'https://api.binance.com/api/v3/exchangeInfo?symbol=BTCUSDT',
    'futures': 'https://fapi.binance.com/fapi/v1/ping',
}


def _timestamp(value):
    try:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and isfinite(value)
            and value >= 0
        )
    except OverflowError:
        return False


def _initial_state(started_at):
    return {
        'started_at': started_at,
        'faults': {},
        'probe_failures': {},
        'probe_failed_since': {},
        'pending': False,
        'incident': {},
        'incident_since': None,
        'last_completed': None,
    }


def _transition(state, faults, now):
    result = deepcopy(state)
    if faults != result['faults']:
        if faults and result['incident_since'] is None:
            result['incident_since'] = now
        result['incident'].update(faults)
        result['faults'] = dict(faults)
        result['pending'] = True
    result['last_check'] = now
    return result


def _storage_fault(state, now, error):
    logger.warning('Watchdog storage degraded (%s)', type(error).__name__)
    faults = dict(state['faults'])
    faults['storage'] = '守护状态无法持久化，已使用内存继续监测；重启后可能重复通知'
    return _transition(state, faults, now)


def _heartbeat_fault(snapshot, now, started_at, last_completed=None):
    if now - started_at < 180:
        return None
    if not isinstance(snapshot, dict):
        return '交易健康快照缺失或无效'
    observed = snapshot.get('observed_at')
    if not _timestamp(observed) or observed > now + 5 or now - observed > 180:
        return '交易健康快照缺失、过期或时钟异常'
    state = snapshot.get('state')
    last = snapshot.get('last_process')
    recent_progress = _timestamp(last_completed) and 0 <= now - last_completed <= 180
    valid_last = _timestamp(last) and last <= now + 5 and last <= observed + 5
    if recent_progress and (
        (last is None and state in ('RUNNING', 'PAUSED', 'STOPPED'))
        or (state in ('RELOAD_CONFIG', 'STARTUP') and (last is None or valid_last))
    ):
        return None  # Brief startup may use previous real progress, never a new grace clock.
    if state == 'STOPPED':
        return '交易处理已停止'
    if state not in ('RUNNING', 'PAUSED'):
        return '机器人尚未进入交易处理状态'
    if not valid_last:
        return '交易循环尚未完成或时间数据无效'
    if now - last > 180:
        return '交易循环超过3分钟未完成'
    return None


class Watchdog:
    def __init__(self, state_file):
        self.state_file = Path(state_file)
        self._state = None

    def _save(self, state):
        def replace(document):
            document.clear()
            document.update(state)

        update_state(self.state_file, replace)
        self._state = state

    def check(self, snapshot, probes, now, send):
        load_error = None
        try:
            stored = self._state if self._state is not None else read_state(self.state_file)
            state = deepcopy(stored) if stored else _initial_state(now)
            state.setdefault('probe_failed_since', {})  # Existing watchdog state remains valid.
            if (
                not _timestamp(state.get('started_at'))
                or state['started_at'] > now + 5
                or not {'faults', 'probe_failures', 'incident', 'pending', 'incident_since'}
                <= state.keys()
                or not all(
                    isinstance(state.get(k), dict)
                    for k in ('faults', 'probe_failures', 'probe_failed_since', 'incident')
                )
                or not all(
                    isinstance(k, str) and isinstance(v, str)
                    for field in ('faults', 'incident')
                    for k, v in state.get(field, {}).items()
                )
                or not isinstance(state.get('pending'), bool)
                or any(
                    not isinstance(v, int) or isinstance(v, bool) or not 0 <= v <= 2
                    for v in state['probe_failures'].values()
                )
                or any(
                    not _timestamp(v) or v > now + 5 for v in state['probe_failed_since'].values()
                )
                or (
                    state.get('incident_since') is not None
                    and not _timestamp(state['incident_since'])
                )
                or (
                    state.get('last_completed') is not None
                    and not _timestamp(state['last_completed'])
                )
            ):
                raise ValueError('Invalid watchdog state')
        except Exception as error:
            load_error = error
            # Lost history must not grant another grace period or silence current faults.
            state = _initial_state(max(0, now - 180))
        faults = {}
        if isinstance(snapshot, dict):
            observed, last = snapshot.get('observed_at'), snapshot.get('last_process')
            if (
                _timestamp(observed)
                and 0 <= now - observed <= 180
                and _timestamp(last)
                and last <= now + 5
                and last <= observed + 5
                and snapshot.get('state') in ('RUNNING', 'PAUSED', 'STOPPED', 'RELOAD_CONFIG')
            ):
                state['last_completed'] = max(last, state.get('last_completed') or 0)
        heartbeat = _heartbeat_fault(
            snapshot, now, state['started_at'], state.get('last_completed')
        )
        if heartbeat:
            faults['heartbeat'] = heartbeat
        for name in PROBES:
            count = (
                0 if probes.get(name) is True else min(2, state['probe_failures'].get(name, 0) + 1)
            )
            state['probe_failures'][name] = count
            if count == 0:
                state['probe_failed_since'].pop(name, None)
                continue
            # Preserve already announced faults when upgrading from the count-only state.
            since = state['probe_failed_since'].setdefault(
                name, now - PROBE_FAILURE_GRACE if name in state['faults'] else now
            )
            if count >= 2 and now - since >= PROBE_FAILURE_GRACE:
                label = '现货元数据' if name == 'spot' else 'USDT合约'
                faults[name] = f'Binance {label}公共接口不可达'
        candidate = _transition(state, faults, now)
        if load_error is not None:
            candidate = _storage_fault(candidate, now, load_error)
        try:
            self._save(candidate)
        except Exception as error:
            # Notification-only degradation: keep deduplication in memory if disk is broken.
            candidate = _storage_fault(_transition(state, faults, now), now, error)
            # Compare against original faults, not a speculative storage-recovery transition.
            final_faults = candidate['faults']
            candidate = _transition(state, final_faults, now)
            self._state = candidate
        state = self._state
        if state['pending']:
            current_faults = state['faults']
            conditions = current_faults or state['incident']
            since = state['incident_since']
            history = ['首次异常：' + local_time(since)] if since is not None else []
            message = notice(
                '运行监控',
                '开单助手 · ' + ('异常提醒' if current_faults else '监控恢复'),
                summary=['检测到异常，请核查。' if current_faults else '此前异常已解除。'],
                sections=[
                    ('异常项目' if current_faults else '已恢复项目', list(conditions.values())),
                    ('事件记录', history),
                ],
                action='检查 /health 与运行日志；独立守护仅通知，不会重启、撤单或修改仓位。',
                at=now,
            )
            try:
                delivered = send(message) is True
            except Exception as error:
                logger.warning('Watchdog notification pending (%s)', type(error).__name__)
                delivered = False
            if delivered:
                acknowledged = deepcopy(state)
                acknowledged['pending'] = False
                if not current_faults:
                    acknowledged['incident'] = {}
                    acknowledged['incident_since'] = None
                try:
                    self._save(acknowledged)
                except Exception as error:
                    self._state = _storage_fault(acknowledged, now, error)
        return deepcopy(self._state)


def probe_public(url):
    started = time.monotonic()
    status, reason = None, 'invalid_payload'
    try:
        with urlopen(url, timeout=5) as response:
            status = response.status
            if status == 200:
                data = json.loads(response.read(65536))
                if url == PROBES['spot']:
                    healthy = any(
                        item.get('symbol') == 'BTCUSDT'
                        and item.get('baseAsset') == 'BTC'
                        and item.get('quoteAsset') == 'USDT'
                        and isinstance(item.get('filters'), list)
                        and bool(item['filters'])
                        for item in data.get('symbols', [])
                    )
                else:
                    healthy = data == {}
                if healthy:
                    return True
            else:
                reason = 'http_status'
    except Exception as error:
        reason = type(error).__name__
        code = getattr(error, 'code', None)
        status = code if isinstance(code, int) else status
    # Never log exception text or a request URL: transport errors may contain secrets.
    name = next((name for name, endpoint in PROBES.items() if endpoint == url), 'other')
    logger.warning(
        'Public probe failed: probe=%s reason=%s http_status=%s elapsed_ms=%d',
        name,
        reason,
        status,
        (time.monotonic() - started) * 1000,
    )
    return False


def send_telegram(config, message):
    payload = {'chat_id': config['chat_id'], 'text': message}
    if config.get('topic_id') is not None:
        payload['message_thread_id'] = config['topic_id']
    request = Request(
        'https://api.telegram.org/bot' + config['token'] + '/sendMessage',
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    try:
        with urlopen(request, timeout=8) as response:
            return response.status == 200 and json.loads(response.read(65536)).get('ok') is True
    except Exception as exc:
        # HTTP exceptions can contain the token-bearing request URL: log the type only.
        logger.warning('Telegram notification pending (%s)', type(exc).__name__)
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--telegram-config', required=True)
    parser.add_argument('--health-file', required=True)
    parser.add_argument('--state-file', required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    try:
        config = read_state(args.telegram_config)
        if (
            not isinstance(config.get('token'), str)
            or not config['token']
            or config.get('chat_id') is None
        ):
            raise ValueError('Invalid TG-only configuration')
    except Exception as exc:
        logger.error('Cannot read TG-only configuration (%s)', type(exc).__name__)
        return 2
    stopped = Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *args: stopped.set())
    watchdog = Watchdog(args.state_file)
    logger.info('Independent watchdog started; poll=30s grace=180s stale=180s probe_grace=180s')
    previous = None
    while not stopped.is_set():
        started = time.monotonic()
        try:
            try:
                snapshot = read_state(args.health_file)
            except Exception:
                snapshot = None
            probes = {name: probe_public(url) for name, url in PROBES.items()}
            state = watchdog.check(
                snapshot, probes, time.time(), lambda text: send_telegram(config, text)
            )
            status = (tuple(sorted(state['faults'])), state['pending'])
            if status != previous:
                logger.info(
                    'Watchdog status: faults=%s notification_pending=%s',
                    ','.join(status[0]) or 'none',
                    status[1],
                )
                previous = status
        except Exception as exc:
            logger.error('Watchdog check failed (%s)', type(exc).__name__)
        stopped.wait(max(0, 30 - (time.monotonic() - started)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
