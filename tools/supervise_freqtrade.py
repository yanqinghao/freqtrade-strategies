"""Recover a stalled Freqtrade process once Binance connectivity returns."""
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)
HEALTH_FILE = Path('/freqtrade/user_data/health/assistant.json')
WATCHDOG_FILE = Path('/freqtrade/user_data/watchdog/watchdog.json')
RECOVERY_FILE = Path('/tmp/freqtrade-last-recovery.json')
HEARTBEAT_TIMEOUT = 180
PROBE_MAX_AGE = 45  # The watchdog normally publishes every 30 seconds.
RECOVERY_COOLDOWN = 600
STOP_TIMEOUT = 20
OPERATOR_STATES = ('STOPPED', 'PAUSED')


def read_snapshot(path):
    try:
        snapshot = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        return None
    return snapshot if isinstance(snapshot, dict) else None


def timestamp(value, latest):
    return type(value) in (int, float) and 0 <= value <= latest


class RecoveryGate:
    def __init__(self, *, after_recovery=False):
        self.minimum_uptime = RECOVERY_COOLDOWN if after_recovery else HEARTBEAT_TIMEOUT
        self.first_probe = None
        self.last_completed = 0
        self.held_since = None

    def observe_health(self, snapshot, now):
        snapshot = snapshot or {}
        state = snapshot.get('state')
        observed = snapshot.get('observed_at')
        if state in OPERATOR_STATES:
            held_at = observed if timestamp(observed, now + 5) else now
            self.held_since = max(self.held_since or 0, held_at)
        if not timestamp(observed, now + 5):
            return not snapshot
        if state == 'RUNNING' and self.held_since is not None and observed > self.held_since:
            self.held_since = None
        last = snapshot.get('last_process')
        if timestamp(last, min(observed, now) + 5):
            self.last_completed = max(self.last_completed, last)
        return state in ('RUNNING', 'STARTUP', 'RELOAD_CONFIG')

    def ready(self, health, watchdog, *, now, uptime, restarted_at=None):
        watchdog = watchdog or {}
        check = watchdog.get('last_check')
        faults = watchdog.get('faults')
        eligible_state = self.observe_health(health, now)
        completed = watchdog.get('last_completed')
        if timestamp(completed, now + 5):
            self.last_completed = max(self.last_completed, completed)
        if (
            uptime < self.minimum_uptime
            or now - self.last_completed <= HEARTBEAT_TIMEOUT
            or health is None
            or not eligible_state
            or self.held_since is not None
            or (
                restarted_at is not None
                and (self.last_completed < restarted_at or now - restarted_at < RECOVERY_COOLDOWN)
            )
            or not isinstance(faults, dict)
            or faults.keys() != {'heartbeat'}
            or watchdog.get('probe_failures') != {'spot': 0, 'futures': 0}
            or not timestamp(check, now)
            or now - check > PROBE_MAX_AGE
        ):
            self.first_probe = None
            return False
        # A reread of the same state is still just one probe. A missed poll breaks the streak.
        if self.first_probe is None or not 0 <= check - self.first_probe <= PROBE_MAX_AGE:
            self.first_probe = check
            return False
        return check > self.first_probe


def should_restart(gate, uptime, restarted_at):
    watchdog = read_snapshot(WATCHDOG_FILE)
    health = read_snapshot(HEALTH_FILE)
    return gate.ready(health, watchdog, now=time.time(), uptime=uptime, restarted_at=restarted_at)


def previous_recovery():
    snapshot = read_snapshot(RECOVERY_FILE)
    if snapshot == {}:
        return None
    now = time.time()
    last = (snapshot or {}).get('at')
    if timestamp(last, now):
        return last
    logger.warning('Invalid Freqtrade recovery record, waiting for new trading progress')
    return now


def start_replacement(command):
    mode = (read_snapshot(HEALTH_FILE) or {}).get('state')
    if mode in OPERATOR_STATES:
        return subprocess.Popen(
            command, env={**os.environ, 'FREQTRADE__INITIAL_STATE': mode.lower()}
        )
    return subprocess.Popen(command)


def stop_child(child):
    child.terminate()
    try:
        child.wait(timeout=STOP_TIMEOUT)
    except subprocess.TimeoutExpired:
        logger.warning('Freqtrade did not exit after SIGTERM, sending SIGKILL')
        child.kill()
        child.wait()


def run(args, stopped):
    command = ['freqtrade', *args]
    gate = RecoveryGate()
    started = time.monotonic()
    restarted_at = previous_recovery()
    child = subprocess.Popen(command) if restarted_at is None else start_replacement(command)
    try:
        while not stopped():
            code = child.poll()
            if code is not None:
                child = None
                return code if code >= 0 else 128 - code
            if stopped():
                break
            uptime = time.monotonic() - started
            if should_restart(gate, uptime, restarted_at):
                if stopped():
                    break
                attempt = time.time()
                try:
                    RECOVERY_FILE.write_text(json.dumps({'at': attempt}))
                except OSError as exc:
                    logger.error('Cannot record Freqtrade recovery (%s)', type(exc).__name__)
                else:
                    restarted_at = attempt
                    logger.warning(
                        'Binance probes healthy but trading stalled, restarting Freqtrade'
                    )
                    stop_child(child)
                    child = None
                    if stopped():
                        break
                    gate = RecoveryGate(after_recovery=True)
                    started = time.monotonic()
                    child = start_replacement(command)
            time.sleep(5)
        return 0
    finally:
        if child is not None:
            stop_child(child)


def main(args):
    if args[:2] != ['--supervise', 'trade']:
        return os.execvp('freqtrade', ['freqtrade', *args])
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop)
    return run(args[1:], lambda: stopped)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
