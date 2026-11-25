import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from threading import Event
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))

from supervise_freqtrade import RecoveryGate, main, run, start_replacement, stop_child


def watchdog(at, *, spot=0, futures=0):
    return {
        'last_check': at,
        'faults': {'heartbeat': 'stalled'},
        'probe_failures': {'spot': spot, 'futures': futures},
    }


def health(at, last=700, state='RUNNING'):
    return {'observed_at': at, 'last_process': last, 'state': state}


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.gate = RecoveryGate()

    def check(self, at, *, sample=None, snapshot=None, uptime=300):
        return self.gate.ready(
            health(at) if snapshot is None else snapshot,
            watchdog(at) if sample is None else sample,
            now=at,
            uptime=uptime,
        )

    def test_requires_two_distinct_successful_polls_after_a_failure(self):
        self.assertFalse(self.check(1000, sample=watchdog(1000, futures=1)))
        self.assertFalse(self.check(1030))
        self.assertFalse(self.check(1035, sample=watchdog(1030)))
        self.assertFalse(self.check(1060, sample=watchdog(1060, spot=1)))
        self.assertFalse(self.check(1090))
        self.assertTrue(self.check(1120))

    def test_ignores_stale_or_unrelated_watchdog_results(self):
        self.assertFalse(self.check(1000, sample=watchdog(900)))
        unrelated = watchdog(1030)
        unrelated['faults']['storage'] = 'unavailable'
        self.assertFalse(self.check(1030, sample=unrelated))
        self.assertFalse(self.check(1060))
        self.assertFalse(self.check(1150))
        self.assertTrue(self.check(1180))

    def test_live_progress_and_manual_stop_or_pause_prevent_restarts(self):
        for snapshot in (
            health(1030, last=1029),
            health(1030, state='STOPPED'),
            health(1030, state='PAUSED'),
        ):
            with self.subTest(snapshot=snapshot):
                self.gate = RecoveryGate()
                self.assertFalse(self.check(1000))
                self.assertFalse(self.check(1030, snapshot=snapshot))
                self.assertFalse(self.check(1060))

    def test_missing_startup_heartbeat_can_recover_after_grace(self):
        self.assertFalse(self.check(1000, snapshot={}, uptime=150))
        self.assertFalse(self.check(1030, snapshot={}, uptime=180))
        self.assertTrue(self.check(1060, snapshot={}, uptime=210))

    def test_missing_live_snapshot_cannot_erase_recent_trading_progress(self):
        self.assertFalse(self.check(1000, snapshot=health(1000, last=999)))
        self.assertFalse(self.check(1030, snapshot={}))
        self.assertFalse(self.check(1060, snapshot=health(1060, last=None)))
        self.assertFalse(self.check(1180, snapshot={}))
        self.assertTrue(self.check(1210, snapshot={}))

    def test_manual_pause_survives_missing_and_older_snapshots_until_resume(self):
        self.assertFalse(self.check(1000, snapshot=health(1000, state='PAUSED'), uptime=179))
        self.assertFalse(self.check(1030, snapshot={}))
        self.assertFalse(self.check(1060, snapshot=health(990)))
        self.assertFalse(self.check(1090, snapshot={}))
        self.assertFalse(self.check(1120, snapshot=health(1120)))
        self.assertTrue(self.check(1150))

    def test_another_recovery_requires_trading_progress_since_last_restart(self):
        def check(at, snapshot):
            return self.gate.ready(snapshot, watchdog(at), now=at, uptime=900, restarted_at=1000)

        self.assertFalse(check(1030, health(1030, last=1029)))
        self.assertFalse(check(1590, {}))
        self.assertFalse(check(1620, {}))
        self.assertTrue(check(1650, {}))
        self.gate = RecoveryGate()
        self.assertFalse(check(1680, health(1680, last=990)))
        self.assertFalse(check(1710, {}))

    def test_one_off_compose_command_bypasses_recovery(self):
        with patch('supervise_freqtrade.os.execvp') as execute:
            main(['trade', '--dry-run'])
        execute.assert_called_once_with('freqtrade', ['freqtrade', 'trade', '--dry-run'])

    def test_replacement_preserves_stop_published_during_shutdown(self):
        with (
            patch('supervise_freqtrade.read_snapshot', return_value=health(1000, state='STOPPED')),
            patch('supervise_freqtrade.subprocess.Popen') as spawn,
        ):
            start_replacement(['freqtrade', 'trade'])
        self.assertEqual(spawn.call_args.kwargs['env']['FREQTRADE__INITIAL_STATE'], 'stopped')

    def test_container_restarts_after_recovery_preserve_manual_pause(self):
        with (
            patch('supervise_freqtrade.previous_recovery', return_value=700),
            patch('supervise_freqtrade.read_snapshot', return_value=health(1000, state='PAUSED')),
            patch('supervise_freqtrade.subprocess.Popen') as spawn,
        ):
            spawn.return_value.poll.return_value = 2
            self.assertEqual(run(['trade'], lambda: False), 2)
        self.assertEqual(spawn.call_args.kwargs['env']['FREQTRADE__INITIAL_STATE'], 'paused')

    def test_replacement_uses_monotonic_uptime_for_cooldown(self):
        self.gate = RecoveryGate(after_recovery=True)
        self.assertFalse(self.check(1800, snapshot=health(1800, last=1100), uptime=300))
        self.assertFalse(self.check(1830, snapshot=health(1830, last=1100), uptime=330))
        self.assertFalse(self.check(2100, snapshot=health(2100, last=1100), uptime=600))
        self.assertTrue(self.check(2130, snapshot=health(2130, last=1100), uptime=630))

    def test_graceful_timeout_kills_and_reaps_child(self):
        child = Mock()
        child.wait.side_effect = [subprocess.TimeoutExpired('freqtrade', 20), -signal.SIGKILL]
        stop_child(child)
        self.assertEqual(
            [call[0] for call in child.mock_calls], ['terminate', 'wait', 'kill', 'wait']
        )

    def test_restarts_once_then_observes_cooldown_and_propagates_exit(self):
        first = Mock()
        first.poll.return_value = None
        second = Mock()
        second.poll.side_effect = [None, 2]
        with (
            tempfile.TemporaryDirectory() as directory,
            patch('supervise_freqtrade.subprocess.Popen', side_effect=[first, second]) as spawn,
            patch('supervise_freqtrade.stop_child') as stop,
            patch('supervise_freqtrade.should_restart', side_effect=[True, False, False]) as check,
            patch('supervise_freqtrade.time.monotonic', side_effect=[0, 300, 300, 301]),
            patch('supervise_freqtrade.time.time', return_value=1000),
            patch('supervise_freqtrade.time.sleep'),
        ):
            marker = Path(directory) / 'last-recovery.json'
            with patch('supervise_freqtrade.RECOVERY_FILE', marker):
                self.assertEqual(run(['trade'], lambda: False), 2)
                self.assertTrue(marker.is_file())
                # A new supervisor reads the attempt left by the previous one.
                second.poll.side_effect = [None, 2]
                spawn.side_effect = [second]
                with patch('supervise_freqtrade.time.monotonic', side_effect=[0, 300]):
                    self.assertEqual(run(['trade'], lambda: False), 2)
                self.assertEqual(check.call_args.args[-1], 1000)
        self.assertEqual(spawn.call_count, 3)
        self.assertEqual(spawn.call_args.args[0], ['freqtrade', 'trade'])
        stop.assert_called_once_with(first)
        self.assertEqual(check.call_count, 3)

    def test_shutdown_signal_stops_child_without_restarting(self):
        stopped = Event()
        child = Mock()
        child.poll.return_value = None

        def stop_during_poll():
            stopped.set()
            return None

        child.poll.side_effect = stop_during_poll
        with (
            patch('supervise_freqtrade.subprocess.Popen', return_value=child) as spawn,
            patch('supervise_freqtrade.stop_child') as stop,
        ):
            self.assertEqual(run(['trade'], stopped.is_set), 0)
        spawn.assert_called_once()
        stop.assert_called_once_with(child)
