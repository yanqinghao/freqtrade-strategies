import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'user_data' / 'strategies'))
sys.path.insert(0, str(ROOT / 'tools'))


class WatchdogTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(
            importlib.util.find_spec('health_watchdog'), 'independent watchdog missing'
        )
        import health_watchdog

        self.module = health_watchdog
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'watchdog.json'
        self.watch = self.module.Watchdog(self.path)
        self.messages = []
        self.call(1000.0)

    def healthy(self, now, state='RUNNING', last=None):
        return {
            'observed_at': now,
            'last_process': now - 1 if last is None else last,
            'state': state,
        }

    def send(self, message):
        self.messages.append(message)
        return True

    def call(self, now, snapshot='healthy', probes=None, send=None):
        if snapshot == 'healthy':
            snapshot = self.healthy(now)
        return self.watch.check(
            snapshot, probes or {'spot': True, 'futures': True}, now, send or self.send
        )

    def test_stale_native_loop_alerts_despite_fresh_publisher_once_and_recovers(self):
        self.call(1301.0, self.healthy(1301.0, last=1000.0))
        self.assertEqual(len(self.messages), 1)
        self.assertIn('交易循环', self.messages[0])
        self.call(1331.0, self.healthy(1331.0, last=1000.0))
        self.assertEqual(len(self.messages), 1)
        self.call(1361.0)
        self.assertEqual(len(self.messages), 2)
        self.assertIn('恢复', self.messages[-1])

    def test_startup_grace_and_restart_cannot_hide_missing_heartbeat(self):
        self.watch = self.module.Watchdog(Path(self.temp.name) / 'new.json')
        self.call(1000.0, None)
        self.call(1150.0, None)
        self.assertEqual(self.messages, [])
        self.watch = self.module.Watchdog(Path(self.temp.name) / 'new.json')
        self.call(1181.0, None)
        self.assertEqual(len(self.messages), 1)

    def test_stopped_notifies_once_and_paused_with_progress_is_healthy(self):
        self.call(1300.0, self.healthy(1300.0, state='STOPPED', last=1000.0))
        self.assertEqual(len(self.messages), 1)
        self.assertIn('停止', self.messages[0])
        self.call(1330.0, self.healthy(1330.0, state='STOPPED', last=1000.0))
        self.assertEqual(len(self.messages), 1)
        self.call(1360.0, self.healthy(1360.0, state='PAUSED'))
        self.assertIn('恢复', self.messages[-1])
        self.call(1390.0, self.healthy(1390.0, state='PAUSED'))
        self.assertEqual(len(self.messages), 2)

    def test_invalid_and_future_health_fields_do_not_count_as_healthy(self):
        for snapshot in (
            {},
            {'state': 'RUNNING', 'observed_at': float('nan'), 'last_process': 1300.0},
            {'state': 'RUNNING', 'observed_at': 2300.0, 'last_process': 2300.0},
            {'state': 'RUNNING', 'observed_at': 1300.0, 'last_process': None},
            {'state': 'RUNNING', 'observed_at': 1000.0, 'last_process': 1299.0},
        ):
            with self.subTest(snapshot=snapshot):
                result = self.call(1300.0, snapshot)
                self.assertTrue(result['faults'])

    def test_exchange_failure_requires_two_observations_then_recovers_once(self):
        self.call(1010.0, probes={'spot': False, 'futures': True})
        self.assertEqual(self.messages, [])
        self.call(1040.0, probes={'spot': False, 'futures': True})
        self.assertEqual(len(self.messages), 1)
        self.assertIn('Binance', self.messages[-1])
        self.call(1070.0, probes={'spot': False, 'futures': True})
        self.assertEqual(len(self.messages), 1)
        self.call(1100.0)
        self.assertEqual(len(self.messages), 2)
        self.assertIn('恢复', self.messages[-1])

    def test_send_failure_is_persisted_and_retry_survives_watchdog_restart(self):
        def fail(message):
            raise TimeoutError('TOKEN_SHOULD_NOT_LEAK')

        result = self.call(1300.0, self.healthy(1300.0, last=1000.0), send=fail)
        self.assertTrue(result['pending'])
        self.watch = self.module.Watchdog(self.path)
        self.call(1330.0, self.healthy(1330.0, last=1000.0))
        self.assertEqual(len(self.messages), 1)
        self.assertFalse(json.loads(self.path.read_text())['pending'])
        self.call(1360.0, self.healthy(1360.0, last=1000.0))
        self.assertEqual(len(self.messages), 1)

    def test_recovered_before_delivery_coalesces_to_one_summary(self):
        self.call(1300.0, self.healthy(1300.0, last=1000.0), send=lambda message: False)
        self.watch = self.module.Watchdog(self.path)
        self.call(1330.0)
        self.assertEqual(len(self.messages), 1)
        self.assertIn('恢复', self.messages[0])
        self.assertIn('此前', self.messages[0])
        self.call(1360.0)
        self.assertEqual(len(self.messages), 1)

    def test_healthy_start_and_frequent_checks_do_not_spam(self):
        for now in (1030.0, 1060.0, 1090.0, 1300.0, 2000.0):
            self.call(now)
        self.assertEqual(self.messages, [])
        self.assertEqual(json.loads(self.path.read_text())['last_check'], 2000.0)

    def test_corrupt_state_still_alerts_without_resetting_grace_and_recovers(self):
        self.path.write_text('{broken json')
        self.watch = self.module.Watchdog(self.path)
        self.call(1300.0, None)
        self.assertEqual(len(self.messages), 1)
        self.assertIn('持久化', self.messages[0])
        self.call(1330.0, None)
        self.assertEqual(len(self.messages), 1, 'memory fallback must deduplicate')
        self.assertIn('heartbeat', self.watch._state['faults'])
        self.path.write_text('{}')
        self.call(1360.0)
        self.assertEqual(len(self.messages), 2)
        self.assertIn('恢复', self.messages[-1])
        self.assertFalse(json.loads(self.path.read_text())['faults'])

    def test_invalid_persisted_schema_still_reports_storage_degradation(self):
        state = json.loads(self.path.read_text())
        del state['incident_since']
        self.path.write_text(json.dumps(state))
        self.watch = self.module.Watchdog(self.path)
        self.call(1300.0, None)
        self.assertTrue(self.messages)
        self.assertIn('持久化', self.messages[-1])

    def test_bot_restart_uses_last_completed_loop_deadline_not_a_new_grace(self):
        self.call(1300.0)
        for now, mode in ((1330.0, 'STOPPED'), (1360.0, 'RELOAD_CONFIG'), (1450.0, 'RUNNING')):
            self.watch = self.module.Watchdog(self.path)
            self.call(now, {'observed_at': now, 'last_process': None, 'state': mode})
            self.assertEqual(self.messages, [], 'short bot startup must not raise a false alarm')
        self.call(1481.0, {'observed_at': 1481.0, 'last_process': None, 'state': 'RUNNING'})
        self.assertEqual(
            len(self.messages), 1, 'restart cannot extend the last completed loop deadline'
        )

    def test_unwritable_state_uses_memory_alerts_then_flushes_on_recovery(self):
        with patch.object(self.module, 'update_state', side_effect=PermissionError('SECRET')):
            self.call(1300.0, self.healthy(1300.0, last=1000.0))
            self.assertEqual(len(self.messages), 1)
            self.assertIn('持久化', self.messages[0])
            self.call(1330.0, self.healthy(1330.0, last=1000.0))
            self.assertEqual(len(self.messages), 1)
        self.call(1360.0)
        self.assertEqual(len(self.messages), 2)
        state = json.loads(self.path.read_text())
        self.assertFalse(state['pending'])
        self.assertEqual(state['faults'], {})

    def test_storage_and_telegram_failure_keeps_pending_in_memory(self):
        with patch.object(self.module, 'update_state', side_effect=OSError('disk full')):
            state = self.call(1300.0, None, send=lambda text: False)
            self.assertTrue(state['pending'])
            self.call(1330.0, None)
            self.assertEqual(len(self.messages), 1)
            self.assertFalse(self.watch._state['pending'])

    def test_metadata_failure_is_detected_when_ping_is_healthy(self):
        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self, *args):
                return b'{}'

        def opened(url, **kwargs):
            if 'exchangeInfo' in url:
                raise OSError('metadata blocked')
            return Response()

        with patch.object(self.module, 'urlopen', side_effect=opened):
            probes = {
                name: self.module.probe_public(url) for name, url in self.module.PROBES.items()
            }
        self.assertFalse(probes['spot'])
        self.assertTrue(probes['futures'])

    def test_spot_probe_validates_minimal_symbol_metadata(self):
        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self, *args):
                return json.dumps(
                    {
                        'symbols': [
                            {
                                'symbol': 'BTCUSDT',
                                'baseAsset': 'BTC',
                                'quoteAsset': 'USDT',
                                'filters': [{'filterType': 'LOT_SIZE'}],
                            }
                        ]
                    }
                ).encode()

        with patch.object(self.module, 'urlopen', return_value=Response()):
            self.assertTrue(self.module.probe_public(self.module.PROBES['spot']))

    def test_secret_bearing_transport_error_is_sanitized(self):
        config = {'token': '123:FAKE_SECRET_TOKEN', 'chat_id': '10', 'topic_id': 5}

        def fail(request, **kwargs):
            raise RuntimeError('https://api.telegram.org/bot123:FAKE_SECRET_TOKEN/sendMessage')

        with patch.object(self.module, 'urlopen', side_effect=fail):
            with self.assertLogs('health_watchdog', level='WARNING') as logs:
                self.assertFalse(self.module.send_telegram(config, 'alert'))
        self.assertNotIn('FAKE_SECRET_TOKEN', ' '.join(logs.output))

    def test_telegram_transport_posts_only_intended_payload_and_checks_ack(self):
        requests = []

        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self, *args):
                return b'{"ok":true}'

        def opened(request, **kwargs):
            requests.append(request)
            return Response()

        with patch.object(self.module, 'urlopen', side_effect=opened):
            self.assertTrue(
                self.module.send_telegram(
                    {'token': '123:TEST', 'chat_id': 10, 'topic_id': 5}, 'hello'
                )
            )
        self.assertEqual(requests[0].get_method(), 'POST')
        self.assertEqual(
            json.loads(requests[0].data), {'chat_id': 10, 'message_thread_id': 5, 'text': 'hello'}
        )

    def test_public_probe_requires_valid_ping_json(self):
        class Response:
            status = 200
            body = b'{}'

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self, *args):
                return self.body

        with patch.object(self.module, 'urlopen', return_value=Response()):
            self.assertTrue(self.module.probe_public('https://fapi.binance.com/fapi/v1/ping'))
        response = Response()
        response.body = b'<html>blocked</html>'
        with patch.object(self.module, 'urlopen', return_value=response):
            self.assertFalse(self.module.probe_public('https://fapi.binance.com/fapi/v1/ping'))


class MinimalWatchdogConfigTests(unittest.TestCase):
    def test_preparation_copies_only_tg_credentials_with_private_permissions(self):
        self.assertIsNotNone(
            importlib.util.find_spec('prepare_watchdog'), 'TG-only config preparation missing'
        )
        from prepare_watchdog import prepare_watchdog

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'config.json'
            target = Path(directory) / 'watchdog.json'
            source.write_text(
                json.dumps(
                    {
                        'exchange': {'key': 'EXCHANGE_SECRET', 'secret': 'NEVER_COPY'},
                        'telegram': {
                            'enabled': True,
                            'token': '123:TEST',
                            'chat_id': '10',
                            'topic_id': 5,
                        },
                    }
                )
            )
            prepare_watchdog(source, target)
            self.assertEqual(
                json.loads(target.read_text()),
                {'token': '123:TEST', 'chat_id': '10', 'topic_id': 5},
            )
            self.assertEqual(os.stat(target).st_mode & 0o777, 0o600)
            self.assertNotIn('EXCHANGE_SECRET', target.read_text())

    def test_compose_watchdog_mounts_no_trading_credentials_or_docker_socket(self):
        import yaml

        compose = yaml.safe_load((ROOT / 'docker-compose.yml').read_text())
        self.assertIn('watchdog', compose['services'], 'independent container missing')
        watchdog = compose['services']['watchdog']
        volumes = watchdog['volumes']
        self.assertTrue(all(isinstance(v, dict) for v in volumes))
        for mount in volumes:
            self.assertNotIn('docker.sock', mount['source'])
            self.assertNotEqual(mount['source'], './deploy')
            self.assertNotEqual(mount['source'], './user_data')
            self.assertNotEqual(mount['source'], './deploy/config.json')
            if mount['target'] != '/state':
                self.assertTrue(mount.get('read_only'))
