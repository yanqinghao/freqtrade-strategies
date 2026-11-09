import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'user_data/strategies'))
from entry_requests import EntryRequests


class PauseStoreTests(unittest.TestCase):
    def test_pause_persists_separately_from_deleted_native_trade(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EntryRequests(Path(directory) / 'requests.sqlite')
            self.assertTrue(hasattr(store, 'put_pause'), 'durable pause intent missing')
            meta = dict(
                ref='a' * 24,
                phase='paused',
                decision='paused',
                deadline=1234,
                request=dict(pair='ETH/USDT:USDT'),
                last_adverse=900,
                order_id='old',
            )
            store.put_pause(meta)
            restarted = EntryRequests(store.path)
            self.assertEqual(restarted.get_pause(meta['ref']), meta)
            self.assertEqual(restarted.pauses(), [meta])
            meta['decision'] = 'cancelled'
            restarted.put_pause(meta)
            self.assertEqual(store.get_pause(meta['ref'])['decision'], 'cancelled')
            meta['decision'] = 'ready'
            with self.assertRaises(ValueError):
                restarted.put_pause(meta)

    def test_unknown_submission_keeps_a_distinct_attempt_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EntryRequests(Path(directory) / 'requests.sqlite')
            self.assertTrue(hasattr(store, 'put_pause'), 'durable pause intent missing')
            request = dict(pair='BTC/USDT:USDT', direction='long')
            store.reserve('a' * 24, request)
            store.set_status('a' * 24, 'submitted', dict(order_ids=['old']))
            store.reserve('b' * 24, request)
            store.set_status('b' * 24, 'unknown', {})
            store.put_pause(
                dict(
                    ref='a' * 24,
                    phase='resuming',
                    attempt_ref='b' * 24,
                    request=request,
                    decision='ready',
                )
            )
            self.assertEqual(store.unresolved()[0]['ref'], 'b' * 24)
            self.assertEqual(store.get_pause('a' * 24)['attempt_ref'], 'b' * 24)
