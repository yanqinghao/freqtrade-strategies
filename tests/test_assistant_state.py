import json
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'user_data' / 'strategies'))
from assistant_state import read_state, update_state


class StateTests(unittest.TestCase):
    def test_concurrent_plan_writers_preserve_all_coins(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'plans.json'

            def save(coin):
                update_state(
                    path,
                    lambda state: state.setdefault('manual_open', {}).update(
                        {coin: {'stop_loss': 90}}
                    ),
                )

            with ThreadPoolExecutor(max_workers=3) as pool:
                list(pool.map(save, ('BTC', 'ETH', 'SOL')))
            self.assertEqual(set(read_state(path)['manual_open']), {'BTC', 'ETH', 'SOL'})

    def test_invalid_write_does_not_destroy_existing_plans(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'plans.json'
            update_state(path, lambda state: state.update(manual_open={'BTC': {'stop_loss': 90}}))
            before = read_state(path)
            with self.assertRaises(ValueError):
                update_state(path, lambda state: state.update(bad=float('nan')))
            self.assertEqual(read_state(path), before)
            path.write_text('incomplete JSON')
            with self.assertRaises(json.JSONDecodeError):
                update_state(path, lambda state: state.clear())
            self.assertEqual(path.read_text(), 'incomplete JSON')
