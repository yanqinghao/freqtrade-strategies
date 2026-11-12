"""A local rejection delays submission, never invents a new market shock."""
import json
import unittest
from copy import deepcopy
from datetime import timedelta
from unittest.mock import patch

import test_entry_pause_execution as execution
from test_volume_pause_policy import rows


class PauseRetryTests(unittest.TestCase):
    setUp = execution.PauseExecutionTests.setUp
    add_order = execution.PauseExecutionTests.add_order
    partial = execution.PauseExecutionTests.partial
    prepare = execution.PauseExecutionTests.prepare
    run_tick = execution.PauseExecutionTests.run_tick

    def rejected(self, short=False):
        self.prepare()
        self.Trade.session.delete(self.trade)
        self.Trade.commit()
        stamp = self.now.timestamp()
        self.retry_ref = 'b' * 24
        meta = self.strategy.entry_requests.get_pause(self.ref)
        meta.update(
            phase='resuming',
            decision='ready',
            ever_paused=True,
            quiet_after=stamp - 1000,
            attempt_ref=self.retry_ref,
            attempts=1,
        )
        if short:
            meta['request'].update(direction='short', stop_loss=110, exit_points=[90, 80, 70])
        self.strategy.entry_requests.put_pause(meta)
        self.strategy.entry_requests.reserve(self.retry_ref, meta['request'])
        self.strategy.entry_requests.set_status(self.retry_ref, 'rejected', {'message': '等待新周期行情'})
        self.refresh(short=short)
        with patch('entry_pause.time.time', return_value=stamp):
            self.controller.reconcile(meta)
        return self.strategy.entry_requests.get_pause(self.ref)

    def refresh(self, short=False, continuity=None):
        stamp = self.now.timestamp()
        price = 99 if short else 101
        self.snapshot.update(
            observed_at=stamp,
            minute=rows(stamp, 60, 30, price),
            volume=rows(stamp, 300, 28, price),
            bid=price,
            ask=price,
            continuous_since=stamp - 3600 if continuity is None else continuity,
        )

    def install_submit(self):
        def submit(request, ref):
            self.submissions.append((deepcopy(request), ref))
            self.strategy.entry_requests.reserve(ref, request)
            return self.strategy.entry_requests.set_status(ref, 'unknown', {})

        self.controller.submit = submit

    def test_rejected_attempt_preserves_quiet_start_and_waits_only_for_retry_boundary(self):
        for short in (False, True):
            with self.subTest(short=short):
                if short:
                    self.doCleanups()
                    self.setUp()
                stamp = self.now.timestamp()
                meta = self.rejected(short)
                self.assertEqual(meta['quiet_after'], stamp - 1000)
                self.assertEqual(meta['retry_after'], stamp + 60)
                self.assertEqual(meta['deadline'], self.meta['deadline'])
                self.install_submit()
                self.now += timedelta(seconds=59)
                self.refresh(short)
                self.run_tick()
                self.assertEqual(self.submissions, [])
                self.now += timedelta(seconds=1)
                self.refresh(short)
                self.run_tick()
                self.assertEqual(len(self.submissions), 1)
                self.assertEqual(self.submissions[0][0]['entry_price'], 100)
                self.assertEqual(self.submissions[0][0]['entry_deadline'], self.meta['deadline'])
                self.run_tick()
                self.assertEqual(len(self.submissions), 1, 'unknown result must never replay')

    def test_rejection_after_old_shock_does_not_move_its_timestamp(self):
        meta = self.rejected()
        meta['last_adverse'] = self.now.timestamp() - 901
        self.strategy.entry_requests.put_pause(meta)
        with patch('entry_pause.time.time', return_value=self.now.timestamp() + 120):
            self.controller.reconcile(meta)
        restored = self.strategy.entry_requests.get_pause(self.ref)
        self.assertEqual(restored['quiet_after'], self.now.timestamp() - 1000)
        self.assertEqual(restored['last_adverse'], self.now.timestamp() - 901)
        self.assertEqual(restored['retry_after'], self.now.timestamp() + 60)

    def test_new_shock_and_observation_gap_still_prevent_retry(self):
        self.rejected()
        self.install_submit()
        self.now += timedelta(seconds=60)
        self.refresh()
        stamp = self.now.timestamp()
        self.snapshot['volume'][-1][4:] = [100, 40]
        meta = self.run_tick()
        self.assertEqual(self.submissions, [])
        self.assertEqual(meta['last_adverse'], stamp)
        self.now += timedelta(seconds=901)
        self.refresh(continuity=self.now.timestamp() - 1)
        meta = self.run_tick()
        self.assertEqual(meta['decision'], 'waiting_continuity')
        self.assertEqual(self.submissions, [])

    def test_rejection_reason_is_durable_and_complete_check_clears_current_error(self):
        meta = self.rejected()
        self.assertIn('等待新周期行情', meta['last_rejection']['message'])
        meta['check_error'] = dict(reason='行情字段缺失', at=self.now.timestamp(), stage='恢复条件')
        self.strategy.entry_requests.put_pause(meta)
        self.controller.evaluate(meta, self.now.timestamp())
        restored = self.strategy.entry_requests.get_pause(self.ref)
        self.assertNotIn('check_error', restored)
        self.assertEqual(restored['last_rejection']['message'], '等待新周期行情')
        self.assertEqual(restored['last_check']['quote'], 101)
        self.assertEqual(restored['last_check']['quiet_until'], self.now.timestamp() - 100)

    def test_guard_reports_missing_field_and_redacts_external_exception(self):
        from entry_risk import EntryRiskError

        self.rejected()
        meta = self.strategy.entry_requests.get_pause(self.ref)
        meta.update(phase='resuming', decision='ready')
        self.strategy.entry_requests.put_pause(meta)
        self.exchange.fetch_l2_order_book = lambda *a: dict(bids=[[101, 1]], asks=[[101, 1]])
        del self.snapshot['minute']
        with patch('entry_pause.time.time', return_value=self.now.timestamp()):
            with self.assertRaisesRegex(EntryRiskError, '行情字段缺失'):
                self.controller.guard(self.retry_ref)
        failure = self.strategy.entry_requests.get_pause(self.ref)['check_error']
        self.assertEqual(failure['stage'], '恢复条件')

        def failed_book(*args):
            raise TimeoutError('PRIVATE_TOKEN=https://private.example')

        self.exchange.fetch_l2_order_book = failed_book
        with patch('entry_pause.time.time', return_value=self.now.timestamp()), self.assertLogs(
            'entry_pause', level='WARNING'
        ) as logs:
            with self.assertRaises(EntryRiskError) as failure:
                self.controller.guard(self.retry_ref)
        self.assertIn('TimeoutError', str(failure.exception))
        self.assertNotIn('PRIVATE_TOKEN', str(failure.exception))
        self.assertNotIn('PRIVATE_TOKEN', '\n'.join(logs.output))
        self.assertNotIn(
            'PRIVATE_TOKEN', json.dumps(self.strategy.entry_requests.get_pause(self.ref))
        )

    def test_request_explains_current_error_retry_and_both_observation_windows(self):
        from test_telegram_assistant import load_telegram

        meta = self.rejected()
        self.controller.evaluate(meta, self.now.timestamp())
        meta = self.strategy.entry_requests.get_pause(self.ref)
        meta['check_error'] = dict(at=self.now.timestamp(), stage='恢复条件', reason='缺少当前行情')
        text = load_telegram().Telegram._format_entry_request(
            dict(ref=self.ref, request=meta['request'], result={'pause_ref': self.ref}, pause=meta)
        )
        for word in ('缺少当前行情', '等待新周期行情', '重试', '安静', '连续', '北京时间'):
            self.assertIn(word, text)
