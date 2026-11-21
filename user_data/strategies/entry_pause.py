"""Durable original-limit withdrawal and recovery, on the trading thread only."""
import logging
import time
from math import isfinite
from copy import deepcopy
from datetime import datetime, timezone

from freqtrade.enums import State
from freqtrade.persistence import Trade, Order
from entry_lifecycle import _receipt, TERMINAL, install_entry_receipt_guard, entry_ttl
from candle_exits import RULE_FIELDS
from entry_targets import UNIT_FIELDS
from entry_risk import EntryRiskError
from entry_volume import (
    analyze_volume,
    advance_pause,
    EntryDataError,
    data_error_reason,
    PAUSE_REASONS,
)
from notification_format import notice

logger = logging.getLogger(__name__)
ENDED = ('invalidated', 'expired', 'cancelled', 'replaced')


class EntryPause:
    def __init__(self, strategy, bot, submit, monitor=None):
        self.strategy, self.bot, self.submit = strategy, bot, submit
        self.store = strategy.entry_requests
        self.monitor = monitor
        self.snapshots = {}
        self._adopted = False

    def find(self, ref):
        return next((m for m in self.store.pauses() if m.get('attempt_ref') == ref), None)

    def notice(self, meta, message):
        self.strategy._exit_notice(
            notice(
                '入场计划',
                meta['request']['pair'],
                summary=[message],
                sections=[('计划编号', [meta['ref']])],
                action=f"查询 /request {meta['ref']}；取消后续恢复 /request {meta['ref']} cancel。",
            ),
            key='entry-pause:' + meta['ref'],
        )

    def cancel(self, ref, reason='cancelled'):
        meta = self.store.get_pause(ref)
        if not meta:
            raise ValueError('未找到可撤的暂停计划')
        if meta.get('decision') not in ENDED:
            meta['decision'] = reason
            self.store.put_pause(meta)
        return meta

    def cancel_trade(self, trade_id, reason='cancelled'):
        for meta in self.store.pauses():
            if meta.get('trade_id') == trade_id and meta['phase'] not in ('filled', 'done'):
                self.cancel(meta['ref'], reason)

    def _check_failed(self, meta, stage, error, now):
        # Reload: a valid shock can have been saved before incomplete price data failed.
        current = self.store.get_pause(meta['ref'])
        reason = data_error_reason(error)
        previous = current.get('check_error', {})
        current['check_error'] = dict(at=now, stage=stage, reason=reason)
        if current.get('decision') in ('normal', 'ready', 'waiting_data'):
            current['decision'] = 'waiting_data'
        self.store.put_pause(current)
        if previous.get('stage') != stage or previous.get('reason') != reason:
            logger.warning(
                'Entry check deferred for %s [%s]: %s', current['request']['pair'], stage, reason
            )
        return reason

    def evaluate(self, meta, now, *, quote=None):
        snapshot = self.snapshots.get(meta['request']['pair'])
        if not snapshot:
            raise EntryDataError('等待新鲜行情')
        observed = snapshot['observed_at']
        signal = snapshot.get('signals', {}).get(meta['request']['direction'])
        if signal and 0 <= now - signal['at'] < 900:
            meta['last_adverse'] = max(meta.get('last_adverse', 0), signal['at'])
            meta['ever_paused'] = True
            if meta['decision'] not in ENDED:
                meta['decision'] = 'paused'
            self.store.put_pause(meta)
        signal = analyze_volume(snapshot['volume'], observed, now, meta['request']['direction'])
        # Persist every observed shock before attempting full price-history validation.
        if signal:
            meta['last_adverse'] = max(meta.get('last_adverse', 0), signal['at'])
            meta['ever_paused'] = True
            if meta.get('decision') not in ENDED:
                meta['decision'] = 'paused'
            self.store.put_pause(meta)
        short = meta['request']['direction'] == 'short'
        quote = snapshot['bid' if short else 'ask'] if quote is None else quote
        updated = advance_pause(
            meta,
            snapshot['minute'],
            signal,
            observed,
            now,
            quote=quote,
            trend=snapshot.get('trend'),
        )
        if updated['decision'] not in ENDED:
            # A missed intrabar reversal cannot be reconstructed from final 5m candles.
            if updated.get('ever_paused') and now - snapshot['continuous_since'] < 900:
                updated['decision'] = 'waiting_continuity'
            elif not updated.get('ever_paused') and not updated.get('history_uncertain'):
                updated['decision'] = 'ready'
        updated.pop('check_error', None)
        updated['last_check'] = dict(
            at=now,
            observed_at=observed,
            quote=quote,
            quiet_until=(
                updated.get('last_adverse', updated.get('quiet_after', updated['created'])) + 900
            )
            if updated.get('ever_paused')
            else None,
            continuity_until=snapshot['continuous_since'] + 900
            if updated.get('ever_paused')
            else None,
        )
        self.store.put_pause(updated)
        return updated

    def prepare(self, request, ref, *, now=None):
        """Return True when the initial request was retained without placing an order."""
        if request.get('entry_price') is None or not ref:
            return False
        now = time.time() if now is None else now
        meta = self.find(ref)
        if meta:
            # Restores have already claimed a distinct request identity.
            if meta['phase'] != 'resuming' or meta.get('decision') in ENDED:
                raise EntryRiskError('该入场计划未获准恢复')
            return False
        existing = Trade.get_trades(
            [Trade.is_open.is_(True), Trade.pair == request['pair']]
        ).first()
        if existing:
            return False  # Initial limits only; never auto-replenish an existing position.
        for old in self.store.pauses():
            if old['request']['pair'] == request['pair'] and old['phase'] not in ('filled', 'done'):
                if old['phase'] in ('canceling', 'resuming', 'unknown', 'watching'):
                    raise EntryRiskError('该币种旧入场计划尚待核对，请先 /request 查询或取消')
                self.cancel(old['ref'], 'replaced')
        meta = dict(
            ref=ref,
            attempt_ref=ref,
            request=deepcopy(request),
            created=now,
            deadline=request.get('entry_deadline', now + request['ttl_seconds']),
            phase='submitting',
            decision='normal',
            attempts=0,
        )
        self.store.put_pause(meta)
        try:
            snapshot = self.snapshots[request['pair']]
            signal = analyze_volume(
                snapshot['volume'], snapshot['observed_at'], now, request['direction']
            )
            signal = signal or snapshot.get('signals', {}).get(request['direction'])
            if signal and 0 <= now - signal['at'] < 900:
                meta.update(last_adverse=signal['at'], ever_paused=True, decision='paused')
        except (KeyError, ValueError):
            meta['decision'] = 'waiting_data'
        if meta['decision'] != 'normal':
            meta.update(phase='paused', ever_paused=True, quiet_after=now)
            self.store.put_pause(meta)
            self.notice(meta, '入场暂缓，等待行情与恢复条件；原限价、SL/TP及截止时间保留。')
            return True
        return False

    def submitted(self, ref, result):
        meta = self.find(ref)
        if not meta or result.get('pause_ref'):
            return
        if len(result.get('order_ids', [])) != 1:
            raise ValueError('原限价订单关联尚未确认')
        meta.pop('zero_receipt', None)
        meta.update(phase='watching', trade_id=result['id'], order_id=result['order_ids'][0])
        self.store.put_pause(meta)

    def guard(self, ref, *, now=None):
        """Final pre-SDK check; never called for closing-only orders."""
        meta = self.find(ref) if ref else None
        if not meta:
            return
        now = time.time() if now is None else now
        if self.monitor:
            self.snapshots.update(
                self.monitor.tick(datetime.fromtimestamp(now, timezone.utc), self.store.pauses())
            )
        if (
            self.bot.state != State.RUNNING
            or now >= meta['deadline']
            or meta.get('decision') in ENDED
            or meta['phase'] not in ('submitting', 'resuming')
        ):
            raise EntryRiskError('入场计划已暂停、结束或到期，本次未提交')
        stage = '入场放量' if meta['phase'] == 'submitting' else '恢复盘口'
        try:
            if meta['phase'] == 'submitting':
                snap = self.snapshots[meta['request']['pair']]
                signal = analyze_volume(
                    snap['volume'], snap['observed_at'], now, meta['request']['direction']
                )
                signal = signal or snap.get('signals', {}).get(meta['request']['direction'])
                if signal and 0 <= now - signal['at'] < 900:
                    meta.update(last_adverse=signal['at'], ever_paused=True, decision='paused')
                    self.store.put_pause(meta)
                    raise EntryDataError('新增逆向放量')
            else:
                # Actual executable best price determines whether the original
                # LIMIT may fill immediately; configured target pricing does not.
                request = meta['request']
                book = self.bot.exchange.fetch_l2_order_book(request['pair'], 1)
                bid, ask = float(book['bids'][0][0]), float(book['asks'][0][0])
                if not all(isfinite(v) and v > 0 for v in (bid, ask)) or bid > ask:
                    raise EntryDataError('恢复前盘口无效')
                if self.bot.state != State.RUNNING:
                    raise EntryDataError('机器人已停止新入场')
                quote = bid if request['direction'] == 'short' else ask
                # I/O can cross freshness, candle boundaries or the original TTL.
                stage = '恢复条件'
                meta = self.evaluate(meta, time.time(), quote=quote)
                if meta['decision'] != 'ready':
                    raise EntryDataError(PAUSE_REASONS.get(meta['decision'], '原价恢复条件已变化'))
        except Exception as error:
            if isinstance(error, EntryRiskError):
                raise
            reason = self._check_failed(meta, stage, error, time.time())
            raise EntryRiskError(f'入场检查未通过（{stage}：{reason}），本次未向交易所提交') from None

    def withdraw(self, meta):
        """Commit receipt evidence before native zero-fill deletion can remove the trade."""
        trade = Trade.get_trades([Trade.id == meta['trade_id']]).first()
        if not trade:
            if (
                meta.get('zero_receipt', {}).get('id') == meta['order_id']
                or meta['decision'] in ENDED
            ):
                meta['phase'] = 'paused' if meta['decision'] not in ENDED else 'done'
                self.store.put_pause(meta)
            return
        order = trade.select_order_by_order_id(meta['order_id'])
        if not order:
            raise ValueError('待撤订单记录缺失')
        install_entry_receipt_guard(self.strategy, self.bot)
        with self.bot._exit_lock:
            receipt = _receipt(
                self.bot.exchange.fetch_order(order.order_id, trade.pair), trade, order.order_id
            )
            if receipt['status'] == 'open':
                receipt = _receipt(
                    self.bot.exchange.cancel_order_with_result(
                        order.order_id, trade.pair, order.safe_amount
                    ),
                    trade,
                    order.order_id,
                )
            if receipt['status'] not in TERMINAL:
                return
            row = self.store.get(meta['attempt_ref'])
            if row and row['state'] in ('submitting', 'unknown'):
                self.store.set_status(
                    meta['attempt_ref'],
                    'submitted',
                    dict(id=trade.id, leverage=trade.leverage, order_ids=[order.order_id]),
                )
            if receipt['filled'] > 0:
                self.bot.update_trade_state(trade, order.order_id, receipt)
                Trade.commit()
                meta['phase'] = 'filled'
                self.store.put_pause(meta)
                self.notice(meta, '撤单核对发现成交，已交由原持仓保护流程管理，不再补单。')
                return
            meta['zero_receipt'] = receipt
            self.store.put_pause(meta)
            self.bot.handle_cancel_enter(trade, receipt, order, '逆向放量暂停 / 计划结束')
            Trade.commit()
            if Trade.get_trades([Trade.id == meta['trade_id']]).first() is not None:
                raise ValueError('原挂单清理尚未确认')
            self.strategy.delete_manual_plan(trade.pair, trade_id=trade.id)
            meta['phase'] = 'paused' if meta['decision'] not in ENDED else 'done'
            self.store.put_pause(meta)
            self.notice(
                meta,
                '已确认零成交撤单，临时暂停，保留原价计划。' if meta['phase'] == 'paused' else '计划已结束，已确认零成交撤单，不再恢复。',
            )

    def reconcile(self, meta):
        ref = meta['attempt_ref']
        row = self.store.get(ref)
        orders = Trade.session.query(Order).filter(Order.ft_order_tag.endswith('|req=' + ref)).all()
        orders = [
            o
            for o in orders
            if o.ft_pair == meta['request']['pair']
            and o.ft_order_side == ('sell' if meta['request']['direction'] == 'short' else 'buy')
        ]
        if len(orders) == 1:
            order = orders[0]
            meta.pop('zero_receipt', None)
            meta.update(phase='watching', order_id=order.order_id, trade_id=order.ft_trade_id)
            self.store.put_pause(meta)
        elif row and row['state'] == 'resolved':
            meta['phase'] = 'done'
            self.store.put_pause(meta)
        elif row and row['state'] == 'rejected':
            # A definite local rejection is not a new adverse-volume observation.
            meta.setdefault('quiet_after', meta['created'])
            previous = meta.get('last_rejection', {})
            if previous.get('ref') != ref:
                meta['retry_after'] = time.time() + 60
            meta.update(
                phase='paused',
                ever_paused=True,
                last_rejection=dict(
                    ref=ref, at=row['updated'], message=row['result'].get('message', '提交前检查未通过')
                ),
            )
            self.store.put_pause(meta)
        else:
            meta['phase'] = 'unknown'
            self.store.put_pause(meta)

    def adopt(self):
        # Only durable, identifiable original initial limits can be recovered.
        for trade in Trade.get_open_trades():
            if trade.nr_of_successful_entries or trade.has_open_position:
                continue
            entries = [
                o for o in trade.orders if o.ft_order_side == trade.entry_side and o.ft_is_open
            ]
            if len(entries) != 1:
                continue
            order = entries[0]
            if (
                order.order_type != 'limit'
                or order.safe_filled
                or '|req=' not in (order.ft_order_tag or '')
            ):
                continue
            ref = order.ft_order_tag.rsplit('|req=', 1)[1]
            if self.find(ref):
                continue
            row = self.store.get(ref)
            ttl = entry_ttl(order)
            if not row or ttl is None or not row['request'].get('stop_loss'):
                continue
            request = deepcopy(row['request'])
            created = order.order_date_utc.timestamp()
            meta = dict(
                ref=ref,
                attempt_ref=ref,
                request=request,
                created=created,
                deadline=created + ttl,
                phase='watching',
                decision='normal',
                attempts=0,
                trade_id=trade.id,
                order_id=order.order_id,
                version=1,
            )
            plan = self.strategy._plan(trade)
            if any(
                plan.get(k) != request.get(k) for k in ('stop_loss', 'exit_points', *RULE_FIELDS)
            ):
                # Previous edits lack a complete timestamped version history. May pause,
                # but never restore stale protection values as though they were current.
                meta['history_uncertain'] = True
            self.store.put_pause(meta)

    def retire_replaced(self, trade_id):
        if Trade.get_trades([Trade.id == trade_id]).first() is not None:
            raise ValueError('旧交易尚未清理，不能结束替换确认')
        for meta in self.store.pauses():
            if meta.get('trade_id') == trade_id and meta['decision'] == 'replaced':
                meta['phase'] = 'done'
                self.store.put_pause(meta)

    def revise_trade(self, trade_id, plan):
        for meta in self.store.pauses():
            if meta.get('trade_id') != trade_id or meta['phase'] in ('filled', 'done'):
                continue
            if meta['decision'] in ENDED:
                continue  # Editing an ended plan cannot revive it.
            meta['original_created'] = meta.get('original_created', meta['created'])
            meta['created'] = time.time()
            meta['checked_until'] = meta['created']
            meta['version'] = meta.get('version', 1) + 1
            for key in ('stop_loss', 'exit_points', *RULE_FIELDS, *UNIT_FIELDS, 'target_basis'):
                if key in plan:
                    meta['request'][key] = deepcopy(plan[key])
                else:
                    meta['request'].pop(key, None)
            for key in (
                'crossed',
                'crossed_at',
                'range_crossed_at',
                'history_uncertain',
                'low_water',
                'high_water',
            ):
                meta.pop(key, None)
            self.store.put_pause(meta)

    def tick(self, now):
        stamp = now.timestamp()
        if not self._adopted:
            self.adopt()
            self._adopted = True
        if self.monitor:
            self.snapshots.update(self.monitor.tick(now, self.store.pauses()))
        for meta in self.store.pauses():
            if meta['phase'] in ('filled', 'done'):
                continue
            try:
                if meta['phase'] in ('submitting', 'resuming', 'unknown'):
                    self.reconcile(meta)
                    continue
                if stamp >= meta['deadline'] and meta['decision'] not in ENDED:
                    meta['decision'] = 'expired'
                    self.store.put_pause(meta)
                if meta['phase'] == 'watching':
                    trade = Trade.get_trades([Trade.id == meta['trade_id']]).first()
                    order = trade.select_order_by_order_id(meta['order_id']) if trade else None
                    if not order or not order.ft_is_open:
                        meta['phase'] = 'filled' if order and order.safe_filled > 0 else 'done'
                        self.store.put_pause(meta)
                        continue
                if meta['decision'] not in ENDED:
                    try:
                        meta = self.evaluate(meta, stamp)
                    except (ValueError, KeyError) as error:
                        self._check_failed(meta, '恢复条件', error, stamp)
                        meta = self.store.get_pause(meta['ref'])
                        # A volume shock can be known even while 1m history is unavailable.
                        if meta['phase'] != 'canceling' and (
                            meta['phase'] != 'watching' or not meta.get('ever_paused')
                        ):
                            continue
                if meta['phase'] == 'watching':
                    # Range/reclaim are only restoration gates, never reasons to cancel.
                    recent = stamp - meta.get('last_adverse', 0) < 900
                    if recent or meta['decision'] in ENDED:
                        meta['phase'] = 'canceling'
                        self.store.put_pause(meta)
                if meta['phase'] == 'canceling':
                    self.withdraw(meta)
                    continue
                if meta['phase'] != 'paused':
                    continue
                if meta['decision'] in ENDED:
                    meta['phase'] = 'done'
                    self.store.put_pause(meta)
                    self.notice(
                        meta,
                        {
                            'invalidated': '等待期间触及原硬SL，计划永久失效。',
                            'expired': '原截止时间已到，计划结束。',
                            'cancelled': '已按你的指令取消计划。',
                            'replaced': '原计划已被新指令替换。',
                        }[meta['decision']],
                    )
                    continue
                if (
                    meta['decision'] != 'ready'
                    or self.bot.state != State.RUNNING
                    or stamp < meta.get('retry_after', 0)
                ):
                    continue
                if Trade.get_trades(
                    [Trade.is_open.is_(True), Trade.pair == meta['request']['pair']]
                ).first():
                    continue
                meta['attempts'] += 1
                ref = self.store.reference(meta['ref'], 'resume:' + str(meta['attempts']))
                meta.update(attempt_ref=ref, phase='resuming')
                self.store.put_pause(
                    meta
                )  # Claim before any outbound submission, including crashes.
                request = {**meta['request'], 'entry_deadline': int(meta['deadline'])}
                result = self.submit(request, ref)
                if result and result['state'] == 'submitted':
                    self.submitted(ref, result['result'])
                    self.notice(meta, f"已恢复原价限价 {request['entry_price']:g}，SL/TP与截止时间保留。")
                else:
                    self.reconcile(self.store.get_pause(meta['ref']))
            except Exception as error:
                logger.warning('Entry pause check deferred (%s)', type(error).__name__)
                Trade.rollback()
