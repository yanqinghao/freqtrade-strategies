"""Hourly position snapshots and conditional advice; never execute trading actions."""
import logging
import time
from math import isfinite
from datetime import datetime, timezone, timedelta

from entry_risk import number, observed_quantities, saved_plan, effective_stop
from entry_volume import checked_rows
from market_monitor import (
    HourlyMarketMonitor,
    TIMEFRAMES,
    analyze_candles,
    analyze_trend,
    level_distance,
)
from order_management import position_profit
from trade_overview import _protection
from candle_exits import describe_rules

logger = logging.getLogger(__name__)
BEIJING = timezone(timedelta(hours=8))


def _positions(trades):
    result = []
    for trade in trades:
        if not trade.is_open:
            continue
        try:
            held, _ = observed_quantities(trade)
            if not held:
                continue
        except Exception:
            # Uncertain quantity is a reason to report uncertainty, not hide a position.
            pass
        result.append(trade)
    return result


def _fresh_frames(market, now):
    result = {}
    for tf, width in TIMEFRAMES.items():
        try:
            frame = market.get('frames', {})[tf]
            if frame['closed_at'].timestamp() != int(now.timestamp()) // (width // 1000) * (
                width // 1000
            ):
                continue
            number(frame['close'], positive=True)
            for key in ('support', 'resistance'):
                if frame.get(key) is not None:
                    number(frame[key], positive=True)
            if frame['trend'] in ('偏多', '偏空', '震荡'):
                result[tf] = frame
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
    return result


def _exit_quote(market, short, now):
    try:
        quote = market['quote']
        age = now.timestamp() - float(number(quote['observed_at'], positive=True))
        bid, ask = (float(number(quote[key], positive=True)) for key in ('bid', 'ask'))
        if not 0 <= age <= 30 or not all(isfinite(v) for v in (bid, ask)) or bid > ask:
            return None
        return ask if short else bid
    except (KeyError, TypeError, ValueError):
        return None


def _advice(trade, pnl, rate, frames, plan, state, stop, targets, stage, pending, protected):
    if pnl['pnl_status'] != '可用':
        return '先核对报价、成交记账和仓位数量，暂不给方向性建议。'
    if pending:
        return '原TP计划待应用，先核对计划与执行阶段，暂不给方向性建议。'
    if not plan or stop is None or not protected:
        return '优先核对原计划与交易所止损保护，暂不给方向性建议。'
    if state.get('full_exit') or any(
        o.ft_is_open and o.ft_order_side == trade.exit_side for o in trade.orders
    ):
        return '已有退出处理或退出订单，先核对成交与余量，避免重复提交退出。'
    if rate >= stop if trade.is_short else rate <= stop:
        return '现价已越过目标保护SL，优先核对止损执行和剩余仓位，避免重复退出。'
    if stage < len(targets) and (
        rate <= targets[stage] if trade.is_short else rate >= targets[stage]
    ):
        return f'已到当前TP{stage + 1}目标 {targets[stage]:g}，核对原策略止盈进度与剩余仓位。'
    if len(frames) != len(TIMEFRAMES):
        return '周期行情不完整或已过期，沿用原保护计划，暂不给方向性建议。'
    level = frames['4h'].get('resistance' if trade.is_short else 'support')
    level_name = '阻力' if trade.is_short else '支撑'
    if level is not None and (rate > level if trade.is_short else rate < level):
        move = '突破' if trade.is_short else '跌破'
        return f'现价盘中已{move}4h{level_name} {level:.8g}，先核对保护并等待收盘确认，暂不按顺势持有判断。'
    favorable, opposite = ('偏空', '偏多') if trade.is_short else ('偏多', '偏空')
    trends = {tf: frame['trend'] for tf, frame in frames.items()}
    if trends['4h'] == trends['1h'] == opposite:
        purpose = '保护浮盈' if pnl['profit_abs'] > 0 else '控制回撤'
        return f'4h与1h均逆向，可考虑减仓或退出以{purpose}；核对原失效条件，不放宽SL。'
    if trends['1d'] == trends['4h'] == favorable and trends['1h'] == trends['15m'] == opposite:
        move = '反弹' if trade.is_short else '回调'
        return f'大周期同向、短线{move}；观察上述4h{level_name}能否守住，失守后再评估减仓。'
    if trends['4h'] == trends['1h'] == favorable:
        return '顺势持有观察（4h/1h同向），沿用原SL/TP；接近原目标时按计划处理，不追价加仓。'
    return '走势震荡或周期分歧，等待收盘方向更清楚，继续遵守原保护计划。'


def build_position_report(strategy, trade, market, now):
    """Use current ORM state on the trading thread; all access here is read-only."""
    frames = _fresh_frames(market, now)
    rate = _exit_quote(market, trade.is_short, now)
    pnl = position_profit(trade, rate)
    try:
        quantity = float(observed_quantities(trade)[0])
    except Exception:
        quantity = None
    state = strategy._execution(trade)
    if not isinstance(state, dict):
        state = {}
    plan, stop, targets, active_targets, stage, pending = {}, None, [], [], 0, False
    try:
        plan = saved_plan(strategy, trade)
        stop = float(effective_stop(strategy, trade, plan))
        targets = sorted(
            {float(number(p, positive=True)) for p in plan.get('exit_points', [])},
            reverse=trade.is_short,
        )
        stage = state.get('stage', 0)
        if not isinstance(stage, int) or isinstance(stage, bool) or not 0 <= stage <= 3:
            raise ValueError('Invalid TP stage')
        active_targets = [float(number(p, positive=True)) for p in strategy._targets(plan, trade)]
        applied = trade.get_custom_data('assistant_plan', default={}) or {}
        pending = bool(
            applied
            and applied.get('exit_points') != plan.get('exit_points')
            and plan.get('tp_keep_progress_from') != applied.get('exit_points')
        )
    except (ValueError, TypeError, ArithmeticError):
        plan, stop, targets, active_targets, stage = {}, None, [], [], 0
    protected = False
    try:
        proof = state['protection']
        age = now.timestamp() - float(number(proof['checked_at'], positive=True))
        price = float(number(proof['stop_price'], positive=True))
        protected = (
            proof['status'] == 'confirmed'
            and 0 <= age <= 60
            and quantity is not None
            and number(proof['covered_amount']) >= number(quantity)
            and stop is not None
            and (price <= stop if trade.is_short else price >= stop)
        )
    except (KeyError, ValueError, TypeError):
        pass
    lines = [
        f"【持仓简报】{trade.pair} #{trade.id} {'空' if trade.is_short else '多'}",
        f"快照：{now.astimezone(BEIJING):%Y-%m-%d %H:%M:%S} 北京时间",
        f'均价 {trade.open_rate:.8g}｜账面数量 {trade.amount:g}｜杠杆 {trade.leverage:g}倍',
    ]
    lines.append(
        f'盘口退出参考价 {rate:.8g}（取样 '
        f"{datetime.fromtimestamp(float(market['quote']['observed_at']), BEIJING):%H:%M:%S} 北京时间）"
        if rate is not None
        else '盘口退出参考价不可用。'
    )
    lines.append(
        f"浮盈亏：{pnl['profit_abs']:+.4f} USDT｜收益率 {pnl['profit']:+.2f}%"
        if pnl['pnl_status'] == '可用'
        else f"浮盈亏不可用：{pnl['pnl_status']}。"
    )
    if pnl['pnl_status'] == '成交待记账' and quantity is not None:
        lines.append(f'已观察数量 {quantity:g}，账面尚未完成记账，不能把账面零数量视为没有成交。')
    lines.append(
        '走势：'
        + '｜'.join(
            f"{tf} {frames[tf]['trend']}" if tf in frames else f'{tf} 不可用' for tf in TIMEFRAMES
        )
    )
    for tf in ('1d', '4h'):
        if tf in frames:
            frame = frames[tf]
            support = level_distance(frame.get('support'), rate, support=True)
            resistance = level_distance(frame.get('resistance'), rate, support=False)
            lines.append(
                f'{tf}参考：{support}；{resistance}；来源收盘 '
                f"{frame['closed_at'].astimezone(BEIJING):%m-%d %H:%M} 北京时间。"
            )
    if plan and stop is not None:
        lines.append(f"原硬SL {plan['stop_loss']:g}｜目标保护SL {stop:g}｜原TP {targets}｜已记录阶段 {stage}。")
        if pending:
            lines.append('TP计划待应用，当前阶段不能视为新计划进度。')
        rules = describe_rules(plan, state)
        if rules:
            lines.append(rules)
    else:
        lines.append('原SL/TP计划不可用，请核对 /overview。')
    lines.append(
        _protection(trade, state, quantity, stop, now)
        if quantity is not None
        else '交易所SL：当前数量不可核实，请核对持仓及止损覆盖。'
    )
    lines.append(
        '建议：'
        + _advice(
            trade, pnl, rate, frames, plan, state, stop, active_targets, stage, pending, protected
        )
    )
    lines.append('盈亏按剩余仓位及已记录费用估算，收益率含杠杆，已实现盈亏另计；操作前核对 /overview。')
    return '\n'.join(lines)


class PositionMonitor(HourlyMarketMonitor):
    def __init__(self, fetch=None, *, clock=time.time, **kwargs):
        super().__init__(fetch, **kwargs)
        self.clock = clock

    def collect(self, targets, slot):
        markets = {}
        for pair in sorted({pair for _, pair in targets}):
            market = {'frames': {}}
            for tf, width in TIMEFRAMES.items():
                try:
                    observed = self.clock()
                    rows = self._fetch(pair, tf)
                    checked_rows(rows, width // 1000, observed, self.clock(), 51)
                    when = datetime.fromtimestamp(observed, timezone.utc)
                    analyze = analyze_candles if tf in ('1d', '4h') else analyze_trend
                    market['frames'][tf] = analyze(rows, tf, when)
                except Exception as error:
                    logger.warning(
                        'Position candles unavailable for %s %s (%s)',
                        pair,
                        tf,
                        type(error).__name__,
                    )
            try:
                observed = self.clock()
                book = self._get_client().fetch_order_book(
                    pair if self._futures else pair.split(':')[0], limit=5
                )
                quote = dict(bid=book['bids'][0][0], ask=book['asks'][0][0], observed_at=observed)
                market['quote'] = quote
            except Exception as error:
                logger.warning('Position quote unavailable for %s (%s)', pair, type(error).__name__)
            markets[pair] = market
        return dict(targets=targets, slot=slot, markets=markets)

    def tick(self, now, strategy, trades):
        active = _positions(trades)
        if self._future is not None:
            if not self._future.done():
                return
            try:
                data = self._future.result()
                stamp = self.clock()
                if data['slot'] == int(stamp) // 3600:
                    targets = set(data['targets'])
                    for trade in active:
                        if (trade.id, trade.pair) not in targets:
                            continue
                        stamp = self.clock()
                        if data['slot'] != int(stamp) // 3600:
                            break
                        try:
                            message = build_position_report(
                                strategy,
                                trade,
                                data['markets'][trade.pair],
                                datetime.fromtimestamp(stamp, timezone.utc),
                            )
                            key = f"market-position:{data['slot']}:{trade.id}"
                            strategy.alert_queue.put_market_once(key, message, group=key, now=stamp)
                        except Exception as error:
                            logger.warning(
                                'Position report unavailable for #%s (%s)',
                                trade.id,
                                type(error).__name__,
                            )
            except Exception as error:
                logger.warning('Position worker unavailable (%s)', type(error).__name__)
            self._future = None
        hour = int(now.timestamp()) // 3600
        if active and self._hour != hour:
            self._hour = hour
            self._future = self._executor.submit(
                self.collect, tuple((t.id, t.pair) for t in active), hour
            )
