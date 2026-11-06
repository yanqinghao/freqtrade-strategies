"""User-defined public-market watches. No trading plans or order access."""
import json
import logging
import math
import time
import uuid
from contextlib import closing
from copy import deepcopy
from datetime import datetime, timezone

from market_monitor import HourlyMarketMonitor, MONITOR_PAIRS, TIMEFRAMES

logger = logging.getLogger(__name__)
MODES = {
    'touch': '到价／进入区域',
    'close': '收盘站上／跌破',
    'reject': '反弹受阻／回踩守住',
    'retest': '突破后回踩／跌破后反抽',
    'fake': '假突破／假跌破收回',
}
STATES = {
    'waiting': '等待前置条件',
    'confirm': '等待收盘确认',
    'retest': '等待回踩／反抽',
    'confirmed': '条件已满足',
    'invalid': '条件已失效',
    'expired': '已到期／窗口结束',
    'cancelled': '已取消',
}
ACTIVE = ('waiting', 'confirm', 'retest')


def price(value):
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError):
        raise ValueError('请输入大于0的价格，例如 95000.5') from None
    if isinstance(value, bool) or not math.isfinite(result) or result <= 0:
        raise ValueError('请输入有限且大于0的价格')
    return result


def validate(spec):
    s = deepcopy(spec)
    if s.get('mode') not in MODES or s.get('side') not in ('long', 'short'):
        raise ValueError('请选择监控模式和多空方向')
    if s.get('pair') not in MONITOR_PAIRS or s.get('tf') not in TIMEFRAMES:
        raise ValueError('支持BTC、ETH、SOL；周期15m、1h、4h、1d')
    for key in ('low', 'high', 'level', 'confirm'):
        if key in s:
            s[key] = price(s[key])
    required = {
        'touch': ('low', 'high'),
        'close': ('confirm',),
        'reject': ('low', 'high', 'confirm'),
        'retest': ('low', 'high', 'level', 'confirm'),
        'fake': ('level', 'confirm'),
    }
    if any(k not in s for k in required[s['mode']]):
        raise ValueError('价格条件未填写完整，请返回修改')
    if 'low' in s and 'high' in s and s['low'] > s['high']:
        raise ValueError('区域下限不能大于上限；单点请填写相同价格')
    short = s['side'] == 'short'
    if s['mode'] in ('reject', 'retest'):
        if (short and s['confirm'] > s['low']) or (not short and s['confirm'] < s['high']):
            raise ValueError('空头确认价须≤区域下限，多头确认价须≥区域上限')
    if s['mode'] == 'retest' and not s['low'] <= s['level'] <= s['high']:
        raise ValueError('回踩／反抽区域须包含突破位')
    if s['mode'] == 'fake' and (
        short and s['confirm'] > s['level'] or not short and s['confirm'] < s['level']
    ):
        raise ValueError('空头收回价须≤关键位，多头收回价须≥关键位')
    if s.get('invalid') is not None:
        s['invalid'] = price(s['invalid'])
        refs = [s[k] for k in required[s['mode']]]
        if short and s['invalid'] <= max(refs) or not short and s['invalid'] >= min(refs):
            raise ValueError('空头失效价须高于所有条件价，多头须低于所有条件价')
    else:
        s['invalid'] = None
    if type(s.get('window')) is not int or s['window'] not in (1, 3, 6):
        raise ValueError('确认窗口请选择1、3或6根')
    if type(s.get('ttl')) is not int or not 60 <= s['ttl'] <= 5 * 86400:
        raise ValueError('有效期须为1分钟至5天')
    return s


def stamp(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime('%m-%d %H:%M:%S UTC')


def describe(s):
    short = s['side'] == 'short'
    relation = '低于' if short else '高于'
    mode = s['mode']
    zone = f"{s['low']:.8g}–{s['high']:.8g}" if 'low' in s else ''
    if mode == 'touch':
        rule = f'当前价进入 {zone}（包含边界）'
    elif mode == 'close':
        rule = f"新 {s['tf']} 收盘严格{relation} {s['confirm']:.8g}"
    else:
        first = (
            f'先进入 {zone}'
            if mode == 'reject'
            else f"先 {s['tf']} 收盘{relation} {s['level']:.8g}，再在后续K线回到 {zone}"
            if mode == 'retest'
            else f"先{'向上刺穿' if short else '向下刺穿'} {s['level']:.8g}"
        )
        rule = f"{first}；再等 {s['tf']} 收盘严格{relation} {s['confirm']:.8g}"
        rule += f"\n确认窗口：{s['window']}根（" + ('突破后的下一根起' if mode == 'retest' else '含触发的这一根') + '）'
    rule += (
        '\n即时失效：价格' + ('≥' if short else '≤') + f"{s['invalid']:.8g}"
        if s.get('invalid') is not None
        else '\n未设价格失效位；到期自动结束'
    )
    return rule


def _candle(row, width):
    if len(row) < 6:
        raise ValueError('行情缺少字段')
    t, o, h, l, c = (float(v) for v in row[:5])
    if not all(math.isfinite(v) for v in (t, o, h, l, c)) or t % (width * 1000):
        raise ValueError('行情时间无效')
    if not 0 < l <= min(o, c) <= max(o, c) <= h:
        raise ValueError('行情价格无效')
    return dict(start=t / 1000, end=t / 1000 + width, high=h, low=l, close=c)


def evaluate(watch, rows, quote, now):
    """Evaluate the latest close and observed quote in timestamp order."""
    w = deepcopy(watch)
    s = w['spec']
    short = s['side'] == 'short'
    mode = s['mode']
    width = TIMEFRAMES[s['tf']] // 1000
    events = []

    def finish(state, reason):
        w.update(state=state, reason=reason)
        events.append(reason)

    def beyond(value, level):
        return value < level if short else value > level

    def invalid(value):
        return s['invalid'] is not None and (
            value >= s['invalid'] if short else value <= s['invalid']
        )

    def arm(start, label):
        w.update(state='confirm', armed=start)
        if mode != 'retest':
            w['deadline'] = start + width * s['window']
        events.append(label + f"；等待收盘，最晚 {stamp(w['deadline'])}")

    def observe_quote():
        if w['state'] in ACTIVE and quote is not None:
            try:
                p = price(quote['price'])
                fresh = w['created'] <= quote['time'] <= now and now - quote['time'] <= 30
            except (ValueError, KeyError, TypeError):
                fresh = False
            if fresh:
                w['last_price'], w['checked'] = p, quote['time']
                if invalid(p):
                    finish('invalid', f'当前价 {p:.8g} 触及失效价，监控结束')
                elif mode == 'touch' and s['low'] <= p <= s['high']:
                    finish('confirmed', f'当前价 {p:.8g} 已进入观察区')
                elif w['state'] in ('waiting', 'retest') and (
                    mode != 'retest' or w['state'] == 'retest'
                ):
                    start = int(quote['time']) // width * width
                    touched = (
                        (p > s['level'] if short else p < s['level'])
                        if mode == 'fake'
                        else s['low'] <= p <= s['high']
                        if mode in ('reject', 'retest')
                        else False
                    )
                    if (
                        touched
                        and (mode != 'retest' or start >= w['armed'])
                        and quote['time'] < w.get('deadline', float('inf'))
                    ):
                        arm(start, f'当前价 {p:.8g} 已' + ('刺穿关键位' if mode == 'fake' else '进入观察区'))

    if w['state'] not in ACTIVE:
        return w, None
    if now >= w['expires']:
        finish('expired', '有效期已到，监控结束')
    elif w.get('deadline') is not None and now > w['deadline'] + 90:
        finish('expired', '确认窗口已结束，未取得窗口内有效确认；不补发历史信号')
    else:
        candle = None
        try:
            closed = [_candle(r, width) for r in rows if float(r[0]) / 1000 + width <= now]
            candle = max(closed, key=lambda c: c['start']) if closed else None
        except (ValueError, TypeError, IndexError, OverflowError):
            candle = None
        # Fetches can straddle a close: preserve when the quote was observed.
        quote_first = bool(candle and quote and quote.get('time', now) < candle['end'])
        if quote_first:
            observe_quote()
        if w['state'] in ACTIVE and candle and candle['start'] > w.get('last_candle', -1):
            w['last_candle'] = candle['start']
            end, start, close = candle['end'], candle['start'], candle['close']
            if end > w['created'] and now - end <= 90:
                whole = start >= w['created']
                w['last_close'] = close
                w['last_close_at'] = end
                if whole and invalid(candle['high'] if short else candle['low']):
                    finish('invalid', f"{stamp(end)} 的K线触及失效价；失效优先")
                elif mode == 'close' and beyond(close, s['confirm']):
                    finish('confirmed', f"{stamp(end)} {s['tf']} 收盘 {close:.8g}，收盘条件已满足")
                elif mode in ('reject', 'fake', 'retest'):
                    if mode == 'retest' and w['state'] == 'waiting' and beyond(close, s['level']):
                        w.update(state='retest', armed=end, deadline=end + width * s['window'])
                        events.append(f"{stamp(end)} 收盘 {close:.8g} 已越过突破位；等待之后回踩／反抽")
                    else:
                        eligible = whole and (
                            mode != 'retest' or start >= w.get('armed', float('inf'))
                        )
                        if w['state'] in ('waiting', 'retest') and eligible:
                            touched = (
                                (
                                    candle['high'] > s['level']
                                    if short
                                    else candle['low'] < s['level']
                                )
                                if mode == 'fake'
                                else (candle['high'] >= s['low'] and candle['low'] <= s['high'])
                            )
                            if touched:
                                arm(start, '已刺穿关键位' if mode == 'fake' else '已触及观察区')
                        if (
                            w['state'] == 'confirm'
                            and start >= w['armed']
                            and end <= w['deadline']
                            and beyond(close, s['confirm'])
                        ):
                            finish(
                                'confirmed',
                                f"{stamp(end)} {s['tf']} 收盘 {close:.8g}；"
                                f"最高 {candle['high']:.8g} / 最低 {candle['low']:.8g}，条件已满足",
                            )
                    if w['state'] in ('confirm', 'retest') and end >= w['deadline']:
                        finish('expired', f"{stamp(end)} 确认窗口内未确认，本次跟踪结束")
        if not quote_first:
            observe_quote()
    message = None
    if events:
        message = (
            f"【条件监控 #{w['id']}】{s['pair'].split('/')[0]} · {MODES[mode]}\n"
            f"状态：{STATES[w['state']]} · {'关注做空' if short else '关注做多'}\n"
            + '\n'.join(events)
            + '\n'
            + describe(s)
            + f'\n记录时间：{stamp(now)}\n仅提醒；条件满足不代表趋势成立，也不会自动下单。'
        )
    return w, message


class WatchStore:
    def __init__(self, queue):
        self.queue = queue

    def _connect(self):
        db = self.queue._connect()
        try:
            db.execute(
                'CREATE TABLE IF NOT EXISTS condition_watches ('
                'id TEXT PRIMARY KEY, owner TEXT NOT NULL, request TEXT NOT NULL, '
                'revision INTEGER NOT NULL, data TEXT NOT NULL, UNIQUE(owner,request))'
            )
        except Exception:
            db.close()
            raise
        return db

    @staticmethod
    def _owner(owner):
        return json.dumps(list(owner), separators=(',', ':'))

    def create(self, spec, owner, request_id, *, now=None):
        now = time.time() if now is None else now
        s = validate(spec)
        owner = self._owner(owner)
        with closing(self._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute(
                'SELECT data FROM condition_watches WHERE owner=? AND request=?',
                (owner, request_id),
            ).fetchone()
            if old:
                return json.loads(old[0])
            active = [json.loads(r[0]) for r in db.execute('SELECT data FROM condition_watches')]
            if sum(w['state'] in ACTIVE for w in active) >= 60:
                raise ValueError('最多同时监控60条，请先取消不需要的条件')
            w = dict(
                id=uuid.uuid4().hex[:10],
                spec=s,
                owner=owner,
                revision=0,
                created=now,
                expires=now + s['ttl'],
                state='waiting',
                reason='',
                health='等待首次行情',
            )
            db.execute(
                'INSERT INTO condition_watches VALUES (?,?,?,?,?)',
                (w['id'], owner, request_id, 0, json.dumps(w)),
            )
        return w

    def list(self, owner=None):
        with closing(self._connect()) as db:
            rows = (
                db.execute('SELECT data FROM condition_watches')
                if owner is None
                else db.execute(
                    'SELECT data FROM condition_watches WHERE owner=?', (self._owner(owner),)
                )
            )
            return sorted(
                (json.loads(r[0]) for r in rows), key=lambda w: w['created'], reverse=True
            )

    def cancel(self, ident, owner, *, now=None):
        with closing(self._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute(
                'SELECT data FROM condition_watches WHERE id=? AND owner=?',
                (ident, self._owner(owner)),
            ).fetchone()
            if not row:
                raise ValueError('监控不存在或不属于当前用户')
            w = json.loads(row[0])
            if w['state'] in ACTIVE:
                w.update(state='cancelled', reason='已手动取消', revision=w['revision'] + 1)
                db.execute(
                    'UPDATE condition_watches SET revision=?,data=? WHERE id=?',
                    (w['revision'], json.dumps(w), ident),
                )
                db.execute('DELETE FROM alerts WHERE key=?', ('market-watch:' + ident,))
            return w

    def apply(self, old, new, message, *, now):
        if old == new:
            return False
        with closing(self._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            new = dict(new, revision=old['revision'] + 1)
            cursor = db.execute(
                'UPDATE condition_watches SET revision=?,data=? WHERE id=? AND revision=?',
                (new['revision'], json.dumps(new), old['id'], old['revision']),
            )
            if not cursor.rowcount:
                return False  # Cancellation or another check won the race.
            if message:
                key = 'market-watch:' + old['id']
                db.execute(
                    'INSERT INTO market_alert_events VALUES (?,?,?,?) ON CONFLICT(key) '
                    'DO UPDATE SET created=excluded.created,expires=excluded.expires',
                    (key, key, now, now + 300),
                )
                db.execute(
                    'INSERT INTO alerts VALUES (?,?,?,?,?,?) ON CONFLICT(key) DO UPDATE SET '
                    'version=excluded.version,message=excluded.message,'
                    'updated=excluded.updated,due=excluded.due',
                    (key, uuid.uuid4().hex, message, now, now, now),
                )
        return True


class ConditionMonitor(HourlyMarketMonitor):
    def __init__(self, store, clock=None, **kwargs):
        super().__init__(**kwargs)
        self.store = store
        self._clock = clock or time.time
        self._slot = None

    def check(self):
        cache = {}

        def fetch(pair, tf):
            key = pair, tf
            if key not in cache:
                try:
                    cache[key] = (self._fetch(pair, tf), self._clock())
                except Exception:
                    cache[key] = ([], self._clock())
            return cache[key]

        for w in self.store.list():
            if w['state'] not in ACTIVE:
                continue
            try:
                now = self._clock()
                s = w['spec']
                quote, rows = None, []
                if now < w['expires'] and now <= w.get('deadline', float('inf')) + 90:
                    ticks, received = fetch(s['pair'], '1m')
                    now = self._clock()
                    try:
                        latest = _candle(ticks[-1], 60)
                        if latest['start'] <= received < latest['end'] and now - received <= 30:
                            quote = dict(price=latest['close'], time=received)
                    except (ValueError, IndexError, TypeError):
                        pass
                    if s['mode'] != 'touch':
                        rows, _ = fetch(s['pair'], s['tf'])
                        now = self._clock()
                new, message = evaluate(w, rows, quote, now)
                valid_close = False
                if rows:
                    try:
                        width = TIMEFRAMES[s['tf']] // 1000
                        closed = [
                            _candle(r, width) for r in rows if float(r[0]) / 1000 + width <= now
                        ]
                        valid_close = bool(closed) and now - max(c['end'] for c in closed) < width
                    except (ValueError, TypeError, IndexError):
                        pass
                new['health'] = (
                    '行情正常'
                    if quote and (s['mode'] == 'touch' or valid_close)
                    else '行情暂不可用／不完整，缺失条件暂不判断'
                )
                self.store.apply(w, new, message, now=now)
            except Exception as error:
                logger.warning('Condition watch %s unavailable (%s)', w['id'], type(error).__name__)

    def tick(self, now):
        if self._future is not None:
            if not self._future.done():
                return
            try:
                self._future.result()
            except Exception as error:
                logger.warning('Condition worker unavailable (%s)', type(error).__name__)
            self._future = None
        slot = int(now.timestamp()) // 10
        if slot != self._slot:
            self._slot = slot
            self._future = self._executor.submit(self.check)
