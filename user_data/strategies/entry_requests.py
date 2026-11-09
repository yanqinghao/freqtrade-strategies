"""Durable entry intent ledger, separate from native trade persistence."""
import hashlib
import json
import re
import sqlite3
import time
from contextlib import closing
from pathlib import Path


class EntryRequests:
    def __init__(self, path):
        self.path = Path(path)

    @staticmethod
    def reference(chat_id, message_id):
        return hashlib.blake2s(f'{chat_id}:{message_id}'.encode(), digest_size=12).hexdigest()

    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        db.execute(
            'CREATE TABLE IF NOT EXISTS entry_requests ('
            'ref TEXT PRIMARY KEY, request TEXT NOT NULL, state TEXT NOT NULL, '
            'result TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL)'
        )
        db.execute(
            'CREATE TABLE IF NOT EXISTS entry_pauses (ref TEXT PRIMARY KEY, body TEXT NOT NULL)'
        )
        return db

    @staticmethod
    def _ref(ref):
        if not isinstance(ref, str) or not re.fullmatch('[0-9a-f]{24}', ref):
            raise ValueError('请求编号无效，请从开单回执或 /request 列表复制')
        return ref

    @staticmethod
    def _row(row):
        if row is None:
            return None
        return {
            **dict(row),
            'request': json.loads(row['request']),
            'result': json.loads(row['result']),
        }

    def reserve(self, ref, request):
        self._ref(ref)
        payload = json.dumps(request, sort_keys=True, allow_nan=False)
        stamp = time.time()
        with closing(self._connect()) as db, db:
            inserted = (
                db.execute(
                    'INSERT OR IGNORE INTO entry_requests VALUES (?,?,?,?,?,?)',
                    (ref, payload, 'reserved', '{}', stamp, stamp),
                ).rowcount
                == 1
            )
            raw = db.execute('SELECT * FROM entry_requests WHERE ref=?', (ref,)).fetchone()
            if raw['request'] != payload:
                raise ValueError('该消息编号已绑定其他参数；编辑或重放原消息不会提交新订单，请主动发送新指令')
            row = self._row(raw)
        return row, inserted

    def get(self, ref):
        self._ref(ref)
        with closing(self._connect()) as db:
            return self._row(
                db.execute('SELECT * FROM entry_requests WHERE ref=?', (ref,)).fetchone()
            )

    def set_status(self, ref, state, result):
        self._ref(ref)
        if state not in ('submitting', 'submitted', 'rejected', 'unknown', 'resolved'):
            raise ValueError('请求状态无效')
        payload = json.dumps(result, sort_keys=True, allow_nan=False)
        with closing(self._connect()) as db, db:
            if (
                db.execute(
                    'UPDATE entry_requests SET state=?,result=?,updated=? WHERE ref=?',
                    (state, payload, time.time(), ref),
                ).rowcount
                != 1
            ):
                raise ValueError('未找到请求')
        return self.get(ref)

    def recent(self, page=1):
        if type(page) is not int or not 1 <= page <= 1000000:
            raise ValueError('页码须为1至1000000的整数')
        with closing(self._connect()) as db:
            return [
                self._row(row)
                for row in db.execute(
                    'SELECT * FROM entry_requests ORDER BY created DESC,ref DESC LIMIT 5 OFFSET ?',
                    ((page - 1) * 5,),
                )
            ]

    def unresolved(self, pair=None):
        with closing(self._connect()) as db:
            rows = [
                self._row(row)
                for row in db.execute(
                    "SELECT * FROM entry_requests WHERE state IN ('submitting','unknown')"
                )
            ]
        return [row for row in rows if pair is None or row['request']['pair'] == pair]

    def get_pause(self, ref):
        self._ref(ref)
        with closing(self._connect()) as db:
            row = db.execute('SELECT body FROM entry_pauses WHERE ref=?', (ref,)).fetchone()
            return json.loads(row['body']) if row else None

    def pauses(self):
        with closing(self._connect()) as db:
            return [json.loads(row['body']) for row in db.execute('SELECT body FROM entry_pauses')]

    def put_pause(self, meta):
        self._ref(meta['ref'])
        payload = json.dumps(meta, sort_keys=True, allow_nan=False)
        with closing(self._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute('SELECT body FROM entry_pauses WHERE ref=?', (meta['ref'],)).fetchone()
            if old:
                decision = json.loads(old['body']).get('decision')
                if (
                    decision in ('invalidated', 'expired', 'cancelled', 'replaced')
                    and meta.get('decision') != decision
                ):
                    raise ValueError('已结束的入场计划不能恢复')
            db.execute(
                'INSERT INTO entry_pauses VALUES (?,?) '
                'ON CONFLICT(ref) DO UPDATE SET body=excluded.body',
                (meta['ref'], payload),
            )
