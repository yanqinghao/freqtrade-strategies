"""Durable, coalescing critical notifications; acknowledgments follow actual delivery."""
import sqlite3
import time
import uuid
from contextlib import closing
from pathlib import Path


class AlertQueue:
    def __init__(self, path):
        self.path = Path(path)

    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        try:
            db.execute(
                'CREATE TABLE IF NOT EXISTS alerts ('
                'key TEXT PRIMARY KEY, version TEXT NOT NULL, message TEXT NOT NULL, '
                'created REAL NOT NULL, updated REAL NOT NULL, due REAL NOT NULL)'
            )
            db.execute(
                'CREATE TABLE IF NOT EXISTS market_alert_events ('
                'key TEXT PRIMARY KEY, scope TEXT NOT NULL, '
                'created REAL NOT NULL, expires REAL NOT NULL)'
            )
        except Exception:
            db.close()
            raise
        return db

    def put(self, key, message, *, now=None):
        stamp = time.time() if now is None else now
        with closing(self._connect()) as db, db:
            db.execute(
                'INSERT INTO alerts VALUES (?,?,?,?,?,?) '
                'ON CONFLICT(key) DO UPDATE SET version=excluded.version, '
                'message=excluded.message, updated=excluded.updated, due=excluded.due',
                (key, uuid.uuid4().hex, message, stamp, stamp, stamp),
            )

    def pending(self, *, limit=3, now=None):
        stamp = time.time() if now is None else now
        with closing(self._connect()) as db, db:
            db.execute(
                'DELETE FROM alerts WHERE key IN '
                '(SELECT key FROM market_alert_events WHERE expires<=?)',
                (stamp,),
            )
            return [
                dict(row)
                for row in db.execute(
                    'SELECT * FROM alerts WHERE due<=? '
                    "ORDER BY (key LIKE 'market-volume:%' OR key LIKE 'market-watch:%' "
                    "OR key LIKE 'market-position:%'),"
                    'due,created,key LIMIT ?',
                    (stamp, limit),
                )
            ]

    def put_market_once(self, key, message, *, group, now):
        """Atomically remember a market event and enqueue a five-minute notice."""
        with closing(self._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            db.execute(
                'DELETE FROM alerts WHERE key IN '
                '(SELECT key FROM market_alert_events WHERE expires<=?)',
                (now,),
            )
            db.execute('DELETE FROM market_alert_events WHERE expires<?', (now - 86400,))
            if db.execute('SELECT 1 FROM market_alert_events WHERE key=?', (key,)).fetchone():
                return False
            last = db.execute(
                'SELECT MAX(created) FROM market_alert_events WHERE scope=?', (group,)
            ).fetchone()[0]
            if last is not None and now - last < 900:
                return False
            db.execute(
                'INSERT INTO market_alert_events VALUES (?,?,?,?)', (key, group, now, now + 300)
            )
            db.execute(
                'INSERT INTO alerts VALUES (?,?,?,?,?,?)',
                (key, uuid.uuid4().hex, message, now, now, now),
            )
        return True

    def acknowledge(self, row):
        with closing(self._connect()) as db, db:
            db.execute('DELETE FROM alerts WHERE key=? AND version=?', (row['key'], row['version']))

    def is_current(self, row, *, now=None):
        stamp = time.time() if now is None else now
        with closing(self._connect()) as db:
            return (
                db.execute(
                    'SELECT 1 FROM alerts WHERE key=? AND version=? AND due<=? '
                    'AND NOT EXISTS (SELECT 1 FROM market_alert_events '
                    'WHERE key=alerts.key AND expires<=?)',
                    (row['key'], row['version'], stamp, stamp),
                ).fetchone()
                is not None
            )

    def defer(self, row, until):
        with closing(self._connect()) as db, db:
            db.execute(
                'UPDATE alerts SET due=? WHERE key=? AND version=?',
                (until, row['key'], row['version']),
            )
