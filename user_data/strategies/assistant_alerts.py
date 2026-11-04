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
        with closing(self._connect()) as db:
            return [
                dict(row)
                for row in db.execute(
                    'SELECT * FROM alerts WHERE due<=? ORDER BY due,created,key LIMIT ?',
                    (stamp, limit),
                )
            ]

    def acknowledge(self, row):
        with closing(self._connect()) as db, db:
            db.execute('DELETE FROM alerts WHERE key=? AND version=?', (row['key'], row['version']))

    def defer(self, row, until):
        with closing(self._connect()) as db, db:
            db.execute(
                'UPDATE alerts SET due=? WHERE key=? AND version=?',
                (until, row['key'], row['version']),
            )
