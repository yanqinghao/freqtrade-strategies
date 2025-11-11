# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from functools import reduce
from pandas import DataFrame, Series
import numpy as np
from typing import Optional, Tuple, Dict, Any

# --------------------------------
import talib.abstract as ta
import pandas_ta as pta
import pandas as pd  # noqa
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import (
    IntParameter,
)
import itertools
import logging
import os
import json
import time
import ccxt
import sqlite3
import math


class HedgeManager:
    """
    极简版：API + SQLite 写在一起
    - 使用 original_config.exchange 构建 ccxt（默认 futures）
    - 仅一张表：hedge_orders（open/close 状态 + 简单 meta_json）
    - notional 用“名义金额(USDT)”语义；下单数量 = notional*leverage/price，并按交易所步长舍入
    """

    def __init__(self, strategy_self, db_path: str = '/freqtrade/user_data/hedge_orders.sqlite'):
        self.s = strategy_self  # ← 你要求“就传 self 进来”
        self.db_path = db_path
        # os.makedirs(os.path.dirname(db_path or '.'), exist_ok=True)
        self._init_db()
        self.ex = self._build_exchange_from_original()

    # --- ccxt ---
    def _build_exchange_from_original(self):
        oc = (self.s.config.get('original_config') or {}).get('exchange') or {}
        name = (self.s.config.get('exchange', {}).get('name') or oc.get('name') or '').lower()
        klass = getattr(ccxt, name)
        api_cfg = {
            'apiKey': oc.get('key') or '',
            'secret': oc.get('secret') or '',
            'enableRateLimit': True,
            'options': {'defaultType': 'future', 'defaultMarket': 'future'},
            'timeout': 20000,
        }
        api_cfg = {k: v for k, v in api_cfg.items() if v is not None}
        ex = klass(api_cfg)
        try:
            ex.load_markets()
        except Exception:
            pass
        return ex

    # --- DB ---
    def _conn(self):
        return sqlite3.connect(self.db_path, timeout=30)

    def _init_db(self):
        with self._conn() as con:
            con.execute(
                """
            CREATE TABLE IF NOT EXISTS hedge_orders(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              pair TEXT NOT NULL,
              direction TEXT NOT NULL,      -- long/short
              leverage REAL NOT NULL,
              entry_price REAL,
              remaining_notional REAL NOT NULL, -- 以名义额(USDT)计
              status TEXT NOT NULL,         -- open/closed
              hedge_exit_stage INTEGER NOT NULL DEFAULT 0, -- 0/1/2/3
              created_at REAL NOT NULL,
              updated_at REAL NOT NULL,
              meta_json TEXT                 -- exit_points, stop_loss, tp1, tp2,...
            );
            """
            )
            con.execute('CREATE INDEX IF NOT EXISTS idx_ho_pair ON hedge_orders(pair);')
            con.execute('CREATE INDEX IF NOT EXISTS idx_ho_status ON hedge_orders(status);')
            con.commit()

    # --- helpers ---
    def _to_symbol(self, pair: str) -> str:
        return pair.replace(':USDT', '')

    def _market_info(self, symbol: str):
        return (
            self.ex.market(symbol)
            if getattr(self.ex, 'markets', None) and symbol in self.ex.markets
            else {}
        )

    def _round_step(self, value: float, step: float) -> float:
        if not step or step <= 0:
            return float(value)
        return float(math.floor(value / step) * step)

    def _notional_to_qty(self, symbol: str, notional_with_leverage: float, price: float) -> float:
        raw = float(notional_with_leverage) / max(float(price), 1e-9)
        mk = self._market_info(symbol)
        step = None
        if mk:
            step = (mk.get('limits', {}) or {}).get('amount', {}).get('step')
            if step is None:
                prec = (mk.get('precision', {}) or {}).get('amount')
                if isinstance(prec, int):
                    step = 10 ** (-prec)
        return self._round_step(raw, step or 0.0)

    # --- queries ---
    def has_active(self, pair: str) -> bool:
        with self._conn() as con:
            r = con.execute(
                "SELECT 1 FROM hedge_orders WHERE pair=? AND status='open' LIMIT 1;", (pair,)
            ).fetchone()
        return bool(r)

    def get_active(self, pair: str):
        with self._conn() as con:
            r = con.execute(
                """
            SELECT id, pair, direction, leverage, entry_price, remaining_notional, status, hedge_exit_stage, meta_json
            FROM hedge_orders WHERE pair=? AND status='open' ORDER BY id DESC LIMIT 1;
            """,
                (pair,),
            ).fetchone()
        if not r:
            return None
        oid, p, d, lev, ep, rem, st, stg, mj = r
        return {
            'id': oid,
            'pair': p,
            'direction': d,
            'leverage': float(lev),
            'entry_price': float(ep or 0.0),
            'remaining_notional': float(rem or 0.0),
            'status': st,
            'hedge_exit_stage': int(stg or 0),
            'meta': json.loads(mj or '{}'),
        }

    # --- core actions ---
    def open(
        self,
        pair: str,
        direction: str,
        leverage: float,
        plan_notional: float,
        ref_price: float,
        meta: dict,
    ):
        """
        市价开对冲（非 reduceOnly），写库；remaining_notional=plan_notional
        """
        symbol = self._to_symbol(pair)
        qty = self._notional_to_qty(symbol, plan_notional * leverage, ref_price)
        side = 'buy' if direction == 'long' else 'sell'
        params = {
            'reduceOnly': False,
            'positionSide': 'LONG' if direction == 'long' else 'SHORT',
            'newClientOrderId': f"hg_open_{int(time.time())}",
        }
        order = self.ex.create_order(symbol, 'market', side, qty, None, params)
        with self._conn() as con:
            con.execute(
                """
            INSERT INTO hedge_orders(pair, direction, leverage, entry_price, remaining_notional, status, hedge_exit_stage, created_at, updated_at, meta_json)
            VALUES(?,?,?,?,?,'open',0,?, ?, ?);
            """,
                (
                    pair,
                    direction,
                    float(leverage),
                    float(ref_price),
                    float(plan_notional),
                    time.time(),
                    time.time(),
                    json.dumps(meta or {}, ensure_ascii=False),
                ),
            )
            con.commit()
        return order

    def close_partial(
        self, pair: str, direction: str, leverage: float, percent: float, ref_price: float
    ):
        """
        reduceOnly 市价平部分（按 remaining_notional 的百分比）
        """
        act = self.get_active(pair)
        if not act:
            return
        rem = float(act['remaining_notional'])
        if rem <= 1e-9:
            return
        close_notional = max(0.0, rem * float(percent) / 100.0)
        if close_notional <= 1e-9:
            return

        symbol = self._to_symbol(pair)
        qty = self._notional_to_qty(symbol, close_notional * leverage, ref_price)
        side = 'sell' if direction == 'long' else 'buy'
        params = {
            'reduceOnly': True,
            'positionSide': 'LONG' if direction == 'long' else 'SHORT',
            'newClientOrderId': f"hg_tp_{int(time.time())}",
        }
        self.ex.create_order(symbol, 'market', side, qty, None, params)

        new_rem = max(0.0, rem - close_notional)
        with self._conn() as con:
            con.execute(
                'UPDATE hedge_orders SET remaining_notional=?, updated_at=? WHERE id=?;',
                (new_rem, time.time(), int(act['id'])),
            )
            con.commit()

    def close_all(
        self, pair: str, direction: str, leverage: float, percent: float, ref_price: float
    ):
        """
        reduceOnly 市价全平（忽略 percent，直接 100%）
        """
        act = self.get_active(pair)
        if not act:
            return
        rem = float(act['remaining_notional'])
        if rem <= 1e-9:
            with self._conn() as con:
                con.execute(
                    "UPDATE hedge_orders SET status='closed', updated_at=? WHERE id=?;",
                    (time.time(), int(act['id'])),
                )
                con.commit()
            return

        symbol = self._to_symbol(pair)
        qty = self._notional_to_qty(symbol, rem * leverage, ref_price)
        side = 'sell' if direction == 'long' else 'buy'
        params = {
            'reduceOnly': True,
            'positionSide': 'LONG' if direction == 'long' else 'SHORT',
            'newClientOrderId': f"hg_all_{int(time.time())}",
        }
        self.ex.create_order(symbol, 'market', side, qty, None, params)

        with self._conn() as con:
            con.execute(
                "UPDATE hedge_orders SET remaining_notional=0.0, status='closed', updated_at=? WHERE id=?;",
                (time.time(), int(act['id'])),
            )
            con.commit()

    def set_exit_stage(self, pair: str, to_stage: int):
        act = self.get_active(pair)
        if not act:
            return
        with self._conn() as con:
            con.execute(
                'UPDATE hedge_orders SET hedge_exit_stage=?, updated_at=? WHERE id=?;',
                (int(to_stage), time.time(), int(act['id'])),
            )
            con.commit()
