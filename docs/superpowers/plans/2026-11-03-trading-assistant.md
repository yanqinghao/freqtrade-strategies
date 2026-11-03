# Trading Assistant Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans. Work in the existing directory as requested; do not commit unrelated changes.

**Goal:** Make this repository a Telegram controlled trading assistant with three hourly higher-timeframe market reports.

**Architecture:** Preserve the Freqtrade runtime and strategy name. Replace signal generation with explicit manual plans, isolate advisory market reporting from order execution, and remove retired AI and research entrypoints.

**Tech Stack:** Existing Python/Freqtrade, pandas, standard-library unittest and SQLite/JSON facilities.

**Spec:** `docs/superpowers/specs/2026-11-03-trading-assistant-design.md`

## Global Constraints

- Modify the current repository; preserve unrelated pre-existing edits. No commits or live restarts.
- No automated entries, direction selection, hedge orders or AI calls.
- Only BTC/ETH/SOL reports; only closed daily and four-hour candles for levels; 15m/1h/4h/1d for trend; hourly reports are separate messages.
- Keep manual exits, stop losses and partial take profit. Reporting never changes trade plans.

## Review Focus

- Partial/rejected exits must not advance the take-profit stage before a fill.
- A market report failure for one coin must not prevent the other reports or block trading.
- Restart and concurrent Telegram edits must preserve manual plans.
- Stop loss must take priority over pending entry/adjustment orders.
- Removed commands or custom keyboard settings must not prevent Telegram startup.

### Task 1: Advisory monitoring

Files: create `user_data/strategies/market_monitor.py`, tests for closed-candle daily/4h analysis and hourly scheduling.

- [x] Write and run failing tests for closed candles, BTC/ETH/SOL reports, and duplicate-hour suppression.
- [x] Implement deterministic higher-timeframe reports and a background runner using the existing exchange/data provider.
- [x] Run tests without network calls or Telegram sends.

### Task 2: Manual execution strategy

Files: simplify `user_data/strategies/KamaFama_Dynamic.py`; add shared atomic state helpers if needed; tests of manual-only signals, TP fills, long/short stop prices and failure recovery.

- [x] Write and run failing behavior tests against the actual strategy class.
- [x] Remove indicator, auto-direction, auto-entry, AI and hedge paths. Integrate monitor via bot callbacks.
- [x] Preserve manual configuration compatibility and move state changes to confirmed order events.
- [x] Run the strategy and monitoring tests.

### Task 3: Telegram and retired jobs

Files: `telegram.py`, `Dockerfile`, `docker-compose.yml`, config files, retired AI/research tools and README.

- [x] Remove retired commands, imports, dependencies, schedules and build copies; retain manual operations.
- [x] Connect `/monitor` to three advisory reports. Update manual command help and obsolete keyboard entries.
- [x] Clean configuration without printing or changing credentials; do not touch live trade records.
- [x] Document commands, monitor semantics and deployment checks; inspect host schedule access without running jobs.

### Task 4: Integration verification

- [x] Run the full new suite, compile touched modules and validate Telegram registration against the installed runtime.
- [x] Review the complete diff with a fresh reviewer and fix correctness findings.
- [x] Record evidence and any deployment limitations; do not start or restart production services.
