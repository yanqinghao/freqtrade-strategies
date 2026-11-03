# Execution Reliability Implementation Plan

> Use superpowers:executing-plans inline, preserving this session's execution method and existing authorization for commit, push and deployment. The user approved the design and said continue; do not add another approval gate.

**Goal:** Independently detect stalled trading/connectivity, and reject entries when exchange/local positions disagree.

**Architecture:** A small entry guard uses official fresh fetch_positions before plan mutation. A read-only heartbeat publisher exports native last_process/state; a separate stdlib watchdog consumes this snapshot and public connectivity, with separate TG-only credentials/state. Core/RPC and exit behavior remain unchanged.

**Tech Stack:** Python stdlib, current Freqtrade 2026.10/CCXT, existing atomic JSON helper, Compose, unittest.

**Spec:** docs/superpowers/specs/2026-11-03-execution-reliability-design.md

## Global Constraints

- Develop in the managed isolated worktree, not live-mounted user_data.
- Watchdog: 30-second polling, 180-second startup grace and stale threshold, two consecutive public-probe failures; RUNNING/PAUSED require fresh completed loops. STOPPED sends one status notice.
- Alert only on transitions, persist pending notifications, coalesce outages, best-effort TG delivery. No auto restart or trading.
- Watchdog sees only its TG-only config, read-only health directory and own state; no exchange secrets, Docker socket, trade database or manual plans.
- Entry guard: fresh official fetch_positions(pair), contract-to-base conversion once, finite nonnegative contracts, valid single direction, strict quantity match apart from floating noise. Fail closed before plan mutation/order submission. Dry-run skips real account access. Exit paths never call guard.
- Keep current monitoring, orders, positions and plans intact. Never submit validation trades or cancel existing orders.

## Review Focus

- Publisher alive while native loop is stuck must still alert; test stale native last_process with fresh observed_at.
- Missing/malformed/future health timestamps and repeated bot restarts cannot indefinitely reset grace.
- TG send timeout/failure cannot mark notification delivered; restart retains pending state and recovery coalesces it.
- Position response wrong units, malformed fields, duplicate/hedged sides or missing local position must block, without plan changes.
- The native temporary strategy import path disappears before callbacks run; all new imports must work in a fresh-process startup test.

## Task 1: Entry guard

Files: create user_data/strategies/entry_guard.py, tests/test_entry_guard.py; modify strategy and telegram.py plus tests/test_telegram_orders.py.
Interface: ensure_position_matches(exchange, pair, trade, dry_run=False) raises ValueError on unsafe/unknown account state; strategy.ensure_entry_position(pair, trade) derives mode and delegates. Telegram calls this after existing local guards, before reading/mutating a bound plan.

- [x] Write failing tests for new flat entry, untracked position, direction/quantity mismatch, externally closed position, contract size, lot-size mismatch, malformed/duplicate data, request failure sanitization and dry-run skip; prove mismatch leaves plan and order calls untouched.
- [x] Run scoped offline tests and inspect expected missing-feature failures.
- [x] Implement guard and wiring with eager strategy-level imports.
- [x] Run full offline unittest suite; expect green. Commit task.

## Task 2: Independent health pipeline

Files: create user_data/strategies/execution_health.py, tools/health_watchdog.py, tools/prepare_watchdog.py and tests/test_execution_health.py, tests/test_health_watchdog.py; modify strategy and docker-compose.yml, README.md.
Interfaces: HeartbeatPublisher(bot,path).publish(now), start(), close() writes observed_at/state/last_process; Watchdog(state_file).check(snapshot,probes,now,send) persists current/notified/pending state. Public probes and TG transport use urllib with finite timeouts. prepare_watchdog.py extracts only token/chat_id/topic_id into ignored deploy/watchdog.json mode0600.

- [x] Write failing tests for stale native cycle despite fresh publisher, startup grace/missing/invalid snapshot, STOPPED/PAUSED, probe debounce, dedup/recovery, pending TG retry/restart/coalescing, safe transport errors and minimal config extraction.
- [x] Run scoped offline tests; inspect expected failures.
- [x] Implement publisher, watchdog loop/transport, least-privilege Compose mounts and preparation utility; keep callbacks free from late import-path dependencies.
- [x] Run all tests plus a fresh-process native import-path test, formatting/lint and Compose validation. Commit task.

## Task 3: Review and rollout

Consumes the two components above; deployment uses the unchanged live database/manual plan and a separate health directory.

- [ ] Request one fresh whole-change read-only review. Fix important findings using RED→GREEN regressions, then green full suite.
- [ ] Fast-forward main, test merged source, push; create TG-only config locally without printing credentials. Backup/recheck live trades/orders/plans and verify Binance connectivity.
- [ ] Gracefully update main service and start watchdog; verify native RUNNING heartbeat, fresh health snapshot, watchdog healthy status, monitoring and preservation of existing orders/plans. No injected live faults or test orders.
- [ ] Record evidence, push completion docs and retain host-managed worktree per platform ownership.

## Progress

Baseline: 83 tests green in /tmp/execution-reliability-baseline.log. No dependency install required; use pinned deployment image with network disabled for tests.
