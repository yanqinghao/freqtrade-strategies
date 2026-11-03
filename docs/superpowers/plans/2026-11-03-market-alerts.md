# Market Alerts Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline, preserving the execution method already used in this session. The user approved implementation; do not add another approval round.

**Goal:** Send closed-15m volume and 4h/1d level-crossing alerts for BTC, ETH and SOL alongside hourly reports.

**Architecture:** Add a public-data-only alert evaluator and persistent per-coin deduplication/cooldown state. Reuse the existing single background monitor worker and atomic JSON helper; the strategy passes a separate alert-state path. Work in `/tmp/freqtrade-market-alerts-20261103` so the running service sees no development edits.

**Tech Stack:** Python, existing pandas/CCXT, stdlib JSON/threading, unittest; official Freqtrade 2026.10 container.

**Spec:** `docs/superpowers/specs/2026-11-03-market-alerts-design.md`

## Global Constraints

- Only BTC/ETH/SOL; advisory messages cannot access plans or submit orders.
- Every minute check closed 15m candles; volume >=2 times the previous 20 bars' mean.
- Levels only from closed 4h/1d candles available by the evaluated 15m candle's start.
- Cross both adjacent closes against the same level; wicks and level refresh alone cannot trigger.
- Merge same-coin/bar events; cooldown 3600 seconds per coin/type across levels/timeframes.
- Persist deduplication; establish a fresh baseline on process start; no historical replay.
- Retain hourly reports/manual `/monitor`; no new dependencies; no live test orders.

## Review Focus

- A new 4h/1d candle closes with the event: do not use it to create the event's reference level.
- Missing or stale 15m candles: do not compare nonadjacent closes or replay a backlog.
- Zero/nonfinite/negative volume: skip volume alerts without inventing ratios.
- Persistence failure: do not emit an alert whose cooldown cannot be recorded.
- Slow/failed requests: never block the trading loop; isolate failures per coin.

## Task 1: Detection and durable alert policy

**Files:** create `user_data/strategies/market_alerts.py`, `tests/test_market_alerts.py`; reuse `market_monitor._closed_frame`, `analyze_candles`, and `assistant_state.update_state`.

**Interfaces:** `MarketAlerts(state_file).check(fetch, now) -> list[str]`; `fetch(pair,timeframe)` returns public OHLCV rows. Store only last candle id and cooldowns in a dedicated JSON file.

- [x] Write failing tests for baseline, 2x volume excluding current bar, merged up/down crosses, wick exclusion, reference cutoff and level refresh, cooldown across new levels, restart/gap recovery, invalid data/volume and failure isolation.
- [x] Run `python -m unittest discover -s /workspace/tests -p test_market_alerts.py -v` inside offline 2026.10 image; expect missing feature assertion failure.
- [x] Implement the evaluator and persistence; use the existing closed-candle helper with an optional minimum of 21 for alert inputs. Preserve default 50 for hourly analyses.
- [x] Run the same command; expect all alert tests pass.

## Task 2: Scheduling, regression and rollout

**Files:** modify `market_monitor.py`, `KamaFama_Dynamic.py`, `tests/test_market_monitor.py`, `tests/test_manual_strategy.py`, `README.md`.

**Interfaces:** `HourlyMarketMonitor(..., alerts=None)` keeps hourly-only behavior for existing callers; production strategy imports `MarketAlerts` while the native temporary strategy path exists and supplies an instance using a dedicated sibling `market_alert_state.json`. One worker cycle runs due hourly reports and minute alert checks, sharing per-cycle fetched rows. Manual reports do not mutate alert state.

- [x] Add failing tests proving minute scheduling, same-hour hourly reports do not repeat, pending jobs do not block ticks, and strategy enables a separate alert-state file.
- [x] Implement scheduling and wiring; document thresholds, startup baseline, cooldown and best-effort notifications.
- [x] Run all unittest tests with `--network none`; expect green suite. Run Black/Flake8 and `git diff --check`.
- [x] Request one fresh read-only whole-change review; address important findings with failing regressions before fixes.
- [ ] Commit reviewed code, fast-forward original checkout and push under prior authorization. Check current assistant positions/plans, build/restart service, inspect redacted logs and baseline state. Do not submit orders.

## Progress

Plan and spec reviewed against each other. Existing session authorizes inline work, commits, push and startup. Development worktree uses the existing pinned Docker image for tests; no dependency installation needed.
