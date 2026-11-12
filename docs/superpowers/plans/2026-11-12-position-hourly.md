# Hourly Position Reports Implementation Plan

> Use superpowers:executing-plans inline, then one fresh whole-change review.

**Goal:** Hourly advisory reports for every actual assistant position.
**Architecture:** New public-only position worker reuses existing candle analytics;
the trading thread composes native remaining-position PnL and plan information,
then uses the persistent advisory queue. No trading actions from the reporter.
**Tech Stack:** Existing Python/Freqtrade 2026.10/pandas/SQLite/unittest.
**Spec:** docs/superpowers/specs/2026-11-12-position-hourly-design.md

## Global Constraints

All held symbols, hourly, completed candles only, 30s quote freshness, original
plans unchanged, five-minute report expiry, critical alerts retain priority.

## Review Focus

- Empty/pending/paused entries must not appear as positions; observed unaccounted
  fills must not appear as zero PnL or safe directional advice.
- A trade closed/reopened while public I/O runs must not receive its old report.
- A slow/erroring symbol must not stop other reports or block the trading loop.
- Current quote/closed-candle/hour boundaries and missing short-history symbols
  must not produce stale PnL, invented levels or unsupported recommendations.
- Reports must not overwrite plans, advance TP stages, duplicate exit actions,
  outrank urgent notifications or duplicate after restart in the same hour.

### Task 1: Report and public worker

Files: new `user_data/strategies/position_monitor.py`, new tests counterpart.
Interface: `PositionMonitor.tick(now, strategy, trades)` consumes current ORM
records only on the caller/trading thread; background collect gets frozen
`(trade_id, pair)` tuples and returns public analysis/quote records. Reuse native
`position_profit` and read-only `saved_plan`/`effective_stop` during composition.

- [x] Write failing tests for positions, both sides of PnL/advice, incomplete data,
  causal completed candles, empty accounts and nonblocking hourly scheduling.
- [x] Implement report formatting/priority and isolated public collection.
- [x] Verify close/reopen filtering, unchanged plans/TP/accounting and deduplication.

### Task 2: Lifecycle, delivery and validation

Files: strategy startup/loop/cleanup, AlertQueue priority, existing lifecycle and
alert tests, README, startup regression tests.

- [x] Add failing lifecycle/delivery priority tests, then wire startup/loop/cleanup
  and lower advisory priority without changing critical notification behavior.
- [x] Run cached project hooks before final pinned offline full suite; obtain
  fresh review and resolve material findings with failing regressions.
- [ ] Prepare candidate source/image and return verified code; record release state.

## Ledger

- Baseline is pushed/deployed c2cd416, whose source passed 428 offline tests.
- User selected hourly from the proposed frequencies and requested operational
  recommendations; use stated deterministic conditional rules without an AI API.
- Ruling: existing helpers cover PnL and market data; no new dependencies or
  trading knobs. Preserve the report's advisory-only scope.
- Isolation: `/tmp/position-hourly-f9y37o3u/repo`, source-only local clone; no
  production database/config copied. Continue development without another design
  confirmation round because the user explicitly asked for this addition.
- RED: module initially absent; main strategy loop did not enqueue position
  reports; an older advisory incorrectly outranked a newer critical stop alert.
- GREEN: all 12 new report/scheduling tests passed in 6.697s after integration,
  including native fee-aware long/short PnL literals, accounting uncertainty,
  stale/incomplete data, unchanged plan/state, nonblocking I/O and hourly dedup.
- Ruling: a current quote beyond the relevant 4h reference level is labeled an
  intrabar condition and withholds a trend-following hold recommendation. A
  completed-candle trend label alone must not hide a new intrabar level breach.
- Self-review regressions first exposed a misleading "unfilled" protection label
  for observed fills pending native accounting and reuse of a pre-enqueue clock
  across reports. Both now use observed quantity and current per-report time.
- First full suite passed 442 tests in 72.052s. Fresh review found one P2:
  original TP3 incorrectly triggered advice after extension activation. New
  long/short regressions first failed, then use of the strategy's read-only
  `_targets` fixes active-target guidance while retaining original display.
- Final targeted position tests: 14 passed in the pinned offline image.
- Final full candidate: 443 offline tests passed in 77.848s; project hooks,
  Black/flake8 and `git diff --check` pass. Fresh re-review confirmed the P2 fixed
  and found no remaining material issue.
- Candidate tag: `freqtrade-strategies-freqtrade:hourly-positions-20261112`.
  The Docker image remains the existing base; strategy functionality is in the
  validated user_data source bind, so rollback must restore source as well.

## Release

- Follow the commit/push/activate workflow already authorized in this conversation;
  the new request explicitly asks for scheduled notifications. Announced the
  release sequence while developing. No automatic trading actions are added.
- Remote main and the live source base match c2cd416. Private online database,
  plan/health and rollback source snapshots are under ignored
  `.assistant-archive/hourly-positions-20261112T045420Z`.
- Pre-release records include actual SKYAI position and an unfilled BTC entry;
  only SKYAI is eligible for a held-position report at that snapshot.
