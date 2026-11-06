# Extreme Volume Notifications Implementation Plan

> Execute inline using superpowers:executing-plans; no parallel implementation.

**Goal:** Notify the configured Telegram chat when a current 5m candle accumulates extreme volume.
**Architecture:** Public market worker detects events, existing SQLite queue atomically deduplicates and retries delivery.
**Tech Stack:** Existing Python/CCXT/SQLite/Freqtrade 2026.10; no new dependencies.
**Spec:** docs/superpowers/specs/2026-11-06-extreme-volume-alerts-design.md

## Tasks

- [x] Write failing detector, queue and scheduling/integration tests.
- [x] Implement volume detector and bounded durable notification delivery.
- [x] Wire independent 10-second checks; retain minute/hour report cadence.
- [x] Update Telegram help, run focused/full offline tests and lint.
- [ ] Review, commit, push and deploy; verify service and unchanged plans.

## Review focus

Current cumulative volume must not be prorated. Exclude current bar from baseline.
Do not replay stale historical candles after restart. Queue and dedup must commit together.
Market notifications cannot expire or delay critical position notices. No order mutations.

Independent review identified shared-worker starvation and stale scheduled timestamps.
Added failing regressions, then isolated the volume worker/client and refreshed time after
each public fetch. Queue atomicity, restart dedup and critical priority were confirmed.

Final offline Freqtrade 2026.10 suite: 299 tests passed in 49.669s. All pre-commit
checks passed. Independent follow-up review found no remaining important issues.
One-minute historical snapshots identify BTC's Nov 4 spike by 22:43 Beijing
(13.90x cumulative base volume), before the 22:45 candle close; these are
historical detection bounds, not claims of real Telegram delivery.
