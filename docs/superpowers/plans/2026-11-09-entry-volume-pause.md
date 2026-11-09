# Entry Volume Pause Implementation Plan

> Execute inline using superpowers:executing-plans; updated user constraints override the withdrawn permanent-cancellation plan.

**Goal:** Pause unsafe initial limit entries, preserve intent, and resume only on safe original-price conditions.
**Architecture:** Reuse public monitor worker pattern, existing request ledger, receipt lifecycle, and Telegram submission path. No trading from worker threads.
**Tech Stack:** Existing Python/pandas/SQLite/Freqtrade2026.10/unittest; no dependencies added.
**Spec:** docs/superpowers/specs/2026-11-09-entry-volume-pause.md

**Research reconciliation:** Implemented locally after user approval; not a live deployment record. Original
manual points only;4x adverse actual forming5m volume,quiet15min,0.25R proximity,
completed1m reclaim if crossed,then restore ORIGINAL LIMIT. No market restoration,
autonomous chasing,reversal,median-template SL/TP or global TP halving.
0.25R is a conservative candidate,not a measured optimum;0.5R had the same
limit-restoration result on this short sample. Preserve prior raw reports.

## Tasks

- [x] Complete pure pause evaluation:0.25R proximity,post-cross reclaim,last qualifying observation persistence and forming-volume causality.
- [x] Add durable pause states and query/cancel support in the existing request ledger.
- [x] Integrate receipt-safe withdrawal and serialized idempotent resubmission.
- [x] Cover final SDK guard, partial fills, unknowns, restart, TTL, market-data failures and pause controls.
- [x] Run full offline suite and fresh review; document deployment state.

## Review focus

No hindsight volume or first-minute pre-creation extremes; cancelled plans never silently revive;
no replacement before terminal zero fill; no duplicate resumes after crash; never block exits.

## Ledger

- Original baseline:353 tests passed. Withdrawn permanent-cancel implementation was not deployed.
- Data replay compares original actual requests only. Historical25.59U deletion difference does not apply to this resumed-entry policy.
- Latest39-request known-intervention sensitivity:original limit-25.72U;original-limit recovery-11.45U. Reduction14.27U,still losing;incomplete edit history and immediate-cancel assumptions prevent live-equivalence claims.
- At plan start runtime had volume alerts only. The implementation now integrates receipt/persistence wiring locally; no live restart performed.

## Implementation ledger (2026-11-09)

- User approved original-limit restoration and the 0.25R explanation; implementation proceeds inline.
- Ruling: isolated checkout /tmp/entry-volume-pause-20261109 from local HEAD; native managed worktree is outside writable roots. No production data/config copied; .git in original checkout is untouched.
- Pre-flight: policy → controller consumes decision/history; ledger → controller persists intent before cancel/submit; controller → Telegram reuses original submission and final SDK guard; monitor only fetches public data.
- Task 1: 12 pure policy tests green after four newly failing quote/range tests. Current quote crossing cannot use a prior closed candle.
- Task 2: two persistence tests failed before implementation; separate table in existing request SQLite preserves terminal decisions and distinct attempt references.
- Initial baseline attempt used incompatible local Freqtrade2024.12.1 and was interrupted. All final validation uses the pinned2026.10 image, network disabled.

- Tasks 1–5 complete: policy, durable plan/attempt state, public worker, serialized withdrawal/recovery, Telegram query/cancel/replace integration and final SDK guard.
- Native destructive accounting is only called after validated terminal receipts; positive fills stay in existing position management.
- Fresh whole-change reviewer found four material issues; all received failing regressions, fixes and targeted reviewer confirmation. Additional observed-signal retention regressions verify forming-candle reversals cannot erase prior evidence.
- Final full suite:392 tests passed in17.706s; `git diff --check` clean. Container used no network or production databases.
- Ruling: retain reviewed changes as local project edits, without committing/merging/pushing or restarting the trading process. User approved implementation; runtime activation is a separate operational action.
