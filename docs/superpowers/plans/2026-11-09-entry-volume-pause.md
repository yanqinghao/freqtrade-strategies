# Entry Volume Pause Implementation Plan

## Follow-up: short original-limit restoration (2026-11-09)

User approved removing the short proximity gate: after 900 quiet seconds, restore
the original sell limit even when price is far below entry. A prior cross still
requires a subsequent completed 1m close and current quote at or below entry.
Long proximity, freshness/continuity, hard SL, TTL and receipt checks stay intact.

- [x] Reproduce 2200 short / 2290 SL / 2130 market with failing pure-policy tests,
  including the 899/900-second boundary and a new cross requiring a later close.
- [x] Restrict the existing range gate to longs and update request status wording.
- [x] Run the pinned Freqtrade offline suite and review the diff: 395 tests passed
  in 53.352s, including 16 policy tests; `git diff --check` clean. Both new
  recovery regressions first failed with `waiting_range`, then passed.
- Candidate image built as `freqtrade-strategies-freqtrade:short-resume-20261109`
  (`sha256:93cb10e9b207e39468bb57c0ae9288931db194bbf07f34ef80d78c6c05a94df3`).
- User separately approved deployment. Candidate-image offline verification passed
  all 395 tests in 56.542s. Commit `59ddf30` was pushed to origin/main before restart.
- Container started at 2026-11-09 12:51:35 UTC and entered RUNNING at 12:51:46 UTC;
  the running image and Telegram/policy source hashes match the candidate.
- ETH plan `91828316cd108d39e5fe671f` moved from `waiting_range` to
  `waiting_continuity` at 20:51:52 Beijing, attempts=0. The restart requires a
  fresh 15-minute continuous observation period; no restored ETH order was
  claimed at verification time. Original deadline and BTC order `1189351748904`
  remain unchanged.
- Rollback image: `freqtrade-strategies-freqtrade:rollback-before-short-resume-20261109`.
  Private snapshots and prior source are under ignored
  `.assistant-archive/short-resume-20261109T124935Z`; never restore an old trading
  database over subsequent live activity.

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

## Authorized deployment follow-up

- User explicitly authorized deployment, then required commit and push first. Commit45e645b was pushed to origin/main before any production restart. SSH22 failed; GitHub SSH443 succeeded without changing remote configuration or host-key checking.
- Preserved the previous image and private database/plan snapshots. First candidate passed392 tests but failed real startup because StrategyResolver removes its temporary import path before Telegram construction. Rolled back to the prior image and verified RUNNING heartbeats; never restored an old trading database.
- Startup fix imports pause modules while the strategy resolver path is active and lets Telegram attach execution through the strategy. A fresh-process native-resolver regression reproduced the original ModuleNotFoundError before the fix;393 full offline tests now pass. The fix will be committed/pushed before redeployment.

- Deployment completed after fix59e97ce was pushed. Final candidate image `sha256:e9173a1948fb2c4b9f3ece78e26e77c5eb2bd5225b0ff32ba24632ca6b88fc81` passed393 offline tests in22.044s.
- Final container started2026-11-09 07:38:38 UTC (15:38:38 Beijing), RUNNING at07:38:48 UTC; restart count0, no ERROR/CRITICAL startup logs. Receipt and SDK guards installed.
- BTC/ETH original pending intents adopted, price-history check timestamps advancing; original two open order IDs retained. Runtime Telegram/order-management/strategy source hashes match the committed project. Fresh heartbeat and trading-loop progress confirmed.
- Prior image retained as `rollback-before-entry-volume-45e645b`; private snapshots under ignored `.assistant-archive/entry-volume-45e645b-20261109`. Never restore stale trade databases over subsequent live fills.
