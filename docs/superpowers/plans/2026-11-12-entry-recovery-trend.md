# Entry Recovery Trend Implementation Plan

> Use superpowers:executing-plans inline, then a fresh whole-change reviewer.

**Goal:** Allow bounded immediate-fill recovery with a 15m opposite-trend veto.
**Architecture:** Extend the pure policy, enrich public worker snapshots, and
reuse the trading-thread controller/final SDK guard. Preserve existing limits.
**Tech Stack:** Python, existing pandas/SQLite/Freqtrade 2026.10/unittest.
**Spec:** docs/superpowers/specs/2026-11-12-entry-recovery-trend-design.md

## Global Constraints

No dependencies or production restart; original limit/SL/TP/deadline persist.
900s quiet/continuity, 30s freshness, 0.25R region, latest completed 15m only.

## Review Focus

- Exact E may immediately fill and must pass the trend gate.
- Crossing beyond the region cannot reuse a pre-cross close, and being inside
  the region must not continually reset an entry-cross timer and block forever.
- Missing trend blocks immediate fills without blocking resting recovery,
  known volume shocks, exits, or cancellation.
- Final refreshed quote/trend must be checked after account/leverage setup.
- Trend candle gaps, malformed candles, stale samples and new 15m boundaries
  must not authorize immediate entry; forming candles must not change direction.

### Task 1: Policy and trend observations

Files: `entry_volume.py`, `entry_pause_monitor.py`, their existing tests.
Interface: `advance_pause(..., quote=None, trend=None)` consumes a trend record
`{direction, observed_at, closed_at}`. `analyze_recovery_trend(rows, observed_at,
now)` validates public 15m rows and returns that record. Snapshot key `trend`.

- [x] Add failing boundary, reclaim, both-direction veto and data-validity tests.
- [x] Run tests and observe missing feature failures before implementation.
- [x] Implement the two recovery branches and public trend snapshot enrichment.
- [x] Verify policy/worker tests under the pinned offline Docker image.

### Task 2: Trading integration, wording and release artifact

Files: `entry_pause.py`, `telegram.py`, README, existing pause integration tests.
Consume snapshot `trend` through the existing controller and SDK guard.

- [x] Add failing original-price/deadline submission and final-guard regressions.
- [x] Wire the snapshot, add clear wait reasons and document final conditions.
- [x] Run the full offline suite; fresh reviewer examines the complete diff.
- [x] Resolve material findings with RED/GREEN verification, build a candidate
  image and return reviewed changes to the original workspace without restart.

## Ledger

- Baseline: inherited symmetric fix passed 409 pinned-image tests this session.
- Ruling: use an isolated local clone in /tmp because native worktree placement
  is outside writable roots and the primary checkout is mounted into production.
  Only tracked source plus this session's existing diff was copied.
- Ruling: user already approved the proposed behavior and asked to implement;
  proceed inline without an additional design/plan approval round.
- Ruling: one completed 1m close in the allowed region is required for immediate
  recovery; original-price reclaim stays on the resting branch. Separate region
  breach timestamps retain causal post-breach confirmation.
- Ruling: no commits/pushes requested; retain reviewable local changes.
- Task 1: 12 new policy/classifier tests and 9 public-worker tests passed in the
  pinned offline image. Before implementation, policy expectations failed on
  old reclaim/ready outcomes and the worker lacked 15m sampling.
- Task 2: RED confirmed the controller ignored trend snapshots; wiring makes
  bounded recovery reach the original submission and rejects changed final
  quote/trend while reduce-only exits pass. An additional RED exposed stale
  region-cross metadata after stop edits; clear it with the existing version
  reset. Six old pure-policy expectations intentionally change from original
  entry reclaim to missing-trend waits inside the new immediate-fill region;
  their subsequent original-entry reclaim checks stay in place.
- Fresh review identified two material final-guard issues: configurable target
  pricing could hide an immediately executable quote, and the validation clock
  was captured before I/O. Added RED regressions for misleading target price,
  slow I/O, minute/15m boundary and expiry, then switched to validated actual
  best book prices and post-I/O time/RUNNING checks. All 7 targeted integration
  tests pass; reviewer confirmed both resolved with no new material findings.
- Candidate image built:
  `freqtrade-strategies-freqtrade:entry-recovery-trend-20261112`,
  `sha256:55078fc1b69dae203a366216d4804d02d43a21feb9cff912b4be4c207c4348da`.
- Final candidate passed all 425 offline tests in 69.848 seconds with network
  disabled and source read-only. `git diff --check` clean. Review findings are
  resolved; no remaining material findings. No replay/profit improvement claim.
- Returned all 14 changed/new source, test and documentation files to the main
  workspace and verified each against the tested isolated copy by SHA-256.
  Production databases, configuration, running container, and live plans were
  not modified; no commit or push. Candidate remains undeployed.

## Follow-up: all manually ordered symbols

User scope clarification: entry protection follows all assistant-tracked initial
limit symbols, rather than a fixed BTC/ETH list. Continue in the existing isolated
clone carrying the reviewed 425-test version; no new production action authorized.

- [x] Add failing non-BTC/ETH initial shock/cold-data and existing-order adoption
  regressions, plus dynamic worker add/remove/oldest-history tests.
- [x] Remove fixed controller eligibility and schedule every active plan symbol;
  reuse MONITOR_PAIRS solely for inexpensive idle volume prewarming.
- [x] Verify the targeted tests, update user-facing coverage documentation, run
  the full pinned offline suite and obtain fresh review of this follow-up.
- [x] Build a distinct candidate, return verified source to the workspace without
  commit/push, real cancellation or production restart.

- RED: SOL/XRP/DOGE initial requests bypassed the pause ledger; an unseen ADA
  request was not retained; an existing XRP initial limit was not adopted;
  dynamic extra symbols were absent from the worker's needs. Seven assertion
  failures observed before implementation (including SOL prewarm coverage).
- GREEN: all 37 pause/controller/worker/startup/store tests passed in 18.873s.
  Fixed-list eligibility was removed, while ownership, partial-fill handling,
  original TTL and quiet/continuity rules remain unchanged.
- Focused fresh review of the follow-up found no material issues. The complete
  reviewed diff is `/tmp/entry-recovery-trend-yhpqhafa/all-symbols.diff`.
- Candidate tag `freqtrade-strategies-freqtrade:all-symbol-recovery-20261112`;
  strategy modules continue to use the project's existing user_data bind mount.
- Full candidate suite: 428 tests passed in 75.661s, offline/network disabled,
  source read-only. `git diff --check` clean; no unresolved review findings.
- Copied 8 follow-up files to the project; all 14 candidate source/test/doc files
  match the isolated validated copy. No production restart, real order changes,
  commit or push; all-symbol protection remains pending deployment.

## Authorized deployment (2026-11-12)

- User explicitly authorized commit, push and deployment, in that order.
- Release preflight on the actual project source and final candidate passed all
  428 offline tests in 60.340s; remote main matches base b05d98e.
- Saved online SQLite backups, original plan/health snapshots and rollback source
  under ignored `.assistant-archive/all-symbol-recovery-20261112T033315Z`.
  Rollback image: `freqtrade-strategies-freqtrade:rollback-before-all-symbol-20261112T033315Z`.
  Never restore an old trading database over subsequent live activity.
- Before deployment, bot heartbeats were RUNNING. An unfilled SKYAI initial limit
  and the original paused BTC intent will be checked/adopted by the new policy.
- GitHub SSH uses port443 with the existing trusted github.com host-key alias;
  strict host-key verification stays enabled and remote config is unchanged.
- Production replacement remains gated on successful commit and push.
- Project pre-commit Black reformatted six Python files. AST equivalence with
  the reviewed version was verified for all six, then all 428 release tests
  passed again in 61.847s. No hooks were skipped.
- Implementation commit `706cf1d` was pushed to origin/main before cutover.
  Candidate image `sha256:55078fc1b69dae203a366216d4804d02d43a21feb9cff912b4be4c207c4348da`
  started at 2026-11-12 03:38:44 UTC (11:38:44 Beijing), restart count 0.
- RUNNING heartbeats and advancing native trading-loop timestamps verified;
  no ERROR/CRITICAL startup messages. Risk guard installed, five runtime source
  hashes match the pushed release. Watchdog was not recreated.
- BTC original intent `41299c66f0abfda4bc95d8f9` moved from `waiting_range` to
  `waiting_continuity`; original E=73100 and original deadline remain intact,
  attempts=0. No successful restoration claimed before the fresh observation
  period completes.
- Existing SKYAI entry order `2796285157` was adopted as plan
  `77cb8aaf9ddb3ed06b9df1c1`. Recent adverse volume triggered withdrawal; exchange
  receipt is canceled with filled=0, original amount=2770. Native zero-fill
  cleanup completed, preserving E=0.0361 and original deadline in the pause
  ledger. Plan now waits for continuity, attempts=0. No duplicate entry submitted.
- No trading database was restored; rollback source/image and private state
  snapshots remain available in the recorded ignored archive.
