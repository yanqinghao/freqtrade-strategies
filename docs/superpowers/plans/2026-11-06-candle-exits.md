# Candle Exits Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add optional closed-candle soft exits and conditional TP3 extension with TP2 protection.

**Architecture:** One small `candle_exits.py` module validates and formats rules and selects fresh closed candles. The strategy subscribes to native informative data, persists trigger state in existing trade custom data, and uses existing full-exit and TP machinery. Telegram and risk preview share validation.

**Tech Stack:** Python, pinned Freqtrade 2026.10, pandas, unittest.

**Spec:** `docs/superpowers/specs/2026-11-06-candle-exits-design.md`

## Global Constraints

No new dependencies or live trading changes. Hard SL remains mandatory. Preserve old plans, TP proportions, partial-fill accounting, durable exit retries and protective monotonicity. Supported timeframes: 15m, 1h, 4h, 1d.

## Review Focus

- Rule edits or restart must not apply an old candle or reverse triggered state.
- A final TP order already started must not become an extension.
- Malformed/stale data must not disable old hard protection.
- New parameters must survive request history/preview and validate inherited plans.
- Pending orders and add-ons must not reset activated protection.

### Task 1: Rule interface and validation

Files: `user_data/strategies/candle_exits.py`, `telegram.py`, `user_data/strategies/entry_risk.py`, `tests/test_candle_exits.py`.

- [x] Write failing parser, merged-plan validation, candle boundary and command round-trip tests.
- [x] Run `.venv/bin/python -m unittest discover -s tests -p test_candle_exits.py`, observe missing feature failures.
- [x] Implement parsing, shared `validate_rules(plan)` and `closed_candle(frame, timeframe, now, after)` plus display helper. Validate before plan/order mutation, including previews.
- [x] Run focused tests; expected PASS.

### Task 2: Persistent exit execution

Files: `user_data/strategies/KamaFama_Dynamic.py`, `tests/test_candle_exits.py`, `user_data/strategies/entry_risk.py`.

- [x] Write failing integration tests using existing strategy fixture for both directions, TP2 completion, original TP3 precedence, soft SL/full-exit priority, persistence, edits and missing data.
- [x] Implement informative subscriptions and rule evaluation; store enabled times and activation on trade data. Reuse full-exit intents, protect TP2 atomically with extension, and preserve final TP tagging.
- [x] Verify focused tests plus existing strategy/order/risk tests pass.

### Task 3: Visibility and regression verification

Files: `telegram.py`, `user_data/strategies/trade_overview.py`, `README.md`, focused integration tests.

- [x] Test manual/overview and preview/history visibility, pending-order/edit and add-on restrictions.
- [x] Update help, examples and operational limitations, including hard SL estimates and soft-rule activation status.
- [x] Run full unittest suite, formatting/checks and `git diff --check`.
- [x] Fresh independent review, address material findings with regression tests, record evidence below.

## Progress and rulings

- User approved the conversational design and authorized implementation; continue inline without another approval round.
- Git metadata is read-only; worktree creation failed. Follow the worktree skill fallback: edit the existing workspace, leave changes uncommitted for review.
- Keep the existing hard-SL requirement because soft-only mode was not chosen. Freeze activated extension parameters and reject add-ons while active to avoid accidentally resetting the confirmed state.


## Verification and review

- Tasks 1–3 complete. First feature tests: 14 tests, missing-feature failures observed before implementation; expanded to 26 focused tests including partial TP2, locked state, pending orders, historical TP3, restart, command validation and rollback.
- Independent read-only review found one Important issue: pending-order creation time was not a valid first-position baseline. Added failing `test_pending_entry_first_fill_cannot_use_pre_position_close` and `test_legacy_position_uses_real_fill_date_or_safe_first_observation`, then persisted the first-fill baseline and recovered legacy values from native `date_entry_fill_utc`. Add-ons keep the original baseline. Both regressions now pass.
- Ruling: when legacy data has no first-fill time, start from first observation rather than infer an earlier fill; cost is waiting for the next close, while hard SL/TP remain effective.
- Final full suite: official local `freqtradeorg/freqtrade:2026.10` image, network disabled, read-only code/test mounts, no production config/credentials, `python -m unittest discover -s tests`: **286 tests passed in 14.588s**. Log: `/tmp/freqtrade-candle-docker-complete.log`.
- Local `.venv` is Freqtrade 2024.12.1, not the pinned project version; its pre-existing failures and sandbox asyncio shutdown hang are not used as the completion gate. No environment/dependency or production changes made.
- Python compilation and `git diff --check` passed. Changed/new Python code formatted with cached Black; unchanged functions retain their original formatting.
- No remaining review findings. Initial implementation was handed off uncommitted with no deployment or live order activity. The user subsequently explicitly authorized committing, pushing, and deploying the completed feature; existing live plans remain unchanged.
- Volume-based entry filtering was discussed only: checking near entry, deferring during adverse surges, and invalidating crossed stop levels. No thresholds or entry behavior were enabled without data validation.
