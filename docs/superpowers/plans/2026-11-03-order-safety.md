# Order Safety Implementation Plan

> Use superpowers:executing-plans inline, preserving the established execution method. User approved the short design and existing commit/push/deploy workflow; no additional approval round.

**Goal:** Bound new limit-entry lifetime, settle partial fills safely, and confirm actual stop protection before allowing adds.

**Architecture:** Add entry_lifecycle and stop_protection helpers, eagerly imported by the strategy. Reuse native order metadata and official reconciliation/cancel/query methods on the trading thread. Store per-order TTL in ft_order_tag and protection/notification state in the existing canonical trade custom-data document.

**Tech Stack:** existing Freqtrade 2026.10, stdlib Python/Decimal, unittest, Docker/Compose.

**Spec:** docs/superpowers/specs/2026-11-03-order-safety-design.md

## Global Constraints

- Default limit TTL4h; explicit integer m/h/d, 1m..5d. Market TTL rejected; legacy untagged orders receive no new TTL.
- Partial fill cancels remainder before native reconciliation, preserving every positive fill. Never automatically add exposure or fabricate a canceled status.
- Only tracked manual limit entries; no cancellation of stop/exit/external orders.
- Stop confirmation uses fresh exchange receipts and normalized base quantity; no double contract conversion. 15s initial grace, normal30s/pending15s checks, persisted transition notices.
- Adds require fresh position AND stop checks before plan mutation; exits remain unblocked. No new auto-liquidation/core patches.
- Work in managed isolated checkout. No live validation trades or injected production faults; preserve plans/records, allow actual configured lifecycle behavior after update.

## Review Focus

- Native cancel-enter refuses tiny partial fills: cancel remaining using official exchange cancel, then reconcile positive terminal fills directly; never delete them as zero.
- Order fills during cancel: reconcile actual terminal fill, never claim cancellation or send a replacement order.
- Crash/restart/plan edits cannot reset order expiry; legacy orders retain original timing.
- Triggered/closed/unknown stop receipts cannot become a positive confirmation; malformed closing flags and contract units fail closed.
- Reconciliation callbacks mutate canonical custom data: always reload before saving lifecycle/protection state, preserving TP stage/stop snapshots.

## Task 1: Limit lifetime and partial settlement

Files: telegram.py; new user_data/strategies/entry_lifecycle.py; strategy integration; tests/test_entry_lifecycle.py and Telegram tests.
Interfaces: ttl_seconds(value); entry_ttl(order) -> duration/None; process_entry_orders(strategy,bot,trade,now) handles only known manual limit entries, delegates official accounting. Telegram encodes TTL into enter_tag and excludes it from per-position plan.

- [x] Write failing duration/tag/expiry/legacy/partial/tiny-fill/cancel-race/restart/unknown-receipt tests, plus parser and per-order tag integration.
- [x] Run offline scoped tests and inspect missing-feature failures.
- [x] Implement lifecycle and wiring, persist deduplicated notices without overwriting native fill state.
- [x] Run full suite and commit after green.

## Task 2: Stop protection confirmation and add gate

Files: new user_data/strategies/stop_protection.py; strategy + Telegram integration; tests/test_stop_protection.py and native/fresh-process regressions.
Interfaces: check_protection(strategy,trade,now,force=False) -> bool, updating canonical protection state; strategy.ensure_entry_protection(trade) forces fresh validation for adds.

- [x] Write failing long/short/price/quantity/closePosition/reduceOnly/triggered/error/grace/dedup tests, proving mismatches block adds without plan mutation and exits still work.
- [x] Run scoped RED, implement and run complete GREEN suite; update help/README. Commit task.

## Task 3: Review and rollout

- [ ] Fresh read-only whole-change review; fix important findings via RED→GREEN and full green suite.
- [ ] Merge main, test merged source, push; prepare consistent DB/plan backup and query current runtime/account read-only.
- [ ] Build/update service, verify guarded artifact and native RUNNING + watchdog health; compare prior order IDs and plans without assuming market fills cannot occur.
- [ ] Record results and push completion docs; retain host-owned worktree, archive own ledger/review artifacts.

## Progress

Preflight: TTL order metadata is per order, protection state shares only the canonical assistant_execution document; callbacks must reload after native update_trade_state. Both helpers imported at strategy module load. No interface conflict.
