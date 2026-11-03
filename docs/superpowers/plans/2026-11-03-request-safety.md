# Request Safety Implementation Plan

> Use superpowers:executing-plans inline, preserving the established execution method and approved two-feature scope; existing commit/push/deploy authorization persists.

**Goal:** Persist entry-request identity/outcomes and block conflicting same-symbol external entry orders.
**Architecture:** Stdlib SQLite request journal, immutable Telegram-message identity, native per-order request tag and conservative unknown recovery; extend fresh pre-entry guard with ordinary/conditional CCXT queries.
**Tech Stack:** pinned Freqtrade2026.10, Python stdlib/sqlite3, unittest, Docker Compose.
**Spec:** docs/superpowers/specs/2026-11-03-request-safety-design.md

## Global Constraints

- New message is new intent; same message cannot submit twice, including after restart or parameter edits.
- Persist submitting before native RPC; unknown never automatically retries and blocks only same-pair exposure increases. Exits unchanged.
- /request [ID] queries latest5 or one; explicit resolve requires fresh positions and no potential entry orders, never resubmits.
- Native order tags preserve TTL24h default, legacy tags/deadlines remain compatible.
- Query regular and trigger open orders; only explicit consistent reduceOnly/closePosition permits ignoring exposure, query/malformed failures block before plan changes.
- No new dependencies, native DB schema/core edits, live validation orders, or injected production failures.

## Review Focus

- Crash after broker acceptance before native persistence: missing local tag never means rejection or permission to resend.
- Different message/edited message and native exceptions cannot mutate an earlier request or stale protection plan.
- Market/limit/add requests correlate per order, including canceled/deleted trades, without extending TTL.
- Conditional and opposite-side orders with false/missing/contradictory closing flags cannot evade guard; safe exits and different symbols remain usable.
- Journal/storage/query failures fail closed for entries but never block native exits or expose raw private API exception text.

## Task 1: External entry orders

Files: entry_guard.py, strategy wiring, tests/test_entry_guard.py and tests/test_telegram_orders.py.
Interfaces: ensure_no_entry_orders(exchange,pair,*,dry_run=False), consumed by ensure_entry_position.
- [x] Add RED ordinary/conditional opening, explicit closing, malformed/error, isolation/simulation tests and before-plan integration.
- [x] Implement fresh query/classification and wiring, complete full GREEN suite; commit.

## Task 2: Persistent request journal and Telegram flow

Files: new entry_requests.py, telegram.py, strategy eager import/init, entry_lifecycle tag parser, tests/test_entry_requests.py and Telegram/lifecycle tests; README/help.
Interfaces: EntryRequests.reserve/get/recent/set_status/unresolved; Telegram execution/status/resolve wrappers, native tag |req=<24hex>.
- [x] Add RED duplicate/new-message/fingerprint/restart/unknown/resolve/storage failure/native tag and TTL tests.
- [x] Implement journal, guarded dispatch/status recovery, explicit resolution and query command; update help/README.
- [x] Run complete GREEN suite and hooks; commit.

## Task 3: Review and rollout

- [ ] Fresh read-only whole-branch review; important findings reproduced RED and fixed GREEN.
- [ ] Merge/push, build and verify merged image tests; consistent DB/plan/request-journal backups, read-only preflight.
- [ ] Update service; verify image artifacts, native RUNNING, Telegram, watchdog and existing records/plans; no live test entries.
- [ ] Record rollout, archive own workspace and preserve host-managed checkout.
