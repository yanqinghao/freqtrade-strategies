# Risk Preview and Overview Implementation Plan

> Use superpowers:executing-plans inline; preserve existing approved commit/push/deploy workflow and host-managed isolation. User supplied scope and latest numeric constraints; no additional approval round for routine implementation.

**Goal:** Show entry risk, enforce user limits on both requested and native final parameters, and expose a read-only combined trade overview.
**Architecture:** Shared entry_risk helper for strict hot-read limits, pure plan/stop estimation, exposure and fee calculation; native exchange create_order guard for final entry parameters; read-only trade_overview formatter. Telegram routes all native-state reads on trading thread.
**Tech Stack:** pinned Freqtrade2026.10, stdlib Decimal/JSON, unittest/SQLite, Docker Compose.
**Spec:** docs/superpowers/specs/2026-11-03-risk-overview-design.md

## Global Constraints

- Deployment caps: margin100USDT, leverage20, assistant-only total notional1000USDT, per-symbol post-add estimated stop loss10USDT. Never auto resize or change leverage.
- Hot config deploy/assistant_risk.json; malformed present config blocks entries, missing is explicitly unconfigured. No effect on closing-only orders or historical trades/orders.
- Include open entry remainders and unaccounted positive fills; no double contract conversion. Existing whole-position losses/fees cannot be bypassed by repeated adds.
- Pure preview/overview: no _plan/_stop_price persistence, no journal reservation/status mutation, no order execution; no real-time SL protection claim from old snapshot.
- Final boundary checks normalized native parameters for entries; reduceOnly exits bypass all risk reads. Known local risk rejection restores plan and remains rejected, not unknown.
- No new dependencies/native core or DB schema edits; preserve TTL24h/request semantics/stop guards. No live test orders or injected faults.

## Review Focus

- Native validation can enlarge minimum stake or change price/leverage; initial-only guard misses adds and adjusted values.
- Pending partial fills, terminal receipts before bookkeeping, missing data/fees and cross-pair unknown requests must not undercount exposure or risk.
- Current request is submitting during final guard; exclude only its own reference, never stale/other unresolved intents.
- Preview/overview reads must not reset TP stages, tighten/persist stops, create requests or falsely confirm stale protection.
- Final rejection before exchange must restore current plan; malformed config/data cannot block exits or leak private exceptions.

## Task 1: Pure risk estimates and preview

Files: new entry_risk.py; strategy helper wiring; telegram preview; tests/test_entry_risk.py and risk Telegram tests.
Interfaces: load_limits(path), plan_for_entry(strategy,request,trade), estimate_exposure(exchange,trades,fresh=True), assess_entry(...); EntryRiskError; strategy preview_entry. /preview long|short uses parse_entry.
- [x] Baseline full suite; RED numeric limits, long/short fee/SL, partial/contract/unknown exposure, add cumulative risk, pure preview tests.
- [x] Implement minimal shared estimates and read-only preview; full GREEN suite; commit.

## Task 2: Entry enforcement at final native boundary

Files: entry_risk.py installation; strategy native context; Telegram preflight and known-rejection handling; journal unresolved-all read; native integration tests.
Interfaces: install_risk_guard(strategy,bot), EntryRiskError class exposed by strategy, strategy current request context; guard uses actual native base amount/rate/leverage and current limits.
- [x] RED requested/final cap breaches including native sizing changes, bypassed add confirmation, self/other pending request handling, restore plan, metadata/query failure and exit bypass.
- [x] Implement enforcement and context lifecycle; GREEN complete suite, hooks; commit.

## Task 3: Read-only overview and documentation

Files: new trade_overview.py, Telegram /overview, tests/test_trade_overview.py; README/help/config example.
Interfaces: overview_messages(strategy,trades,bot_state,now), pure snapshot rendering.
- [x] RED no trades/pending/TTL/TP/SL age/unknown request/invalid field/multimessage read-only tests.
- [x] Implement overview and help/docs with exact units/limits and estimation caveats; complete GREEN, commit.

## Task 4: Review and rollout

- [x] Fresh whole-change read-only review, fix important findings via RED→GREEN and full suite.
- [x] Merge/build/verify merged image/push; back up native/request DBs, plans and old risk file; write approved100/20/1000/10 limits atomically.
- [x] Update service and verify source artifacts, policy read, preview/overview read-only, RUNNING/TG/watchdog and unchanged tracked records; no test orders.
- [x] Record results, push docs, archive own execution artifacts and retain host-managed checkout.

Result:223 offline tests passed; review findings fixed, main pushed and deployment verified RUNNING with active100/20/1000/10 limits and preserved records. See [verification and decisions](2026-11-03-risk-overview-progress.md).
