# Telegram Order Panel Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans task by task.

**Goal:** Make pending orders and positions manageable through Telegram buttons.
**Architecture:** Native Telegram list/detail/reply flow, serialized trading-thread functions, existing Freqtrade accounting and persistent request ledger.
**Tech Stack:** Python, python-telegram-bot, pinned Freqtrade 2026.10, unittest.
**Spec:** docs/superpowers/specs/2026-11-06-order-panel.md

## Global Constraints
Chinese native Telegram UI. No new dependencies. Actual entry fills immutable. No real trading during tests. Preserve protection and TP execution state. Unknown outcomes never automatically retry.

## Review Focus
Partial fills during amendment; stale cross-user replies; TP progress reset; duplicate destructive callbacks; fresh exchange failure.

- [x] Add failing business tests for fresh snapshots, safe plan edits and cancel/replacement races in tests/test_order_management.py.
- [x] Implement user_data/strategies/order_management.py using existing lifecycle/risk/accounting helpers. Run tests.
- [x] Add failing UI tests for scoped reply/callback state, pagination, confirmation and errors in tests/test_order_panel.py.
- [x] Implement user_data/strategies/order_panel.py and register handlers in telegram.py; document native UI in DESIGN.md.
- [x] Run pinned-container targeted/full tests and pre-commit; request independent safety review and fix findings.
- [ ] Commit/push/deploy with backup, preserve plans/orders, verify runtime health and hashes; report usage and limitations.

## Verification and rulings
- Pinned offline suite: 322 tests passed; pre-commit passed; candidate Docker build and native import/handler smoke passed without network access.
- Independent review covered partial/tiny fills, ambiguous closes, deadline crossings and stale callbacks; all important findings fixed and re-reviewed.
- Ruling: use a durable TP edit marker to preserve execution progress across the JSON/native database write boundary; completed/partially filled targets cannot be edited.
- Ruling: a pending-price amendment cancels then recreates under the existing request ledger. Unknown outcomes block replacement; an absolute deadline survives new order timestamps and is checked immediately before the SDK request, after leverage preparation. Network transit after initiation cannot be time-bounded locally.
- Ruling: all panel callbacks are single-use and user/chat/topic/message bound; new views invalidate older mutation buttons. Telegram owns rendering. Text/keyboard/interaction tests validate native UI behavior; no live test order or close is executed.
