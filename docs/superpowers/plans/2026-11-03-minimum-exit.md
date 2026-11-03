# Minimum Exit Fallback Implementation Plan

> Execute inline with superpowers:executing-plans and test-driven-development. User approved the fallback policy in chat.

**Goal:** Legal partial exits continue; non-executable partial exits or dust remainders become a durable full-close intent.

**Architecture:** A small shared exit-policy function validates exchange filters and native remainder constraints. The existing strategy and TG command consume its decision. Native core/RPC remain unmodified. Runtime stop-protection verification uses the existing native handler on the bot thread.

**Spec:** `docs/superpowers/specs/2026-11-03-minimum-exit-design.md`

## Constraints

- Binance USDT futures reduce-only min-notional exemption is respected.
- Check base quantity, contract conversion, applicable min/max and step rules plus native remainder blocking.
- No stage advancement or cleanup without confirmed fills. Full fallback remains latched after partial fills.
- No extra entries, real orders, credentials in output or service startup.

## Tasks

- [x] Write failing tests for below-minimum partials, tiny remainders, reduce-only exemptions, contract units, full-invalid and ordinary legal exits.
- [x] Implement shared policy and use it for TP and manual partial-close decisions.
- [x] Persist full intent, handle partial fills/replay, warn on failures and verify native stop protection; add regression tests.
- [x] Update operator documentation; run full offline 2026.10 suite, lint and a fresh scoped code review.

## Progress

Initial diagnosis: the old strategy only rounded tranche quantities. Official 2026.10 auto adjustments return without executing when remainder value is below their threshold; manual RPC raises on too-small remainder. Full-close orders use the native full path or a full-size tagged request, never an opening order.

Added regression tests first, including native bot/RPC integration, ORM reload, failure alerts and native stop-handler invocation. Review found an unsafe retry after order reconciliation and a position/order maximum mix-up; failing regressions reproduced both before correction. Native integration also reproduced float conversion dropping a validated minimum lot by one step. Exact inverse arithmetic plus a native-equivalent roundtrip check now protects the quantity.

All 62 offline tests pass on the freshly built 2026.10 deployment image (`/tmp/minimum-exit-final-tests.log`). Scoped Flake8, Black, `git diff --check`, Compose validation and image build passed. Final independent review found no remaining critical or important findings. No live startup or orders.
