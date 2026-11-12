# Entry target units implementation plan

Goal: Accept USDT/underlying-percent SL and multi-TP, convert to safe native prices, freeze actual initial-fill basis.
Spec: docs/superpowers/specs/2026-11-12-entry-target-units-design.md
Execute inline with TDD; existing pinned runtime and isolated source clone /tmp/entry-target-units-0ffds18m/repo.

## Constraints / review focus

- Fee-aware signed long/short math, native contract precision, actual base quantity, no second leverage multiplication.
- Raw serialization/preview must not mix generated prices with user units, mutate ledgers or create orders.
- Unresolved metadata must never bypass SL/TTL/volume/final-SDK/risk guards; limit restores retain original price plan.
- Filled initial basis survives partial TP, duplicated callbacks, restarts and a crash before plan binding; SL only tightens.
- Existing trades/manual edits/add-ons must not accidentally rebase a prior unit plan.

## Tasks

- [x] RED pure math, precision, validation, parser/history/preview contracts; implement minimum converter and normalization.
- [x] RED integration tests for preflight guards, actual-fill anchoring, partial fills/TP, restart, pause/restore and manual override; implement persisted lifecycle.
- [x] Update help/README/DESIGN, run project hooks and full pinned offline suite, fresh branch review and repair findings.
- [x] Back up state, commit/push/deploy and verify running hashes/heartbeats without live test orders.

## Ledger

- Pure math/parser first failed (missing converter/aliases), then5 passed with hand-derived fee-aware long/short tick values. Lifecycle first failed absent native-price resolution; initial12 then16 tests passed.
- First full479 tests: two help-example-count failures after intentionally adding two unit examples. Updated to parse all three examples while preserving the original soft/extension example assertions.
- Fresh code review reproduced2 P1s (late closed-trade callback overwriting a new same-pair plan; recovery anchor lacking seen causing late initial callback to reset TP) and1 P2 (valid zero fee replaced by estimate). Three added regressions first failed.
- Fixes: live/owner guard plus atomic JSON ownership/spec recheck on target writes; initial anchored order receipt markers registered in both recovery and normal fill, without clearing TP progress; zero and None rates distinguished. Target-specific19 tests pass (6.564s).

- Fresh re-review confirmed all3 findings fixed, independently reran19 new tests successfully. Recovery also restores missing receipt initialization after a JSON/custom-data split, and target writes re-check on-disk ownership under the existing state lock.

- Final full candidate482 offline tests passed (89.513s); project hooks, Black/flake8 and diff checks pass. Packaged RPC imports new unit converter via the resolver's loaded strategy cache; parse/serialize smoke succeeded and native STATUS / paused-order panel remain available.
- Candidate image be264b1f320dc527d3c36470a1effda66a88a64a24c574270bc38362db712b81 (entry-target-units-20261112). Strategy/helper logic is also in the user_data source bind: rollback must restore source and image, never roll live databases back over subsequent fills.

- Released commit `0003a31` to main and deployed candidate image at2026-11-12T08:46:25.889435828Z. Nine live strategy/RPC file hashes match. New post-start RUNNING heartbeat verified, restart count0, watchdog faults0. One startup Telegram ConnectError recovered; read-only getMe succeeded and subsequent45s had no errors. Private snapshots, startup log, hashes and release evidence are in `.assistant-archive/entry-target-units-20261112T084305Z/`. No live test orders or messages were sent.
