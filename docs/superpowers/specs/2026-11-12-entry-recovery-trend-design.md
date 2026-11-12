# Entry recovery with a bounded immediate-fill branch

User approved developing the proposed version on 2026-11-12. This extends the
unreleased symmetric resting-limit fix; production activation is separate.

## Follow-up: follow every manually ordered symbol

The user clarified that every manually ordered symbol must receive these entry
checks. Remove BTC/ETH eligibility restrictions from new initial limits and
adoption of identifiable existing zero-fill initial limits. The public worker
must dynamically add every symbol with an unfinished plan, preserve the oldest
unchecked history start across same-symbol plans, and drop inactive extra
symbols on later scheduling rounds. Reuse the existing three advisory symbols
only for background volume prewarming; they do not limit protection coverage.
An unseen symbol without fresh data retains its intent and waits through the
existing quiet/continuous observation rules rather than bypassing checks.
Advisory reports remain on their existing three symbols. Partial positions,
add-on orders, market orders and independent external orders retain the prior
scope and lifecycle; this change expands symbol coverage, not order ownership.

## Behavior

Let E be the original limit, S the hard SL, R=abs(E-S). Restore the original
limit, amount/risk request, SL/TP and original deadline; never switch to market.

- Resting side: long ask>E / short bid<E has no distance or trend gate. If entry
  was crossed, retain the existing completed 1m close/current quote reclaim of E.
- Potential immediate fill: long ask<=E / short bid>=E requires both current quote
  and latest completed 1m close within the allowed region: long >= E-0.25R;
  short <= E+0.25R. Exact boundary and equality at E are included.
- A breach beyond the allowed region must be followed by a completed 1m close
  back in the region; a pre-breach close cannot authorize recovery. The breach
  candle itself may qualify once closed. Original-entry reclaim is deliberately
  replaced by region reclaim for the immediate-fill branch.
- Only immediate-fill recovery checks 15m trend. Reuse existing EMA20/EMA50:
  close>EMA20>EMA50 is bullish, close<EMA20<EMA50 bearish, otherwise neutral.
  Bearish vetoes long, bullish vetoes short, neutral allows either.
- Trend uses the latest completed 15m candle and up to 120 completed candles,
  at least 50; never include a forming candle. Validate raw candle continuity,
  OHLC/volume, latest candle, and snapshot age <=30s. Missing/invalid/stale trend
  blocks immediate-fill recovery, not resting recovery or volume publication.
- Preserve quiet 900s, continuous volume observation 900s, price-history checks,
  terminal hard SL/TTL/user cancellation/replacement, zero-fill receipts,
  partial-fill management, bot state/account/risk checks and final SDK recheck.
- Final resume checks fetch actual best ask/bid directly from the native order
  book API, not configurable entry target pricing. Recheck time and RUNNING
  after network I/O so old minute/15m data cannot pass a crossed boundary.
- Initial normal entries and existing positions gain no new trend/range gate.
  Cancellation remains `/request <original reference> cancel`.

## Implementation choices

Reuse the existing public background worker and `market_monitor.analyze_trend`;
publish valid volume and minute/book snapshots before extra trend I/O. Keep
separate original-entry and allowed-region crossing timestamps to prevent a
continuously-below-entry quote from making region recovery impossible. No new
dependencies, configuration knobs, or changes to advisory trend classification.

## Approaches

Selected the agreed narrow reverse-trend veto for potential immediate fills.
Requiring a favorable trend for every restoration would also reject neutral
markets and change the resting-order policy; retaining original-price reclaim
would make the new immediate-fill region unreachable. Neither matches the
approved proposal. This version has functional verification, not profit claims.

## Operational scope

Develop on an isolated source-only clone under /tmp, carrying the current
uncommitted symmetric fix. Do not copy production databases/configuration,
commit/push shared history, cancel a real plan, or restart production.
