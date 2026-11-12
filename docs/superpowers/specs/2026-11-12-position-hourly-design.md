# Hourly position advisory reports

User requested scheduled checks for all currently held symbols, showing trends,
levels, PnL and recommended actions, and selected hourly delivery. Implement as
advisory notifications using existing deterministic rules and public market data.

## Coverage and report

- Follow every assistant-recorded open position, including partial/observed fills
  pending native accounting. Do not report unfilled or paused entry intent as a
  held position. No fixed BTC/ETH/SOL restriction; no external position adoption.
- Check at each hour boundary, plus the first eligible cycle after startup.
  Empty accounts generate no report or market requests. One short message per
  held trade, headed with Beijing snapshot time, direction, quantity and leverage.
- Show original average entry, executable exit-side reference (bid for long,
  ask for short), native remaining-position unrealized PnL and percentage; reuse
  `order_management.position_profit`, excluding realized proceeds. Quote failure
  or unaccounted fill must remain unavailable, never zero.
- Show latest closed 1d/4h/1h/15m EMA20/50 direction, and closed 1d/4h support and
  resistance with source close times and current-price distances. Reuse existing
  trend/level functions, validate continuity/OHLC/volume/latest period first, and
  never include forming candles in analysis. Short/new listings may lack history.
- Separately show original SL, effective protected target SL, original TP levels,
  TP stage and any existing soft-exit rules. Advisory reference levels are never
  written into the trading plan. Stop protection is a dated verification snapshot.

## Recommended action, in priority order

1. Missing quote, unaccounted position, unusable protection/plan: verify records
   and stop protection first, no directional hold/reduce recommendation.
2. Existing full-exit intent or exit order: verify execution, avoid duplicate exits.
3. Current price beyond effective SL or reached the next currently active TP:
   verify the original strategy's exit/TP execution, do not send another order
   automatically. Reuse the strategy's read-only `_targets` to respect extended
   TP3 while still displaying original targets and the active extension separately.
4. Missing/stale timeframe data: show healthy fields but withhold directional advice.
   If the current quote has already crossed the relevant 4h support/resistance,
   describe that as an intrabar condition and withhold a trend-following hold
   recommendation until completed-candle evidence is available.
5. 4h and 1h both opposite position: consider reducing/exiting; protect gains when
   profitable and control drawdown otherwise. Never recommend automatic averaging.
6. 4h and 1h both favorable: observe/hold with existing SL/TP. Favorable 1d/4h but
   opposite 1h/15m: describe pullback/rebound and watch the relevant 4h level.
7. Mixed/neutral: wait for clearer completed-candle evidence, follow the original
   protection plan. Advice is conditional, no profit or reversal guarantee.

## Execution and delivery

Public I/O runs on a separate worker. Only immutable trade IDs/pairs cross into
the worker, never ORM objects or private exchange credentials. Compose PnL,
protection and current plan on the trading thread after results arrive; filter
out trades already closed while fetching. Use actual completion time to validate
quotes (<=30s), latest completed candle periods and the current hourly slot.

Reuse AlertQueue.put_market_once for persisted hourly per-trade deduplication,
delivery retry and five-minute expiry. Reports are lower priority than critical
trade alerts. Clearly timestamp reports as snapshots, not execution instructions.
Notification delivery is not an atomic guarantee that a position is still open
when Telegram displays it. Existing market digest, entry checks and exits retain
their behavior. No automatic orders, modifications, extra services or dependencies.

## Approach and operational scope

Choose an independent position report worker so private snapshots and slow public
requests cannot block entry/exit processing. Reuse the existing analytics, native
PnL helper and notification queue rather than add an AI service or another queue.
Develop in an isolated source-only clone, then run pinned offline tests, project
format/lint hooks and a fresh review before returning changes for release.
