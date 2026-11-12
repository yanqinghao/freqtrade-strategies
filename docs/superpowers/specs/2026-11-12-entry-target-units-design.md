# Amount / underlying-price-percent exit targets

Approved user behavior: explicit margin/leverage remain; SL may be absolute `sl=`, net-loss-USDT `loss=`, or adverse underlying move `sl_pct=`. TP may be absolute `tp=`, full-original-position net-profit-USDT `profit=`, or favorable underlying move `tp_pct=`. Each TP list has up to three stages; do not mix units within one leg, but SL and TP may use different units. Percent is underlying price move, not leveraged ROI. Existing price commands remain unchanged.

New units apply to initial entries (market and limit). Add-ons and /setmanual continue using price parameters; explicit price changes override the affected unit specification. An absolute add-on inherits resolved prices but clears unit metadata rather than silently recomputing a new full-position monetary goal. No new market-volume bypass/override and no new order types.

Canonical raw request fields: loss(float), sl_pct(float), profit(list), tp_pct(list). Resolved plans retain those input fields and `target_basis` metadata alongside native stop_loss/exit_points. History serializer emits only the original unit for each leg, never both derived price and relative unit. Preview is read-only and shows indicative converted prices. All conflicts, nonfinite/nonpositive values, >3 targets, impossible/nonpositive prices and tick-collapsed TP stages reject before submission.

## Conversion and protection

Pure Decimal conversion in entry_targets.py, using native exchange price_to_precision: ROUND_UP for long stop/TP, ROUND_DOWN for short stop/TP. With E entry price, Q original full base quantity, fo/fc fee rates:

- Long net profit at P = Q*P*(1-fc) - Q*E*(1+fo).
- Short net profit at P = Q*E*(1-fo) - Q*P*(1+fc).
- Solve net profit = -loss for SL and = each profit target for TP. Reject a loss smaller than estimated fees when the resulting SL is not on the losing side. Exclude future funding/slippage, consistent with existing estimates.
- Percent conversion: long SL E*(1-p/100), TP E*(1+p/100); short vice versa. Validate resulting positive prices and strictly favorable TP order.

Before submission, materialize a provisional price plan using current quote or original limit, normalized Q, taker fees. Both preview and native final risk guard use the same conversion path; all configured risk limits remain active. Persist that executable plan before SDK submission. Limit pause metadata receives native SL/TP plus unit inputs and provisional basis so existing SL/history/quarter-R gates continue working. Restores preserve this original provisional price plan/deadline, not a new quote-based one.

After an actual initial fill (including terminal partial entry), use actual trade average/Q and recorded fees to calibrate once and freeze TP prices. Persist source entry order identity and filled quantity. TP1/TP2, later fees, hourly reports, restart and add-ons must not rebase targets to remaining quantity. Native 30%/50%-of-remainder/all staging remains unchanged. SL never loosens versus the already prepared/active protection. If post-fill conversion cannot produce a safe valid plan, keep the provisional SL and persist protective full-exit intent; never drop protection or fabricate fulfilled targets.

Persist realized basis in both plan JSON and native custom-data; update only the matching current trade. Late/repeated callbacks and crash between native submit/plan-bind must not overwrite the actual basis. Recover unanchored plans from entry order receipts on a trading-thread loop before TP processing; no external-account mutations for recovery. A conflicting frozen basis is not silently recalculated.

## Integration

- entry_targets.py owns pure conversion, unit-field clearing/description; entry_risk resolves quotes/quantities/fees and applies existing risk gates.
- telegram.py parses/copies/help/previews, creates provisional executable requests and plans. /setmanual new unit inputs reject clearly instead of being silently ignored.
- strategy owns initial-fill anchoring and restart recovery, plus unchanged native exits. Pause metadata and manual edits carry/clear unit metadata consistently.
- /orders details display input units and the current resolved levels/basis. Do not alter upstream startup/trade notification formats or add keyboard status messages.

No new dependencies, no exchange/API credentials in diagnostics, no live test trades, no risk-limit changes or automatic sizing. Existing authorization covers commit/push/deploy after offline verification and fresh review.

Builder-check rulings: preserve provisional pending price plans at final SDK boundary; enforce actual normalized quantity/price against the user's cash-loss budget even if global caps are disabled. Do not silently regenerate a restored plan. JSON is written before custom-data; on restart a matching anchored copy repairs a provisional copy using only still-active unit legs. Conflicting anchored records block TP and latch protective exit. Manual price edits clear the affected raw units, so delayed fills cannot resurrect them. Fee basis uses recorded opening fee rate when available, otherwise the persisted preflight estimate; closing fee is explicitly an estimate, frozen with the basis. Future funding/slippage are excluded. Unanchored positions retain provisional SL and cannot trigger TP until a terminal entry receipt anchors them.
