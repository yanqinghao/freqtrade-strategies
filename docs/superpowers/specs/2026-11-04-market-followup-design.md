# Market report clarity and one-shot follow-up

User approved the four review recommendations: explicit timeframe trends, distance to key levels, crossing magnitude/volume context, and one follow-up using frozen original levels. Existing hourly digest + key-event cadence remains.

- Compact digest shows daily/4h/1h/15m trends, timestamp of 15m reference close, and 4h support/resistance with distance relative to that reference close. Already crossed levels are labeled as crossed rather than as upcoming barriers. Daily levels remain in /monitor details.
- Crossing alerts identify the reference timeframe, frozen level, percentage beyond it, and whether volume meets the existing 2x threshold. No profitability/strength claims.
- One pending record per coin stores original event candle and all emitted level crossings. Evaluate only the immediately next closed 15m candle, using strict greater/less comparisons; equality is labeled back at the level. Combine all level outcomes and any new event for that coin in one message.
- A restart may complete an existing pending follow-up if the latest closed candle is exactly the target candle. If the target has already passed, expire without historical catch-up. Initial baselining still suppresses new historical events. Failed or stale data cannot consume the pending record; persistence failure emits nothing and retries safely.
- Follow-up is exempt from initial-event cooldown; original 60-minute direction cooldown remains. Persist before sending, matching existing advisory-market best-effort delivery. No order/plan access.
- Reuse existing modules and JSON state; no dependencies or extra workers. Submit/push/deploy following the established user-authorized rollout after offline regression and review.
