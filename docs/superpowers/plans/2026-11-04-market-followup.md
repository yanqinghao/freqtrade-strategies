# Market report clarity implementation plan

- [x] Add RED tests for compact trends/distances, missing data and already crossed references; update market_monitor formatting.
- [x] Add RED tests for crossing magnitude, low/invalid volume and frozen one-shot follow-up: up/down/equality, refreshed levels, restart, skipped target, storage failure and multi-coin aggregation.
- [x] Implement pending event in market_alerts JSON and merged feedback, preserving existing cooldowns and closed-candle rules.
- [x] Update README, run full offline suite, independently review.
- Rollout: Commit with hooks, push main, deploy trading service only, verify fresh execution heartbeat and logs.

Verification: 249 offline tests passed. Independent review found no important issues. Follow-up applies to the exact next closed candle; a later snapshot expires the pending record rather than substituting a later outcome. Hourly digest stays separate from the merged alert/follow-up message.
