# Notifications and request history implementation plan

Goal: reliable actionable alerts, fewer unsolicited messages, reusable entry history.
Architecture: existing strategy and Telegram extension, persistent SQLite alert queue;
existing public candle analysis retained. No trading behavior changes or deployment.

- [x] Add failing persistence/delivery tests; implement alert queue and Telegram delivery acknowledgment, bounded retry and coalescing. Adapt notification assertions to queue boundary.
- [x] Add failing stop-notification tests; suppress reason/coverage-only repeats while preserving first failure and recovery.
- [x] Add failing compact-report tests; implement hourly summary and contextual observation hints, suppress volume-only pushes and merge simultaneous market alerts.
- [x] Add failing history pagination/command round-trip tests; implement compact history and detailed copyable command.
- [x] Update README/help; run full offline suite and review integration.

Review focus: a failed send must remain pending after restart; acknowledgment must not delete newer events; query must never submit; command formatting must preserve numeric precision and omitted parameters; market data failure must never turn into confident guidance.

Validation: 236 offline unittest cases passed. Independent review found exit notification intent could be lost on queue failure plus restart; fixed by persisting notice_pending with trade execution state and covered by test_failed_exit_notice_survives_queue_failure_and_restart.

Ruling: work in the existing checkout; no live restart, merge or push. History commands reconstruct saved parameters, not original message spelling. Volume-only events are folded into the hourly snapshot; no delayed replay of suppressed market events.
