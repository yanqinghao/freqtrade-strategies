# History preview implementation

- [x] RED: readonly history preview, repeated button presses, auth rejection, current limits, missing SL, missing reference, editable command roundtrip, allowed versus blocked manual command.
- [x] Implement shared preview renderer, request buttons/callback, reference shortcut, bounded HTML output.
- [x] Update help/README; full offline suite, independent review, formatting/lint.
- Rollout: commit/push, rebuild Telegram image, recreate trading service, verify hash and execution heartbeat without sending test commands.

Verification: 259 offline tests passed. Independent review found no important issues. Verified actual authorized callback path, repeated unknown-request preview without resolution, current limits, plan/TP immutability, precision/TTL roundtrip and last-chunk-only keyboard attachment.
