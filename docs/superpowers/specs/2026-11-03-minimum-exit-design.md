# Minimum-exit fallback

The user approved full closure as the fallback when a requested partial exit is not executable or would leave an unmanageable remainder. Preserve ordinary partial exits whenever legal. Use actual closing rules: Binance linear futures reduce-only orders do not inherit opening min-notional restrictions; amount, contract size and step filters still apply. Also satisfy the pinned Freqtrade runtime's local remainder check.

Validate both automatic TP and explicit `/close` reductions. Persist the full-exit intent until confirmed flat; partial fills, rejection, price pullback and restart must not lose that intent or advance a TP stage. Never add exposure to meet a minimum. A full close that is itself invalid must alert and retain the position/plan. A failed fallback request must alert, check/re-establish native exchange stop protection, and wait for explicit operator retry rather than blindly submitting repeated unknown orders.

No live service startup or trading is part of this change. Existing Binance HTTP 451 remains an external deployment blocker.
