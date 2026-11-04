# Preview a historical entry

User accepted prioritizing reusable history → risk preview; overview prioritization and notification preferences remain deferred.

Use existing Telegram controls and parser: /request list/detail gets per-request risk-preview buttons; /preview <request-ref> provides a text fallback. Both load the saved request without reconciliation or status changes, validate via the existing entry parser, and calculate using current state/limits/prices. No automatic order, reservation, resolution or plan write.

Every valid preview shows a copyable full /preview long|short ... command with all explicitly saved parameters, preserving numeric precision and omitted fields. Users edit the familiar command and send it to recalculate. A locally allowed result additionally shows the matching manual /forcelong or /forceshort command, explicitly describing a new entry/add request and actual-submit rechecks. Blocked results show reasons and the editable preview only. Missing SL remains missing; explain inheritance from current position if applicable, rather than silently inserting a historical or tightened stop.

Reuse authorized_only for callbacks, acknowledge callback promptly, register nonblocking handler; invalid/expired refs fail visibly without side effects. HTML-escape commands and labels; attach up to five preview buttons only to the final chunk of a history response, no additional notification spam.

No new dependencies, schemas or UI framework. Preserve original preview syntax. Follow established commit/push/deploy workflow after offline testing/review.
