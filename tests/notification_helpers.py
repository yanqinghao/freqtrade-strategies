def capture_alerts(strategy, messages):
    """Observe real durable enqueue operations without contacting Telegram."""
    put = strategy.alert_queue.put

    def record(key, message, **kwargs):
        put(key, message, **kwargs)
        messages.append(message)

    strategy.alert_queue.put = record
