"""Extension hooks.

`on_recorded(record)` is called once, from the outbox delivery thread, after an attendance event was
recorded by the sink:
  * chain sink (default): record = {"event": <payload>, "response": {"status": "recorded", "tx_hash", "block_number"}}
  * django sink:          record = {"event": <payload>, "response": <Django's JSON body>}

Use it for side effects such as telling the dashboard about the tx hash. Keep it fast and never raise:
exceptions are caught and logged, and the event is NOT re-sent.
"""
import logging

log = logging.getLogger("matcher.hooks")


def on_recorded(record):
    log.debug("on_recorded hook: event %s (no-op)", record["event"].get("event_id"))
