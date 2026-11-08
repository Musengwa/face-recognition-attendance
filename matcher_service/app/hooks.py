"""Extension hooks.

=====================================================================
 BLOCKCHAIN HOOK — intentionally NOT wired to any chain yet.
=====================================================================
`on_recorded(record)` is called once, from the outbox delivery thread, after Django answered
`{"status": "recorded"}` for an attendance event. This is where a future `markAttendance`
(e.g. an `attendance_chain.py` web3 call) would go once the write order (Django first vs chain
first, and how tx_hash/block_number flow back to Django) has been decided.

Keep it fast and never raise: exceptions are caught and logged, and the event is NOT re-sent.
"""
import logging

log = logging.getLogger("matcher.hooks")


def on_recorded(record):
    """record = {"event": <payload sent to Django>, "response": <Django's JSON body>}."""
    log.debug("on_recorded hook: event %s (no-op)", record["event"].get("event_id"))
