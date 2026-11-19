"""Durable outbox: attendance events go to SQLite first, a background thread delivers them to the sink.

The sink is the Ethereum contract (default) or the Django dashboard, see sinks.py. The camera loop only
does a fast local INSERT, so a slow or dead node/dashboard never blocks it. Events are delivered in order
with exponential backoff and keep their original captured_at.
"""
import json
import logging
import sqlite3
import threading
import time

from .sinks import DeliveryRejected, DeliveryRetry

log = logging.getLogger("matcher.outbox")


class Outbox:
    def __init__(self, path, send, on_delivered=None, max_backoff=60.0, clock=time.monotonic):
        """send(payload, progress, checkpoint) -> result dict; raises DeliveryRetry / DeliveryRejected (sinks.py).
        on_delivered(payload, result) is called once per delivered event."""
        self._send = send
        self._on_delivered = on_delivered
        self._clock = clock
        self.max_backoff = max_backoff
        self.backoff = 0.0
        self.next_attempt = 0.0
        self.last_error = None
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS events ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " payload TEXT NOT NULL,"
            " state TEXT NOT NULL DEFAULT 'pending',"   # pending | dead
            " attempts INTEGER NOT NULL DEFAULT 0,"
            " last_error TEXT,"
            " created_at REAL NOT NULL)"
        )
        columns = {row[1] for row in self._db.execute("PRAGMA table_info(events)")}
        if "progress" not in columns:       # outboxes created before the chain sink
            self._db.execute("ALTER TABLE events ADD COLUMN progress TEXT")
        self._wake = threading.Event()

    def put(self, payload):
        with self._lock:
            self._db.execute("INSERT INTO events (payload, created_at) VALUES (?, ?)", (json.dumps(payload), time.time()))
        self._wake.set()

    def counts(self):
        with self._lock:
            rows = self._db.execute("SELECT state, COUNT(*) FROM events GROUP BY state").fetchall()
        out = {"pending": 0, "dead": 0}
        out.update(dict(rows))
        return out

    def pending(self):
        with self._lock:
            rows = self._db.execute("SELECT payload FROM events WHERE state='pending' ORDER BY id").fetchall()
        return [json.loads(r[0]) for r in rows]

    def deliver_due(self):
        """Try to deliver pending events, oldest first. Returns the number delivered."""
        if self._clock() < self.next_attempt:
            return 0
        delivered = 0
        while True:
            with self._lock:
                row = self._db.execute(
                    "SELECT id, payload, progress FROM events WHERE state='pending' ORDER BY id LIMIT 1").fetchone()
            if row is None:
                return delivered
            event_id, payload, progress = row[0], json.loads(row[1]), row[2]
            try:
                result = self._send(payload, progress, lambda value, eid=event_id: self._checkpoint(eid, value))
            except DeliveryRetry as exc:
                self._fail(event_id, str(exc))
                return delivered
            except DeliveryRejected as exc:
                # The event itself was rejected: retrying will not help. Keep it for inspection.
                with self._lock:
                    self._db.execute("UPDATE events SET state='dead', attempts=attempts+1, last_error=? WHERE id=?",
                                     (str(exc)[:300], event_id))
                log.error("event %s rejected (%s); kept as dead in outbox", payload.get("event_id"), exc)
                continue
            with self._lock:
                self._db.execute("DELETE FROM events WHERE id=?", (event_id,))
            self.backoff, self.last_error = 0.0, None
            delivered += 1
            self._notify(payload, result)

    def _checkpoint(self, event_id, value):
        """Persist sink progress (the chain tx hash) before waiting for the outcome."""
        with self._lock:
            self._db.execute("UPDATE events SET progress=? WHERE id=?", (value, event_id))

    def _fail(self, event_id, error):
        with self._lock:
            self._db.execute("UPDATE events SET attempts=attempts+1, last_error=? WHERE id=?", (error, event_id))
        self.backoff = min(self.max_backoff, max(1.0, self.backoff * 2))
        self.next_attempt = self._clock() + self.backoff
        if error != self.last_error:
            log.warning("delivery failed (%s); retrying in %.0fs", error, self.backoff)
        self.last_error = error

    def _notify(self, payload, result):
        if self._on_delivered is None:
            return
        try:
            self._on_delivered(payload, result)
        except Exception:
            log.exception("on_delivered hook failed for event %s", payload.get("event_id"))

    def run(self, stop):
        while not stop.is_set():
            try:
                self.deliver_due()
            except Exception:
                log.exception("outbox loop error")
            wait = max(0.2, self.next_attempt - self._clock()) if self.backoff else 1.0
            self._wake.wait(timeout=wait)
            self._wake.clear()

    def close(self):
        with self._lock:
            self._db.close()
