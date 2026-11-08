"""Durable outbox: attendance events go to SQLite first, a background thread delivers them to Django.

The camera loop only does a fast local INSERT, so a slow or dead Django never blocks it.
Events are delivered in order with exponential backoff and keep their original captured_at.
"""
import json
import logging
import sqlite3
import threading
import time

import httpx

log = logging.getLogger("matcher.outbox")

RETRYABLE_4XX = {401, 403, 404, 408, 409, 425, 429}


class Outbox:
    def __init__(self, path, send, on_delivered=None, max_backoff=60.0, clock=time.monotonic):
        """send(payload) -> httpx.Response (may raise httpx.HTTPError).
        on_delivered(payload, response) is called once per event after a 2xx answer."""
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
                row = self._db.execute("SELECT id, payload FROM events WHERE state='pending' ORDER BY id LIMIT 1").fetchone()
            if row is None:
                return delivered
            event_id, payload = row[0], json.loads(row[1])
            try:
                resp = self._send(payload)
                status = resp.status_code
            except httpx.HTTPError as exc:
                self._fail(event_id, f"{type(exc).__name__}: {exc}")
                return delivered
            if 200 <= status < 300:
                with self._lock:
                    self._db.execute("DELETE FROM events WHERE id=?", (event_id,))
                self.backoff, self.last_error = 0.0, None
                delivered += 1
                self._notify(payload, resp)
            elif status >= 500 or status in RETRYABLE_4XX:
                self._fail(event_id, f"HTTP {status}")
                return delivered
            else:
                # The payload itself was rejected: retrying will not help. Keep it for inspection.
                with self._lock:
                    self._db.execute("UPDATE events SET state='dead', attempts=attempts+1, last_error=? WHERE id=?",
                                     (f"HTTP {status}: {resp.text[:200]}", event_id))
                log.error("Django rejected event %s with HTTP %s; kept as dead in outbox", payload.get("event_id"), status)

    def _fail(self, event_id, error):
        with self._lock:
            self._db.execute("UPDATE events SET attempts=attempts+1, last_error=? WHERE id=?", (error, event_id))
        self.backoff = min(self.max_backoff, max(1.0, self.backoff * 2))
        self.next_attempt = self._clock() + self.backoff
        if error != self.last_error:
            log.warning("delivery to Django failed (%s); retrying in %.0fs", error, self.backoff)
        self.last_error = error

    def _notify(self, payload, resp):
        if self._on_delivered is None:
            return
        try:
            self._on_delivered(payload, resp)
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
