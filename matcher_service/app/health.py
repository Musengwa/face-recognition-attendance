"""Background connectivity probes (embedding service, blockchain node) for the status page.

Each probe runs a cheap read-only call every `interval` seconds and remembers the outcome, so the
dashboard can show whether the matcher is connected even when no frames or events are flowing.
Django is not probed here: the gallery thread already polls it and records its own state.
"""
import logging
import threading
import time

log = logging.getLogger("matcher.health")


class Probe:
    def __init__(self, name, check):
        """check() -> detail dict; raise to report the service as down."""
        self.name = name
        self._check = check
        self._lock = threading.Lock()
        self.ok = None                # None until the first check finishes
        self.error = None
        self.detail = {}
        self.latency_ms = None
        self.checked_at = None        # wall clock (epoch seconds)
        self.last_ok_at = None

    def run_once(self):
        started = time.monotonic()
        try:
            detail = self._check() or {}
            ok, error = True, None
        except Exception as exc:      # any failure means "not connected"; keep the message for the page
            detail, ok, error = {}, False, f"{type(exc).__name__}: {exc}"[:300]
        latency = round((time.monotonic() - started) * 1000)
        with self._lock:
            if ok != self.ok:
                if ok:
                    log.info("%s reachable (%d ms)", self.name, latency)
                elif self.ok is not None:
                    log.warning("%s unreachable: %s", self.name, error)
            self.ok, self.error, self.latency_ms, self.checked_at = ok, error, latency, time.time()
            if ok:
                self.detail, self.last_ok_at = detail, self.checked_at

    def snapshot(self):
        with self._lock:
            return {"ok": self.ok, "error": self.error, "detail": dict(self.detail), "latency_ms": self.latency_ms,
                    "checked_at": self.checked_at, "last_ok_at": self.last_ok_at}


class HealthMonitor:
    def __init__(self, probes, interval=15.0):
        self.probes = {p.name: p for p in probes}
        self.interval = interval

    def snapshot(self):
        return {name: p.snapshot() for name, p in self.probes.items()}

    def run(self, stop):
        while not stop.is_set():
            for probe in self.probes.values():
                if stop.is_set():
                    break
                probe.run_once()
            stop.wait(self.interval)


def embed_probe(embed):
    def check():
        info = embed.health()
        return {"model": info.get("model"), "status": info.get("status")}
    return Probe("embedder", check)


def chain_probe(sink):
    def check():
        info = sink.check()
        if not info["owner_ok"]:
            raise RuntimeError(f"sender {info['sender']} is not the contract owner {info['owner']}; writes will revert")
        return {k: info[k] for k in ("chain_id", "contract", "sender", "records")}
    return Probe("chain", check)
