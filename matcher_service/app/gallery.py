"""Keeps the session and full galleries in sync with Django; survives Django outages."""
import logging
import threading
import time
from dataclasses import dataclass

import httpx

from .matching import Gallery, GalleryError

log = logging.getLogger("matcher.gallery")


@dataclass(frozen=True)
class Galleries:
    session: Gallery
    full: Gallery
    session_ids: tuple
    version: str
    loaded_at: float


class GalleryManager:
    def __init__(self, django, refresh_seconds=60.0, poll_seconds=10.0, clock=time.monotonic):
        self.django = django
        self.refresh_seconds = refresh_seconds
        self.poll_seconds = poll_seconds
        self._clock = clock
        self._current = None
        self.last_error = None
        self.last_ok = None
        self.last_ok_at = None        # wall clock, for the status page

    @property
    def current(self):
        """Latest Galleries, or None if nothing was ever loaded. Swapped atomically."""
        return self._current

    def refresh(self, force=False):
        """Poll active sessions; re-download galleries if forced, stale, or the version/sessions changed.

        Returns True if galleries were (re)loaded. On any failure the previous galleries stay in use.
        """
        try:
            active = self.django.active_sessions()
            session_ids = tuple(sorted(str(s["id"]) for s in active.get("sessions") or []))
            version = str(active.get("gallery_version", ""))
            cur = self._current
            stale = cur is None or self._clock() - cur.loaded_at >= self.refresh_seconds
            changed = cur is not None and (version != cur.version or session_ids != cur.session_ids)
            if not (force or stale or changed):
                self.last_ok, self.last_ok_at, self.last_error = self._clock(), time.time(), None
                return False

            full = Gallery.from_payload(self.django.embeddings_all())
            if session_ids:
                session = Gallery.from_payload(self.django.embeddings_for_sessions(session_ids))
            else:
                session = Gallery.empty(full.model, version)
            if session.model != full.model:
                raise GalleryError(f"session gallery model {session.model!r} != full gallery model {full.model!r}")
            self._current = Galleries(session, full, session_ids, version or full.version, self._clock())
            self.last_ok, self.last_ok_at, self.last_error = self._clock(), time.time(), None
            log.info("gallery loaded: version=%s sessions=%s session_members=%d all_members=%d",
                     self._current.version, ",".join(session_ids) or "-", len(session), len(full))
            return True
        except (httpx.HTTPError, GalleryError, ValueError, KeyError, TypeError) as exc:
            msg = f"{type(exc).__name__}: {exc}"
            if msg != self.last_error:
                log.warning("gallery refresh failed, keeping previous gallery: %s", msg)
            self.last_error = msg
            return False

    def run(self, stop):
        while not stop.is_set():
            self.refresh()
            stop.wait(self.poll_seconds)


def start_thread(target, *args, name):
    t = threading.Thread(target=target, args=args, name=name, daemon=True)
    t.start()
    return t
