"""The recognition loop: frame -> embedding service -> match -> vote -> cooldown -> outbox (-> chain or Django)."""
import logging
import threading
import time
import uuid

import cv2
import httpx

from . import hooks
from .cooldown import Cooldowns
from .matching import IMPOSTER, decide
from .tracker import Tracker

log = logging.getLogger("matcher.worker")


class Stats:
    def __init__(self):
        self._lock = threading.Lock()
        self.counters = {k: 0 for k in (
            "frames", "embed_errors", "faces", "faces_low_quality", "votes_member", "votes_imposter",
            "votes_ambiguous", "reported_member", "reported_imposter", "suppressed_cooldown",
            "skipped_no_gallery", "skipped_model_mismatch", "not_written_imposter", "not_written_banned",
            "delivered", "recorded")}
        self.state = {"embed_ok": None, "model_mismatch": None, "last_frame_at": None}

    def inc(self, key, n=1):
        with self._lock:
            self.counters[key] += n

    def set(self, **kw):
        with self._lock:
            self.state.update(kw)

    def snapshot(self):
        with self._lock:
            return {"counters": dict(self.counters), **self.state}


def resize_to_width(frame, width):
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    return cv2.resize(frame, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)


class Worker:
    def __init__(self, settings, embed, galleries, outbox, stats=None, clock=time.monotonic):
        self.s = settings
        self.embed = embed
        self.galleries = galleries          # GalleryManager
        self.outbox = outbox
        self.stats = stats or Stats()
        self.clock = clock
        self.tracker = Tracker(window=settings.vote_window, needed=settings.votes_needed)
        self.cooldowns = Cooldowns()

    # ---------- one frame ----------
    def process_frame(self, frame, captured_at):
        """Raises httpx.HTTPError if the embedding service is unreachable (caller backs off)."""
        self.stats.inc("frames")
        small = resize_to_width(frame, self.s.frame_width)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, self.s.jpeg_quality])
        if not ok:
            return []
        try:
            result = self.embed.embed_jpeg(buf.tobytes())
        except httpx.HTTPError:
            self.stats.inc("embed_errors")
            self.stats.set(embed_ok=False)
            raise
        self.stats.set(embed_ok=True, last_frame_at=captured_at.isoformat(timespec="seconds"))
        return self.handle_embed_result(result, captured_at)

    def handle_embed_result(self, result, captured_at):
        g = self.galleries.current
        if g is None:
            self.stats.inc("skipped_no_gallery")
            return []
        if result.get("model") != g.full.model:
            self.stats.inc("skipped_model_mismatch")
            if not self.stats.state.get("model_mismatch"):
                log.error("model mismatch: embedding service %r vs gallery %r; refusing to match",
                          result.get("model"), g.full.model)
            self.stats.set(model_mismatch={"embedding_service": result.get("model"), "gallery": g.full.model})
            return []
        self.stats.set(model_mismatch=None)

        detections = []
        for face in result.get("faces", []):
            self.stats.inc("faces")
            if not face.get("quality", {}).get("ok"):
                self.stats.inc("faces_low_quality")
                detections.append((face["box"], None, 0.0))
                continue
            d = decide(face["embedding"], g.session, g.full, self.s.match_threshold, self.s.match_margin)
            self.stats.inc("votes_imposter" if d.label == IMPOSTER else "votes_ambiguous" if d.label is None else "votes_member")
            detections.append((face["box"], d.label, d.score))

        now = self.clock()
        events = []
        for win in self.tracker.update(detections, now):
            event = self._report(win, captured_at, now)
            if event:
                events.append(event)
        return events

    def _report(self, win, captured_at, now):
        if self.s.attendance_sink == "chain" and not self._chain_writable(win.label, now):
            return None
        if win.label == IMPOSTER:
            key, seconds, member_id = ("imposter", self.s.camera_id), self.s.imposter_cooldown_seconds, None
        else:
            key, seconds, member_id = ("member", win.label), self.s.member_cooldown_seconds, win.label
        if not self.cooldowns.allow(key, seconds, now):
            self.stats.inc("suppressed_cooldown")
            return None
        event = {
            "member_id": member_id,
            "confidence": round(max(0.0, min(1.0, win.confidence)), 4),
            "camera_id": self.s.camera_id,
            "captured_at": captured_at.isoformat(timespec="seconds"),
            "tx_hash": "",
            "block_number": None,
            "event_id": str(uuid.uuid4()),
        }
        self.outbox.put(event)
        self.stats.inc("reported_imposter" if member_id is None else "reported_member")
        log.info("queued %s (confidence %.3f) at %s", member_id or "IMPOSTER", event["confidence"], event["captured_at"])
        return event

    def _chain_writable(self, label, now):
        """The contract only stores attendance: imposters and banned members are logged, never written."""
        if label == IMPOSTER:
            if self.cooldowns.allow(("imposter", self.s.camera_id), self.s.imposter_cooldown_seconds, now):
                self.stats.inc("not_written_imposter")
                log.warning("IMPOSTER at camera %s (not written to the chain)", self.s.camera_id)
            return False
        g = self.galleries.current
        if g is not None and label in g.full.banned:
            if self.cooldowns.allow(("member", label), self.s.member_cooldown_seconds, now):
                self.stats.inc("not_written_banned")
                log.warning("banned member %s seen at camera %s (not written to the chain)", label, self.s.camera_id)
            return False
        return True

    # ---------- loop ----------
    def run(self, camera, stop):
        period = 1.0 / max(self.s.fps, 0.1)
        backoff, last_seq = 0.0, -1
        while not stop.is_set():
            started = time.monotonic()
            frame, captured_at, seq = camera.latest()
            if frame is not None and seq != last_seq:
                last_seq = seq
                try:
                    self.process_frame(frame, captured_at)
                    backoff = 0.0
                except httpx.HTTPError as exc:
                    backoff = min(30.0, max(1.0, backoff * 2))
                    log.warning("embedding service unavailable (%s); retrying in %.0fs", type(exc).__name__, backoff)
                    stop.wait(backoff)
                    continue
                except Exception:
                    log.exception("frame processing failed")
            stop.wait(max(0.0, period - (time.monotonic() - started)))


def on_delivered(stats):
    """Build the outbox callback: count deliveries and fire the hook for `recorded` results."""
    def callback(payload, result):
        stats.inc("delivered")
        if isinstance(result, dict) and result.get("status") == "recorded":
            stats.inc("recorded")
            if result.get("tx_hash"):
                log.info("on chain: %s tx=%s block=%s", payload.get("member_id"), result["tx_hash"], result.get("block_number"))
            hooks.on_recorded({"event": payload, "response": result})
    return callback
