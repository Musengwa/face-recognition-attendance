"""Camera reader that always holds the newest frame and reconnects on its own.

Network streams (RTSP / IP Webcam MJPEG) buffer frames; reading them at 4 FPS would give stale
pictures, so a thread reads continuously and keeps only the latest one. Video files are paced at
their own frame rate and loop, which makes them usable as a test camera. Frames stay in memory only.
"""
import logging
import os
import threading
import time
from datetime import datetime

import cv2

log = logging.getLogger("matcher.camera")


def parse_source(source):
    source = str(source).strip()
    return int(source) if source.isdigit() else source


class Camera:
    def __init__(self, source, max_backoff=30.0):
        self.source = parse_source(source)
        self.is_file = isinstance(self.source, str) and os.path.isfile(self.source)
        self.max_backoff = max_backoff
        self.connected = False
        self.last_error = None
        self._frame = None
        self._captured_at = None
        self._seq = 0
        self._lock = threading.Lock()

    def latest(self):
        """(frame, captured_at, seq) of the newest frame, or (None, None, seq) before the first one."""
        with self._lock:
            return self._frame, self._captured_at, self._seq

    def run(self, stop):
        backoff = 1.0
        while not stop.is_set():
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                self._set_down(f"cannot open camera source {self._label()}")
                stop.wait(backoff)
                backoff = min(self.max_backoff, backoff * 2)
                continue
            log.info("camera connected: %s", self._label())
            self.connected, self.last_error, backoff = True, None, 1.0
            delay = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 25.0) if self.is_file else 0.0
            while not stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    if self.is_file:
                        log.debug("video file ended; looping")
                    else:
                        self._set_down("camera read failed; reconnecting")
                    break
                with self._lock:
                    self._frame = frame
                    self._captured_at = datetime.now().astimezone()
                    self._seq += 1
                if delay:
                    time.sleep(delay)
            cap.release()
            if not self.is_file:
                stop.wait(backoff)

    def _set_down(self, msg):
        if msg != self.last_error:
            log.warning(msg)
        self.connected, self.last_error = False, msg

    def _label(self):
        # Do not log credentials embedded in stream URLs (rtsp://user:pass@host/...).
        if isinstance(self.source, str) and "@" in self.source and "://" in self.source:
            scheme, rest = self.source.split("://", 1)
            return f"{scheme}://***@{rest.split('@', 1)[1]}"
        return str(self.source)
