"""Tiny read-only HTTP endpoint: GET /status -> health and counters (no identities, no embeddings)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def build_status(settings, camera, galleries, outbox, stats):
    g = galleries.current
    snap = stats.snapshot()
    healthy = bool(camera.connected and snap["embed_ok"] and g is not None and not snap["model_mismatch"]
                   and not galleries.last_error and not outbox.last_error)
    return {
        "status": "ok" if healthy else "degraded",
        "camera": {"id": settings.camera_id, "connected": camera.connected, "error": camera.last_error},
        "embedding_service": {"ok": snap["embed_ok"], "model_mismatch": snap["model_mismatch"]},
        "gallery": None if g is None else {
            "model": g.full.model, "version": g.version, "active_sessions": list(g.session_ids),
            "session_members": len(g.session), "all_members": len(g.full),
        },
        "django": {"gallery_error": galleries.last_error},
        "delivery": {"sink": settings.attendance_sink, "error": outbox.last_error,
                     "retry_backoff_seconds": outbox.backoff},
        "outbox": outbox.counts(),
        "last_frame_at": snap["last_frame_at"],
        "counters": snap["counters"],
    }


def serve_status(host, port, get_status):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.split("?")[0] not in ("/status", "/health"):
                self.send_error(404)
                return
            body = json.dumps(get_status()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):     # keep access logs out of the worker log
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, name="status", daemon=True).start()
    return server
