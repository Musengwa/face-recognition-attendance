"""Tiny read-only HTTP endpoint (no identities, no embeddings, no camera images):

    GET /         -> dashboard page (polls /status)
    GET /status   -> health, connections and counters as JSON (also /health)
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DASHBOARD = Path(__file__).with_name("dashboard.html")
STARTED_AT = time.time()


def _connections(settings, galleries, snap, health):
    probes = health.snapshot() if health is not None else {}
    embed = probes.get("embedder") or {"ok": snap["embed_ok"], "error": None, "detail": {}}
    if snap["model_mismatch"]:
        embed = {**embed, "ok": False, "error": "embedding model differs from the gallery model"}
    django = {
        "ok": None if galleries.last_error is None and getattr(galleries, "last_ok_at", None) is None
        else galleries.last_error is None,
        "error": galleries.last_error,
        "last_ok_at": getattr(galleries, "last_ok_at", None),
        "detail": {"url": settings.django_url},
    }
    embed["detail"] = {"url": settings.embed_url, **(embed.get("detail") or {})}
    out = {"embedder": embed, "django": django}
    if settings.attendance_sink in ("chain", "both"):
        chain = probes.get("chain") or {"ok": None, "error": None, "detail": {}}
        chain["detail"] = {"rpc_url": settings.chain_rpc_url, **(chain.get("detail") or {})}
        out["chain"] = chain
    return out


def build_status(settings, camera, galleries, outbox, stats, health=None, django_outbox=None):
    g = galleries.current
    snap = stats.snapshot()
    connections = _connections(settings, galleries, snap, health)
    healthy = bool(camera.connected and g is not None and not outbox.last_error
                   and not (django_outbox is not None and django_outbox.last_error)
                   and all(c["ok"] is not False for c in connections.values())
                   and snap["embed_ok"] is not False)
    return {
        "status": "ok" if healthy else "degraded",
        "uptime_seconds": round(time.time() - STARTED_AT),
        "camera": {"id": settings.camera_id, "connected": camera.connected, "error": camera.last_error},
        "connections": connections,
        "embedding_service": {"ok": snap["embed_ok"], "model_mismatch": snap["model_mismatch"]},
        "gallery": None if g is None else {
            "model": g.full.model, "version": g.version, "active_sessions": list(g.session_ids),
            "session_members": len(g.session), "all_members": len(g.full),
        },
        "django": {"gallery_error": galleries.last_error},
        "delivery": {"sink": settings.attendance_sink, "error": outbox.last_error,
                     "retry_backoff_seconds": outbox.backoff},
        "outbox": outbox.counts(),
        **({} if django_outbox is None else {
            "django_delivery": {"error": django_outbox.last_error, "retry_backoff_seconds": django_outbox.backoff},
            "django_outbox": django_outbox.counts()}),
        "last_frame_at": snap["last_frame_at"],
        "counters": snap["counters"],
    }


def serve_status(host, port, get_status):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/status", "/health"):
                self._send(json.dumps(get_status(), default=str).encode(), "application/json")
            elif path in ("/", "/index.html"):
                self._send(DASHBOARD.read_bytes(), "text/html; charset=utf-8")
            else:
                self.send_error(404)

        def _send(self, body, content_type):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):     # keep access logs out of the worker log
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, name="status", daemon=True).start()
    return server
