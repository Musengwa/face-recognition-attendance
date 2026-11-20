import json
import urllib.request

from conftest import gallery, member, unit

from app.gallery import Galleries
from app.status import build_status, serve_status
from app.worker import Stats


class Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def parts(gallery_error=None, delivery_error=None):
    g = gallery(member("A", unit(1)))
    stats = Stats()
    stats.set(embed_ok=True)
    camera = Obj(connected=True, last_error=None)
    galleries = Obj(current=Galleries(g, g, ("7",), "3", 0.0), last_error=gallery_error)
    outbox = Obj(last_error=delivery_error, backoff=0.0, counts=lambda: {"pending": 0, "dead": 0})
    return camera, galleries, outbox, stats


def test_ok_when_everything_is_up(settings):
    s = build_status(settings, *parts())
    assert s["status"] == "ok" and s["gallery"]["all_members"] == 1


def test_degraded_when_django_unreachable(settings):
    assert build_status(settings, *parts(gallery_error="ConnectError"))["status"] == "degraded"
    assert build_status(settings, *parts(delivery_error="HTTP 503"))["status"] == "degraded"


def test_status_endpoint_exposes_no_identities(settings):
    camera, galleries, outbox, stats = parts()
    server = serve_status("127.0.0.1", 0, lambda: build_status(settings, camera, galleries, outbox, stats))
    try:
        body = urllib.request.urlopen(f"http://127.0.0.1:{server.server_address[1]}/status", timeout=5).read()
    finally:
        server.shutdown()
    data = json.loads(body)
    assert data["status"] == "ok"
    assert b'"A"' not in body and b"embedding" not in body.replace(b"embedding_service", b"")


def test_dashboard_page_is_served(settings):
    camera, galleries, outbox, stats = parts()
    server = serve_status("127.0.0.1", 0, lambda: build_status(settings, camera, galleries, outbox, stats))
    try:
        resp = urllib.request.urlopen(f"http://127.0.0.1:{server.server_address[1]}/", timeout=5)
        body = resp.read()
    finally:
        server.shutdown()
    assert resp.headers["Content-Type"].startswith("text/html") and b"/status" in body


def test_connections_report_failed_probes(settings):
    from dataclasses import replace

    from app.health import HealthMonitor, Probe

    def down():
        raise ConnectionError("node offline")

    health = HealthMonitor([Probe("embedder", lambda: {"model": "m"}), Probe("chain", down)])
    for p in health.probes.values():
        p.run_once()
    s = build_status(replace(settings, attendance_sink="chain"), *parts(), health)
    assert s["connections"]["embedder"]["ok"] is True and s["connections"]["embedder"]["detail"]["model"] == "m"
    assert s["connections"]["chain"]["ok"] is False and "node offline" in s["connections"]["chain"]["error"]
    assert s["status"] == "degraded"


def test_bare_http_camera_url_gets_video_path():
    from app.camera import parse_source
    assert parse_source("http://10.0.0.5:8080") == "http://10.0.0.5:8080/video"
    assert parse_source("http://10.0.0.5:8080/shot.mjpg") == "http://10.0.0.5:8080/shot.mjpg"
    assert parse_source("1") == 1
