"""Run the matcher: `python -m app` (configuration from environment variables)."""
import logging
import signal
import threading

from .camera import Camera
from .clients import DjangoClient, EmbedClient
from .config import Settings
from .gallery import GalleryManager, start_thread
from .outbox import Outbox
from .sinks import DjangoSink
from .status import build_status, serve_status
from .worker import Stats, Worker, on_delivered

log = logging.getLogger("matcher")


def make_sink(s, django):
    if s.attendance_sink == "django":
        return DjangoSink(django)
    from .chain import ChainSink, load_contract_address, make_web3     # web3 is only needed for the chain
    w3 = make_web3(s.chain_rpc_url, timeout=s.http_timeout, tunnel_token=s.chain_tunnel_token or None)
    sink = ChainSink(w3, load_contract_address(s.chain_contract_address, s.chain_deployed_file),
                     private_key=s.chain_private_key or None, sender=s.chain_from or None,
                     receipt_timeout=s.chain_receipt_timeout)
    try:
        log.info("chain: %s", sink.check())
    except Exception as exc:       # not fatal: events wait in the outbox until the node is reachable
        log.warning("blockchain node not usable yet (%s: %s); events will queue in the outbox", type(exc).__name__, exc)
    return sink


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)      # one line per frame otherwise
    s = Settings.from_env()
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    stats = Stats()
    embed = EmbedClient(s.embed_url, s.embed_token, timeout=s.http_timeout)
    django = DjangoClient(s.django_url, s.django_token, timeout=s.http_timeout)
    sink = make_sink(s, django)
    outbox = Outbox(s.outbox_path, sink.deliver, on_delivered=on_delivered(stats))
    galleries = GalleryManager(django, s.gallery_refresh_seconds, s.session_poll_seconds)
    camera = Camera(s.camera_source)
    worker = Worker(s, embed, galleries, outbox, stats)

    try:
        info = embed.health()
        log.info("embedding service model: %s", info.get("model"))
    except Exception as exc:       # not fatal: the loop keeps retrying
        log.warning("embedding service not reachable yet: %s", exc)

    start_thread(galleries.run, stop, name="gallery")
    outbox_thread = start_thread(outbox.run, stop, name="outbox")
    start_thread(camera.run, stop, name="camera")
    server = serve_status(s.status_host, s.status_port, lambda: build_status(s, camera, galleries, outbox, stats))
    log.info("matcher running: camera=%s fps=%s attendance -> %s; status on :%d",
             s.camera_id, s.fps, sink.name, s.status_port)

    try:
        worker.run(camera, stop)
    finally:
        stop.set()
        server.shutdown()
        outbox_thread.join(timeout=s.http_timeout + 2)
        log.info("matcher stopped; outbox %s", outbox.counts())
        outbox.close()
        embed.close()
        django.close()


if __name__ == "__main__":
    main()
