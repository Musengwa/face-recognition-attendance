import httpx
import numpy as np
import pytest
import respx
from conftest import MODEL, captured, face, gallery, member, toward, unit, with_

from app.clients import EmbedClient
from app.gallery import Galleries
from app.matching import Gallery
from app.worker import Worker, resize_to_width

A, B, BANNED = unit(1), unit(2), unit(3)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class ListOutbox:
    def __init__(self):
        self.events = []

    def put(self, e):
        self.events.append(e)


class StaticGalleries:
    def __init__(self, session, full):
        self.current = Galleries(session, full, ("7",), "1", 0.0)


def make_worker(settings, session=None, full=None, embed=None):
    full = full or gallery(member("A", A), member("B", B), member("X", BANNED, banned=True))
    session = session or gallery(member("A", A), member("X", BANNED, banned=True))
    clock, outbox = Clock(), ListOutbox()
    w = Worker(settings, embed, StaticGalleries(session, full), outbox, clock=clock)
    return w, clock, outbox


def frames(w, clock, faces_per_frame, n, start=0, model=MODEL, step=0.25):
    out = []
    for i in range(n):
        clock.t += step
        out += w.handle_embed_result({"model": model, "faces": faces_per_frame}, captured(start + i))
    return out


def test_registered_unregistered_and_banned(settings):
    w, clock, outbox = make_worker(settings)
    frames(w, clock, [face(toward(A, unit(9), 0.7), box=(0, 0, 100, 100)),
                      face(unit(50), box=(300, 0, 100, 100)),
                      face(toward(BANNED, unit(9), 0.8), box=(600, 0, 100, 100))], 3)
    got = {e["member_id"]: e for e in outbox.events}
    assert set(got) == {"A", None, "X"}
    e = got["A"]
    assert e["camera_id"] == "door-1" and e["captured_at"] == "2026-11-09T09:12:32+02:00"
    assert e["confidence"] == pytest.approx(0.7, abs=1e-3)
    assert e["tx_hash"] == "" and e["block_number"] is None and len(e["event_id"]) == 36
    assert 0.0 <= got[None]["confidence"] < settings.match_threshold


def test_chain_sink_queues_only_members_allowed_to_attend(settings):
    w, clock, outbox = make_worker(with_(settings, attendance_sink="chain"))
    frames(w, clock, [face(toward(A, unit(9), 0.7), box=(0, 0, 100, 100)),
                      face(unit(50), box=(300, 0, 100, 100)),
                      face(toward(BANNED, unit(9), 0.8), box=(600, 0, 100, 100))], 3)
    assert [e["member_id"] for e in outbox.events] == ["A"]
    assert w.stats.counters["not_written_imposter"] == 1 and w.stats.counters["not_written_banned"] == 1


def test_member_outside_session_is_reported_by_id(settings):
    w, clock, outbox = make_worker(settings)
    frames(w, clock, [face(toward(B, unit(9), 0.7))], 3)
    assert [e["member_id"] for e in outbox.events] == ["B"]


def test_needs_three_of_five_frames(settings):
    w, clock, outbox = make_worker(settings)
    frames(w, clock, [face(A)], 2)
    assert outbox.events == []
    frames(w, clock, [face(A)], 1)
    assert len(outbox.events) == 1


def test_low_quality_faces_are_not_matched(settings):
    w, clock, outbox = make_worker(settings)
    frames(w, clock, [face(A, ok=False)], 5)
    assert outbox.events == []
    assert w.stats.counters["faces_low_quality"] == 5


def test_member_cooldown_5_minutes(settings):
    w, clock, outbox = make_worker(settings)
    frames(w, clock, [face(A)], 3)
    clock.t += 5                      # track expires, the person walks in again
    frames(w, clock, [face(A)], 3)
    assert len(outbox.events) == 1
    assert w.stats.counters["suppressed_cooldown"] == 1
    clock.t += 300
    frames(w, clock, [face(A)], 3)
    assert len(outbox.events) == 2


def test_imposter_cooldown_20_seconds_per_camera(settings):
    w, clock, outbox = make_worker(settings)
    # two different unknown people at the same time -> one imposter event
    frames(w, clock, [face(unit(60), box=(0, 0, 100, 100)), face(unit(61), box=(400, 0, 100, 100))], 3)
    assert [e["member_id"] for e in outbox.events] == [None]
    clock.t += 10
    frames(w, clock, [face(unit(62))], 3)
    assert len(outbox.events) == 1
    clock.t += 15
    frames(w, clock, [face(unit(63), box=(200, 200, 90, 90))], 3)
    assert len(outbox.events) == 2


def test_imposter_cooldown_does_not_block_members(settings):
    w, clock, outbox = make_worker(settings)
    frames(w, clock, [face(unit(60), box=(0, 0, 100, 100)), face(A, box=(400, 0, 100, 100))], 3)
    assert sorted(e["member_id"] or "" for e in outbox.events) == ["", "A"]


def test_model_mismatch_refuses_to_match(settings):
    w, clock, outbox = make_worker(settings)
    frames(w, clock, [face(A)], 5, model="other-model")
    assert outbox.events == []
    assert w.stats.counters["skipped_model_mismatch"] == 5
    assert w.stats.state["model_mismatch"] == {"embedding_service": "other-model", "gallery": MODEL}


def test_gallery_from_other_model_refused(settings):
    other = gallery(member("A", A), model="yunet2022+sface2021dec")
    w, clock, outbox = make_worker(settings, session=Gallery.empty("yunet2022+sface2021dec"), full=other)
    frames(w, clock, [face(A)], 5)
    assert outbox.events == []


def test_no_gallery_loaded_yet_reports_nothing(settings):
    w, clock, outbox = make_worker(settings)
    w.galleries.current = None
    frames(w, clock, [face(unit(70))], 5)
    assert outbox.events == []      # must not call everyone an imposter before the gallery is known


def test_custom_thresholds(settings):
    w, clock, outbox = make_worker(with_(settings, match_threshold=0.8))
    frames(w, clock, [face(toward(A, unit(9), 0.7))], 3)
    assert [e["member_id"] for e in outbox.events] == [None]


def test_resize_to_width():
    assert resize_to_width(np.zeros((960, 1280, 3), np.uint8), 640).shape == (480, 640, 3)
    assert resize_to_width(np.zeros((240, 320, 3), np.uint8), 640).shape == (240, 320, 3)


@respx.mock
def test_process_frame_sends_downscaled_jpeg(settings):
    route = respx.post("http://embed.test/v1/embed").mock(
        return_value=httpx.Response(200, json={"model": MODEL, "image": {"width": 640, "height": 480}, "faces": [face(A)]}))
    w, clock, outbox = make_worker(settings, embed=EmbedClient("http://embed.test", "et"))
    frame = np.full((720, 1280, 3), 100, np.uint8)
    for i in range(3):
        clock.t += 0.25
        w.process_frame(frame, captured(i))
    req = route.calls[0].request
    assert req.headers["Authorization"] == "Token et"
    assert b"image/jpeg" in req.content and b"\xff\xd8\xff" in req.content
    assert [e["member_id"] for e in outbox.events] == ["A"]


@respx.mock
def test_embedding_service_down_raises_for_backoff(settings):
    respx.post("http://embed.test/v1/embed").mock(side_effect=httpx.ConnectError("down"))
    w, clock, outbox = make_worker(settings, embed=EmbedClient("http://embed.test", "et"))
    with pytest.raises(httpx.HTTPError):
        w.process_frame(np.zeros((480, 640, 3), np.uint8), captured())
    assert w.stats.state["embed_ok"] is False


def test_both_sink_sends_members_to_chain_and_imposters_banned_to_django(settings):
    full = gallery(member("A", A), member("B", B), member("X", BANNED, banned=True))
    session = gallery(member("A", A), member("X", BANNED, banned=True))
    clock, chain_box, django_box = Clock(), ListOutbox(), ListOutbox()
    w = Worker(with_(settings, attendance_sink="both"), None, StaticGalleries(session, full), chain_box,
               clock=clock, django_outbox=django_box)
    frames(w, clock, [face(toward(A, unit(9), 0.7), box=(0, 0, 100, 100)),
                      face(unit(50), box=(300, 0, 100, 100)),
                      face(toward(BANNED, unit(9), 0.8), box=(600, 0, 100, 100))], 3)
    assert [e["member_id"] for e in chain_box.events] == ["A"]
    assert sorted(e["member_id"] or "" for e in django_box.events) == ["", "X"]
