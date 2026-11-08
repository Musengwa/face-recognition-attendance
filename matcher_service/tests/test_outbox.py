import json

import httpx
import respx
from conftest import captured

from app import hooks
from app.clients import DjangoClient
from app.outbox import Outbox
from app.worker import Stats, on_delivered

URL = "http://django.test/api/attendance/"


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def event(i, member_id="A"):
    return {"member_id": member_id, "confidence": 0.7, "camera_id": "door-1",
            "captured_at": captured(i).isoformat(), "tx_hash": "", "block_number": None, "event_id": f"e{i}"}


def make(tmp_path, stats=None):
    clock = Clock()
    stats = stats or Stats()
    box = Outbox(str(tmp_path / "o.sqlite3"), DjangoClient("http://django.test", "dt").post_attendance,
                 on_delivered=on_delivered(stats), clock=clock)
    return box, clock, stats


@respx.mock
def test_delivered_on_2xx(tmp_path):
    route = respx.post(URL).mock(return_value=httpx.Response(201, json={"status": "recorded"}))
    box, clock, stats = make(tmp_path)
    box.put(event(0))
    assert box.deliver_due() == 1
    assert box.counts() == {"pending": 0, "dead": 0}
    assert json.loads(route.calls[0].request.content) == event(0)
    assert route.calls[0].request.headers["Authorization"] == "Token dt"


@respx.mock
def test_django_down_queues_then_delivers_in_order_with_original_captured_at(tmp_path):
    route = respx.post(URL).mock(side_effect=httpx.ConnectError("refused"))
    box, clock, stats = make(tmp_path)
    for i in range(3):
        box.put(event(i, member_id=f"M{i}"))
    assert box.deliver_due() == 0
    assert box.counts()["pending"] == 3
    assert box.backoff == 1.0

    # still inside the backoff window: no attempt
    assert box.deliver_due() == 0 and route.call_count == 1
    clock.t = 1.0
    box.deliver_due()
    assert box.backoff == 2.0 and route.call_count == 2

    route.mock(side_effect=None, return_value=httpx.Response(200, json={"status": "already_marked"}))
    clock.t = 10.0
    assert box.deliver_due() == 3
    bodies = [json.loads(c.request.content) for c in route.calls[2:]]
    assert [b["member_id"] for b in bodies] == ["M0", "M1", "M2"]
    assert [b["captured_at"] for b in bodies] == [captured(i).isoformat() for i in range(3)]
    assert box.backoff == 0.0 and box.counts()["pending"] == 0


@respx.mock
def test_queue_survives_restart(tmp_path):
    route = respx.post(URL).mock(return_value=httpx.Response(503))
    box, clock, _ = make(tmp_path)
    box.put(event(0))
    box.deliver_due()
    box.close()
    route.mock(return_value=httpx.Response(201, json={"status": "imposter"}))
    box2, _, _ = make(tmp_path)
    assert box2.pending() == [event(0)]
    assert box2.deliver_due() == 1


@respx.mock
def test_backoff_caps_at_60s(tmp_path):
    respx.post(URL).mock(return_value=httpx.Response(500))
    box, clock, _ = make(tmp_path)
    box.put(event(0))
    for _ in range(10):
        clock.t += 100
        box.deliver_due()
    assert box.backoff == 60.0


@respx.mock
def test_rejected_payload_becomes_dead_and_does_not_block_queue(tmp_path):
    respx.post(URL).mock(side_effect=[httpx.Response(400, json={"detail": "bad"}), httpx.Response(201, json={"status": "recorded"})])
    box, clock, _ = make(tmp_path)
    box.put(event(0))
    box.put(event(1))
    assert box.deliver_due() == 1
    assert box.counts() == {"pending": 0, "dead": 1}


@respx.mock
def test_unauthorized_is_retried_not_dropped(tmp_path):
    respx.post(URL).mock(return_value=httpx.Response(401))
    box, clock, _ = make(tmp_path)
    box.put(event(0))
    box.deliver_due()
    assert box.counts() == {"pending": 1, "dead": 0}


@respx.mock
def test_on_recorded_hook_only_for_recorded(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(hooks, "on_recorded", lambda record: calls.append(record))
    respx.post(URL).mock(side_effect=[
        httpx.Response(201, json={"status": "recorded", "record": {"id": 1}}),
        httpx.Response(200, json={"status": "already_marked"}),
        httpx.Response(201, json={"status": "banned"}),
        httpx.Response(201, json={"status": "imposter"}),
    ])
    box, clock, stats = make(tmp_path)
    for i in range(4):
        box.put(event(i))
    assert box.deliver_due() == 4
    assert len(calls) == 1
    assert calls[0]["event"] == event(0) and calls[0]["response"]["record"] == {"id": 1}
    assert stats.counters["delivered"] == 4 and stats.counters["recorded"] == 1


@respx.mock
def test_hook_failure_does_not_resend(tmp_path, monkeypatch):
    def boom(record):
        raise RuntimeError("chain down")
    monkeypatch.setattr(hooks, "on_recorded", boom)
    route = respx.post(URL).mock(return_value=httpx.Response(201, json={"status": "recorded"}))
    box, clock, _ = make(tmp_path)
    box.put(event(0))
    assert box.deliver_due() == 1
    assert box.deliver_due() == 0 and route.call_count == 1


def test_put_does_not_touch_network(tmp_path):
    def never(payload):
        raise AssertionError("put() must not send")
    box = Outbox(str(tmp_path / "o.sqlite3"), never)
    box.put(event(0))
    assert box.counts()["pending"] == 1
