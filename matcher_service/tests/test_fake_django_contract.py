"""The matcher's clients against tests/fake_django.py (in-process), i.e. against API_CONTRACT.md section 2."""
import importlib.util
from pathlib import Path

import pytest
from conftest import MODEL, captured, unit
from fastapi.testclient import TestClient

from app.clients import DjangoClient
from app.gallery import GalleryManager
from app.outbox import Outbox
from app.sinks import DjangoSink

FAKE = Path(__file__).resolve().parents[2] / "tests" / "fake_django.py"
TOKEN = "dt"


def load_fake():
    spec = importlib.util.spec_from_file_location("fake_django", FAKE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def django(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_DJANGO_STATE", str(tmp_path / "unused.json"))
    app = load_fake().create_app(token=TOKEN, state_path="")
    tc = TestClient(app, headers={"Authorization": f"Token {TOKEN}"})
    tc.post("/fake/members", json={"member_id": "202400004", "embeddings": [list(unit(1))]})
    tc.post("/fake/members", json={"member_id": "202400005", "embeddings": [list(unit(2))]})
    tc.post("/fake/members", json={"member_id": "202400019", "banned": True, "embeddings": [list(unit(3))]})
    tc.post("/fake/sessions", json={"id": 7, "member_ids": ["202400004"], "active": True})
    client = DjangoClient("http://testserver", TOKEN, transport=tc._transport)
    return tc, client


def ev(member_id, i=0, event_id=None):
    return {"member_id": member_id, "confidence": 0.7, "camera_id": "door-1", "captured_at": captured(i).isoformat(),
            "tx_hash": "", "block_number": None, "event_id": event_id or f"ev-{member_id}-{i}"}


def test_wrong_token_is_401(django):
    tc, _ = django
    assert tc.get("/api/sessions/active/", headers={"Authorization": "Token nope"}).status_code == 401


def test_gallery_manager_against_fake(django):
    tc, client = django
    mgr = GalleryManager(client)
    assert mgr.refresh() is True
    g = mgr.current
    assert g.full.model == MODEL
    assert set(g.session.member_ids) == {"202400004", "202400019"}      # session member + banned
    assert set(g.full.member_ids) == {"202400004", "202400005", "202400019"}
    assert g.session_ids == ("7",)


def test_attendance_statuses(django):
    tc, client = django
    cases = [
        (ev("202400004"), 201, "recorded"),
        (ev("202400004", 1), 200, "already_marked"),
        (ev("202400005"), 201, "recorded"),            # outside session
        (ev("202400005", 2), 200, "already_seen"),
        (ev(None), 201, "imposter"),
        (ev("202400019"), 201, "banned"),
        (ev("202400019", 0), 200, "already_seen"),      # same event_id retried
    ]
    for payload, code, status in cases:
        r = client.post_attendance(payload)
        assert (r.status_code, r.json()["status"]) == (code, status), payload
    outside = client.post_attendance(ev("202400005", 9)).json()
    assert outside["status"] == "already_seen"
    rec = tc.get("/fake/attendance").json()["attendance"]
    assert [(a["member_id"], a["outside_session"]) for a in rec] == [("202400004", False), ("202400005", True)]


def test_outbox_delivers_to_fake_and_keeps_captured_at(django, tmp_path):
    tc, client = django
    box = Outbox(str(tmp_path / "o.sqlite3"), DjangoSink(client).deliver)
    box.put(ev("202400004", 5))
    assert box.deliver_due() == 1
    events = tc.get("/fake/attendance").json()["events"]
    assert events[-1]["captured_at"] == captured(5).isoformat() and events[-1]["result"] == "recorded"


def test_gallery_version_changes_on_ban(django):
    tc, client = django
    v1 = client.active_sessions()["gallery_version"]
    tc.post("/fake/members/202400005/ban", json={"banned": True})
    assert client.active_sessions()["gallery_version"] != v1
