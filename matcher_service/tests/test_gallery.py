import httpx
import respx
from conftest import MODEL, member, unit

from app.clients import DjangoClient
from app.gallery import GalleryManager

URL = "http://django.test"


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def emb_payload(*members, version="1", model=MODEL):
    return {"model": model, "gallery_version": version, "members": list(members)}


def setup(version="1", sessions=({"id": 7},)):
    clock = Clock()
    mgr = GalleryManager(DjangoClient(URL, "dt"), refresh_seconds=60, poll_seconds=10, clock=clock)
    active = respx.get(f"{URL}/api/sessions/active/").mock(
        return_value=httpx.Response(200, json={"gallery_version": version, "sessions": list(sessions)}))

    def embeddings(request):
        if request.url.params.get("scope") == "all":
            return httpx.Response(200, json=emb_payload(member("A", unit(1)), member("B", unit(2)),
                                                        member("X", unit(3), banned=True), version=version))
        assert request.url.params["session"] == "7"
        return httpx.Response(200, json=emb_payload(member("A", unit(1)), member("X", unit(3), banned=True), version=version))

    emb = respx.get(f"{URL}/api/embeddings/").mock(side_effect=embeddings)
    return mgr, clock, active, emb


@respx.mock
def test_initial_load():
    mgr, clock, active, emb = setup()
    assert mgr.refresh() is True
    g = mgr.current
    assert g.session.member_ids == ("A", "X") and g.full.member_ids == ("A", "B", "X")
    assert g.full.banned == {"X"} and g.session_ids == ("7",)
    assert emb.call_count == 2
    assert active.calls[0].request.headers["Authorization"] == "Token dt"


@respx.mock
def test_no_reload_until_refresh_interval():
    mgr, clock, active, emb = setup()
    mgr.refresh()
    clock.t = 30
    assert mgr.refresh() is False and emb.call_count == 2
    clock.t = 61
    assert mgr.refresh() is True and emb.call_count == 4


@respx.mock
def test_version_change_reloads_immediately():
    mgr, clock, active, emb = setup()
    mgr.refresh()
    active.mock(return_value=httpx.Response(200, json={"gallery_version": "2", "sessions": [{"id": 7}]}))
    clock.t = 1
    assert mgr.refresh() is True and mgr.current.version == "2"


@respx.mock
def test_session_change_reloads_and_no_session_means_empty_session_gallery():
    mgr, clock, active, emb = setup()
    mgr.refresh()
    active.mock(return_value=httpx.Response(200, json={"gallery_version": "1", "sessions": []}))
    clock.t = 1
    assert mgr.refresh() is True
    assert len(mgr.current.session) == 0 and len(mgr.current.full) == 3


@respx.mock
def test_django_down_keeps_last_gallery():
    mgr, clock, active, emb = setup()
    mgr.refresh()
    before = mgr.current
    active.mock(side_effect=httpx.ConnectError("refused"))
    clock.t = 120
    assert mgr.refresh() is False
    assert mgr.current is before
    assert "ConnectError" in mgr.last_error


@respx.mock
def test_django_down_at_start_means_no_gallery():
    mgr, clock, active, emb = setup()
    active.mock(return_value=httpx.Response(503))
    assert mgr.refresh() is False and mgr.current is None


@respx.mock
def test_bad_gallery_payload_keeps_previous():
    mgr, clock, active, emb = setup()
    mgr.refresh()
    before = mgr.current
    emb.mock(side_effect=None, return_value=httpx.Response(200, json={"members": []}))   # no model
    assert mgr.refresh(force=True) is False and mgr.current is before
