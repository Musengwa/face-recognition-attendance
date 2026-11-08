"""A small fake of the Django dashboard API (see API_CONTRACT.md section 2), for developing the matcher.

Run:   FAKE_DJANGO_TOKEN=dev-django-token python -m uvicorn tests.fake_django:app --port 8001
State (members, sessions, attendance) is kept in FAKE_DJANGO_STATE (default fake_django_state.json),
so killing and restarting it keeps registrations and shows which events arrived.

Contract endpoints:  GET /api/sessions/active/   GET /api/embeddings/   POST /api/attendance/
Test-only helpers:   /fake/members, /fake/sessions, /fake/attendance, /fake/reset
"""
import hmac
import json
import os
import threading
from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

MODEL = "yunet2023mar+sface2021dec"


def empty_state():
    return {"version": 1, "model": MODEL, "members": {}, "sessions": {}, "attendance": [], "events": []}


class Store:
    def __init__(self, path=None):
        self.path = path
        self.lock = threading.Lock()
        self.data = empty_state()
        if path and os.path.exists(path):
            with open(path) as f:
                self.data = json.load(f)

    def save(self):
        if self.path:
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.data, f)
            os.replace(tmp, self.path)

    def bump(self):
        self.data["version"] += 1


def create_app(token=None, state_path=None):
    token = token if token is not None else os.environ.get("FAKE_DJANGO_TOKEN", "dev-django-token")
    store = Store(state_path if state_path is not None else os.environ.get("FAKE_DJANGO_STATE", "fake_django_state.json"))
    app = FastAPI(title="Fake Django dashboard")
    app.state.store = store

    def auth(request: Request):
        scheme, _, given = request.headers.get("authorization", "").partition(" ")
        if scheme.lower() != "token" or not hmac.compare_digest(given.encode(), token.encode()):
            raise HTTPException(401, "invalid token")

    def active_sessions():
        return [s for s in store.data["sessions"].values() if s["active"]]

    # ---------------- contract endpoints ----------------

    @app.get("/api/sessions/active/", dependencies=[Depends(auth)])
    def sessions_active():
        with store.lock:
            return {
                "gallery_version": str(store.data["version"]),
                "sessions": [{"id": s["id"], "name": s["name"], "starts_at": s.get("starts_at"), "ends_at": s.get("ends_at")}
                             for s in active_sessions()],
            }

    @app.get("/api/embeddings/", dependencies=[Depends(auth)])
    def embeddings(session: str = None, scope: str = None):
        with store.lock:
            members = store.data["members"]
            if scope == "all":
                chosen = list(members.values())
            elif session:
                ids = {x.strip() for x in session.split(",") if x.strip()}
                wanted = set()
                for sid in ids:
                    s = store.data["sessions"].get(sid)
                    if s:
                        wanted.update(s["member_ids"])
                chosen = [m for m in members.values() if m["member_id"] in wanted or m["banned"]]
            else:
                raise HTTPException(400, "pass ?session=<ids> or ?scope=all")
            return {
                "model": store.data["model"],
                "gallery_version": str(store.data["version"]),
                "members": [{"member_id": m["member_id"], "banned": m["banned"], "embeddings": m["embeddings"]} for m in chosen],
            }

    @app.post("/api/attendance/", dependencies=[Depends(auth)])
    async def attendance(request: Request):
        body = await request.json()
        try:
            captured = datetime.fromisoformat(body["captured_at"])
            camera_id = str(body["camera_id"])
            float(body["confidence"])
        except (KeyError, TypeError, ValueError):
            return JSONResponse({"detail": "invalid payload"}, status_code=400)
        if captured.tzinfo is None:
            return JSONResponse({"detail": "captured_at needs a UTC offset"}, status_code=400)

        with store.lock:
            data = store.data
            event_id = body.get("event_id")
            if event_id and any(e.get("event_id") == event_id for e in data["events"]):
                return JSONResponse({"status": "already_seen"}, status_code=200)

            member_id = body.get("member_id")
            status, code, record = None, 201, None
            if member_id is None:
                status = "imposter"
            else:
                member = data["members"].get(str(member_id))
                if member is None:
                    return JSONResponse({"detail": f"unknown member {member_id}"}, status_code=400)
                if member["banned"]:
                    status = "banned"
                else:
                    sessions = [s for s in active_sessions() if member["member_id"] in s["member_ids"]]
                    session_id = sessions[0]["id"] if sessions else None
                    day = captured.date().isoformat()
                    dup = [a for a in data["attendance"] if a["member_id"] == member["member_id"]
                           and (a["session_id"] == session_id if session_id else (a["session_id"] is None and a["day"] == day))]
                    if dup:
                        status, code = ("already_marked" if session_id else "already_seen"), 200
                    else:
                        record = {"id": len(data["attendance"]) + 1, "member_id": member["member_id"], "session_id": session_id,
                                  "outside_session": session_id is None, "camera_id": camera_id,
                                  "captured_at": body["captured_at"], "day": day}
                        data["attendance"].append(record)
                        status = "recorded"
            data["events"].append({**body, "result": status, "received_at": datetime.now().astimezone().isoformat(timespec="seconds")})
            store.save()
        resp = {"status": status}
        if record:
            resp["record"] = record
            resp["outside_session"] = record["outside_session"]
        return JSONResponse(resp, status_code=code)

    # ---------------- test helpers (not part of the contract) ----------------

    @app.post("/fake/members", dependencies=[Depends(auth)])
    async def add_member(request: Request):
        body = await request.json()
        mid = str(body["member_id"])
        if body.get("model", MODEL) != store.data["model"]:
            raise HTTPException(400, "embeddings come from a different model")
        with store.lock:
            store.data["members"][mid] = {"member_id": mid, "name": body.get("name", mid), "banned": bool(body.get("banned")),
                                          "embeddings": body["embeddings"]}
            store.bump()
            store.save()
        return {"member_id": mid, "samples": len(body["embeddings"])}

    @app.post("/fake/members/{member_id}/ban", dependencies=[Depends(auth)])
    async def ban(member_id: str, request: Request):
        body = await request.json()
        with store.lock:
            if member_id not in store.data["members"]:
                raise HTTPException(404, "no such member")
            store.data["members"][member_id]["banned"] = bool(body.get("banned", True))
            store.bump()
            store.save()
        return {"member_id": member_id, "banned": store.data["members"][member_id]["banned"]}

    @app.post("/fake/sessions", dependencies=[Depends(auth)])
    async def put_session(request: Request):
        body = await request.json()
        sid = str(body["id"])
        with store.lock:
            store.data["sessions"][sid] = {"id": body["id"], "name": body.get("name", f"Session {sid}"),
                                           "member_ids": [str(m) for m in body.get("member_ids", [])],
                                           "active": bool(body.get("active", True)),
                                           "starts_at": body.get("starts_at"), "ends_at": body.get("ends_at")}
            store.bump()
            store.save()
        return store.data["sessions"][sid]

    @app.get("/fake/attendance", dependencies=[Depends(auth)])
    def list_attendance():
        with store.lock:
            return {"attendance": store.data["attendance"], "events": store.data["events"]}

    @app.post("/fake/reset", dependencies=[Depends(auth)])
    def reset():
        with store.lock:
            store.data = empty_state()
            store.save()
        return {"ok": True}

    return app


app = create_app()
