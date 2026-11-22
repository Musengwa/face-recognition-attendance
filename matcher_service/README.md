# Matcher service

A long-running worker that sits next to the camera. It grabs frames, sends them to the [embedding service](../embedding_service/), compares each face with the registered members it downloads from the Django dashboard, and votes across several frames. It then writes attendance to the **`Attendance` smart contract** (or to Django, or to both).

Setup, Docker, the full list of environment variables and the blockchain setup are in the [root README](../README.md#2-matcher-service-machine-b-next-to-the-camera). Wire formats are in [API_CONTRACT.md](../API_CONTRACT.md).

---

## How it works

```
             ┌──────────── camera thread ────────────┐
 camera ───► │ Camera.run: read continuously, keep    │
             │ only the newest frame (auto-reconnect) │
             └───────────────┬────────────────────────┘
                             │ latest()  at FPS (default 4)
             ┌───────────────▼──────────── main thread: Worker.run ─────────────────────────┐
             │ resize to FRAME_WIDTH → JPEG → POST /v1/embed  (embedding service)           │
             │   for each face with quality.ok:                                             │
             │     decide(): session gallery → full gallery → IMPOSTER   (threshold+margin) │
             │ Tracker: follow faces by box overlap, label wins at 3 of last 5 votes        │
             │ Cooldowns: 5 min per member, 20 s per camera for imposters                   │
             │ → Outbox.put(event)   (fast local SQLite INSERT)                             │
             └───────────────┬──────────────────────────────────────────────────────────────┘
                             │
             ┌───────────────▼──────────── outbox thread ───────────────┐
             │ deliver oldest pending event → sink                       │
             │   ChainSink: markAttendance(memberId), wait for receipt   │
             │   DjangoSink: POST /api/attendance/                       │
             │ retry with backoff 1 s … 60 s; nothing is dropped         │
             └───────────────────────────────────────────────────────────┘

 gallery thread : poll /api/sessions/active/ every 10 s; re-download galleries when
                  gallery_version / sessions change, or every 60 s
 health thread  : probe embedder /health and the chain every 15 s
 status thread  : http://localhost:8090/  (dashboard)   /status  (JSON)
```

### Recognition step by step

1. **Frame.** `Camera` runs its own thread and always holds just the newest frame. Network streams buffer frames, so reading them slowly would give you stale pictures. `Worker.run` takes `camera.latest()` at `FPS` and skips frames it has already seen (`seq`).
2. **Embed.** The frame is downscaled to `FRAME_WIDTH` (640), JPEG-encoded and sent to `/v1/embed`. If the embedding service is unreachable, the loop backs off 1 s → 30 s and retries.
3. **Guards.** No frames are processed until a gallery has been loaded once. If the embedding service's `model` differs from the gallery's `model`, matching is refused and the problem shows on `/status`.
4. **Match** (`matching.decide`). Each `quality.ok` face is scored against every person: the score is the cosine similarity to that person's best sample.
   - The **session gallery** (members of the running sessions plus all banned members) is tried first, then the **full gallery**.
   - A **match** needs `best ≥ MATCH_THRESHOLD` (0.363) *and* `best − second ≥ MATCH_MARGIN` (0.05).
   - Above the threshold but inside the margin is **ambiguous**: it gets no label and is never reported, because calling a registered member an imposter is worse than waiting for a clearer frame.
   - Below the threshold is an **imposter**.
   - Low-quality faces vote `None`.
5. **Vote** (`Tracker`). Faces are linked across frames by box IoU (≥ 0.3). A track is dropped after 2 s unseen. A label wins once it holds 3 of the last 5 votes of a track, and each track reports a given label only once. This removes single-frame flicker.
6. **Cooldown** (`Cooldowns`). The same member is not re-queued within `MEMBER_COOLDOWN_SECONDS`. Imposters are limited to one event per `IMPOSTER_COOLDOWN_SECONDS` per camera.
7. **Routing by sink** (`ATTENDANCE_SINK`):
   - `chain` (default): only members are queued. The contract stores attendance only, so imposters and banned members are **logged** and counted (`not_written_imposter` / `not_written_banned`) but not written.
   - `django`: every event goes to `POST /api/attendance/`.
   - `both`: members go to the chain first. Once mined, the event is copied to a second outbox for Django with its real `tx_hash` and `block_number`. Imposters and banned members go straight to the Django outbox.
8. **Deliver** (`Outbox` → sink). This runs in the background, oldest event first, with exponential backoff. The camera loop never waits on the network.

### Event payload

```json
{"member_id": "202400004", "confidence": 0.6123, "camera_id": "door-1",
 "captured_at": "2026-11-22T09:14:03+02:00", "tx_hash": "", "block_number": null,
 "event_id": "uuid4"}
```
`member_id` is `null` for imposters. `event_id` lets Django deduplicate re-deliveries.

### Delivery guarantees (chain sink)

- The tx hash is **checkpointed in SQLite right after sending**, before waiting for the receipt. After a timeout, crash or restart, `ChainSink._resume` looks that transaction up instead of writing the member twice:
  - mined and successful: done
  - still pending: retry later
  - dropped by the node, or reverted: send again
- Node down, tunnel `401`, a revert (wrong sender) or a receipt timeout all raise `DeliveryRetry`. The event stays pending.
- Only an event that can never succeed (`DeliveryRejected`, e.g. no `member_id`, or a Django 4xx that is not retryable) is marked `dead`. It is kept in the outbox for inspection.
- The on-chain time is `block.timestamp` (when the transaction was mined), not `captured_at`.

---

## Files

```
matcher_service/
├── app/
│   ├── __main__.py     wiring: build clients/sink/outboxes, start threads, run the worker, clean shutdown
│   ├── config.py       Settings.from_env() + a tiny .env loader
│   ├── camera.py       Camera: newest-frame reader with reconnect, video-file looping
│   ├── clients.py      EmbedClient, DjangoClient (httpx)
│   ├── gallery.py      GalleryManager: keeps session + full galleries in sync with Django
│   ├── matching.py     Gallery, match(), decide(), pure numpy, no I/O
│   ├── tracker.py      Tracker: IoU tracking + majority voting
│   ├── cooldown.py     Cooldowns: per-key rate limiting
│   ├── worker.py       Worker (the recognition loop), Stats, on_delivered callback
│   ├── outbox.py       Outbox: durable SQLite queue + delivery thread
│   ├── sinks.py        DeliveryRetry / DeliveryRejected, DjangoSink
│   ├── chain.py        ChainSink: web3 writes to the Attendance contract
│   ├── health.py       Probe, HealthMonitor: embedder/chain connectivity
│   ├── status.py       build_status(), serve_status(): JSON + dashboard server
│   ├── dashboard.html  live status page (polls /status)
│   └── hooks.py        on_recorded(record) extension point (no-op)
├── tests/              pytest suite (see below)
├── Dockerfile          python:3.12-slim, outbox in the /data volume, non-root user
├── .env.example
├── requirements.txt    httpx, numpy, opencv-python-headless==5.0.0.93, web3
└── requirements-dev.txt
```

### `__main__.py`
- **`main()`** loads `.env` (real environment variables take precedence) and builds `Settings`, the clients, the sink and the outbox(es), `GalleryManager`, `Camera`, `Worker` and `HealthMonitor`. It then starts the gallery, outbox, camera, health and status threads and runs `worker.run()` on the main thread. SIGINT/SIGTERM set a shared `stop` event. On shutdown it joins the outbox thread and logs how many events are still pending.
- **`make_sink(settings, django)`** returns `DjangoSink` or `ChainSink`. For the chain it also runs `sink.check()` and logs the chain id, contract, sender, owner and record count, plus an error if the sender is not the owner. An unreachable node is only a warning: events wait in the outbox until it is back.

### `config.py`
- **`load_dotenv(path)`**: a minimal KEY=VALUE loader that never overrides variables already set.
- **`Settings`** (frozen dataclass) / **`Settings.from_env()`** validates `ATTENDANCE_SINK`. It requires `EMBED_*` and `DJANGO_*`, and for `chain`/`both` it also requires `CHAIN_RPC_URL` plus `CHAIN_CONTRACT_ADDRESS` or `CHAIN_DEPLOYED_FILE`. `vote_window` (5), `votes_needed` (3) and `jpeg_quality` (85) are code defaults, not env vars.

### `camera.py`
- **`parse_source(source)`** turns `"0"` into device index 0. A bare `http://host:port` (the IP Webcam app) gets `/video` appended.
- **`Camera.run(stop)`** opens `cv2.VideoCapture`, reads continuously and stores `(frame, captured_at, seq)` under a lock. It reconnects with backoff up to 30 s. Video files play at their own frame rate and loop, which makes them handy as a test camera.
- **`Camera.latest()`** returns the newest frame. **`_label()`** masks `user:pass@` in stream URLs before logging.

### `clients.py`
- **`EmbedClient`**: `health()`, `embed_jpeg(bytes)` (multipart `POST /v1/embed`).
- **`DjangoClient`**: `active_sessions()`, `embeddings_for_sessions(ids)`, `embeddings_all()`, `post_attendance(payload)`. Both clients send `Authorization: Token …` and accept an httpx `transport`, so tests can mock them.

### `gallery.py`
- **`Galleries`** is an immutable snapshot with `session`, `full`, `session_ids`, `version` and `loaded_at`.
- **`GalleryManager.refresh(force=False)`** polls `/api/sessions/active/`. It re-downloads `?scope=all` and `?session=…` only when forced, when the data is stale (`GALLERY_REFRESH_SECONDS`), or when `gallery_version` or the set of active sessions changed. It checks that both galleries use the same model and then **swaps `current` atomically**. On any error the previous gallery stays in use and the error is shown on `/status`.
- **`run(stop)`** loops every `SESSION_POLL_SECONDS`. **`start_thread(...)`** is a small helper that starts a daemon thread.

### `matching.py`
Pure functions with no I/O, so they are easy to unit-test.
- **`Gallery.from_payload(payload)`** builds the gallery from Django's `/api/embeddings/` response:
  - All samples are stacked into one `(n, 128)` L2-normalised matrix. An `owner` array maps each row to a person.
  - Wrong dimensions or non-finite values raise `GalleryError`.
  - Banned members are collected into a set.
- **`Gallery.person_scores(embedding)`** computes one matrix-vector product, then `np.maximum.at` to take each person's best sample score.
- **`match(embedding, gallery, threshold, margin)`** → `Match(kind="match"|"ambiguous"|"none", member_id, score, second)`.
- **`decide(embedding, session, full, threshold, margin)`** → `Decision(label, score, source, banned)`. It tries the session gallery, then the full gallery, then falls back to `IMPOSTER`. An ambiguous result gives `label=None`.
- **`IMPOSTER = "__imposter__"`** is the internal track label. It is sent as `member_id: null`.

### `tracker.py`
- **`iou(a, b)`**: overlap of two `[x, y, w, h]` boxes.
- **`Tracker.update(detections, now)`**: detections are `(box, label|None, score)`. It drops stale tracks and greedily pairs tracks with detections by highest IoU. Each track keeps a deque of its last `window` votes. It returns the new `Winner(label, confidence, track_id)` entries, where `confidence` is the mean score of the winning votes.

### `cooldown.py`
- **`Cooldowns.allow(key, seconds, now)`** is thread-safe. It returns `True` and starts the cooldown, or `False` while the key is still cooling down. Expired keys are pruned once the map holds more than 10,000 entries.

### `worker.py`
- **`Worker.run(camera, stop)`**: the paced main loop, with backoff when the embedding service is down.
- **`Worker.process_frame(frame, captured_at)`**: resize, encode, call the embedding service, then `handle_embed_result`.
- **`Worker.handle_embed_result(result, captured_at)`** runs the gallery/model guards, calls `decide` per face, runs `tracker.update`, then `_report` for each winner. Tests call it directly with synthetic results.
- **`Worker._report(win, …)`** applies the sink routing (`_chain_writable`, `_not_for_chain`) and the cooldowns, then builds the event and puts it into the right outbox.
- **`Stats`**: thread-safe counters (`frames`, `faces`, `votes_*`, `reported_*`, `suppressed_cooldown`, `delivered`, `recorded`, …) and state, shown on `/status`.
- **`on_delivered(stats, django_outbox)`** builds the outbox callback. It counts deliveries, forwards chain records to the Django outbox in `both` mode and calls `hooks.on_recorded`.

### `outbox.py`
- **`Outbox(path, send, on_delivered)`**: SQLite in WAL mode, with table `events(id, payload, state, attempts, last_error, created_at, progress)`. Older databases get the `progress` column added automatically.
- **`put(payload)`** does an INSERT and wakes the delivery thread.
- **`deliver_due()`** sends pending events oldest first:
  - success: notify, *then* delete. A crash in between means re-delivery, never a lost callback.
  - `DeliveryRetry`: back off (1 s → 60 s) and stop for now, which keeps events in order.
  - `DeliveryRejected`: mark the event `dead` and continue.
- **`_checkpoint(id, value)`** saves the sink's progress (the tx hash). **`counts()`** and **`pending()`** are for status and tests. **`run(stop)`** is the delivery thread loop.

### `sinks.py`
- **The sink contract:** `deliver(payload, progress=None, checkpoint=None) -> dict`. A sink either returns a result, raises **`DeliveryRetry`** (may succeed later), or raises **`DeliveryRejected`** (never will).
- **`DjangoSink`** treats 2xx as done. 5xx, network errors and `401/403/404/408/409/425/429` are retried. Any other 4xx is rejected.

### `chain.py`
- **`ATTENDANCE_ABI`**: the parts of the contract ABI the matcher uses (`markAttendance`, `getRecordCount`, `owner`, the `AttendanceMarked` event).
- **`load_contract_address(address, deployed_file)`**: the env address wins, otherwise the `address` field from `deployed.json`.
- **`make_web3(rpc_url, timeout, tunnel_token)`**: an HTTP provider. It adds `X-Tunnel-Authorization` for private Microsoft dev tunnels.
- **`ChainSink`**
  - **`check()`**: the contract exists, owner versus sender, record count.
  - **`deliver()`**: `_resume(progress)` if a tx hash was checkpointed, otherwise `_send` → `checkpoint(tx_hash)` → wait for the receipt → `_result`.
  - **`_send()`**: signs locally with `CHAIN_PRIVATE_KEY`, or uses `eth_sendTransaction` from `CHAIN_FROM` or the node's first unlocked account (Ganache).
  - **`_result()`** returns `{"status":"recorded","tx_hash","block_number"}`. A reverted transaction raises `DeliveryRetry`.

### `health.py`
- **`Probe(name, check)`** runs `check()` and records `ok`, `error`, `latency_ms`, `checked_at` and `last_ok_at`. It logs only when the state changes.
- **`HealthMonitor(probes, interval=15)`** runs all probes in a loop.
- **`embed_probe(embed)`** calls `/health`. **`chain_probe(sink)`** calls `sink.check()` and fails if the sender is not the owner. Django is not probed here: the gallery thread already reports its state.

### `status.py` / `dashboard.html`
- **`build_status(...)`** reports:
  - overall `ok`/`degraded`, uptime
  - camera, and connections (embedder, Django, chain)
  - gallery (model, version, active sessions, member counts)
  - delivery errors and backoff, outbox counts, counters
  
  It contains no identities, embeddings or images.
- **`serve_status(host, port, get_status)`** runs a stdlib `ThreadingHTTPServer`: `GET /` serves the dashboard page, `GET /status` and `GET /health` serve the JSON.

### `hooks.py`
- **`on_recorded(record)`** is called once per successfully recorded event, with `{"event": payload, "response": result}` (for the chain, the result includes `tx_hash` and `block_number`). It is a no-op by default and is meant for side effects. Exceptions are caught and logged, and the event is not re-sent.

---

## Threads at a glance

| Thread | Code | Shared state it touches |
|---|---|---|
| main | `Worker.run` | reads `camera.latest()`, `galleries.current`; writes `Outbox.put`, `Stats` |
| camera | `Camera.run` | newest frame (locked) |
| gallery | `GalleryManager.run` | swaps `galleries.current` (atomic reference swap) |
| outbox (+ outbox-django) | `Outbox.run` | SQLite (locked), `Stats`, hook |
| health | `HealthMonitor.run` | probe snapshots (locked) |
| status | `serve_status` | read-only snapshots |

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

No camera, models or network are needed. The suite uses synthetic vectors, `respx` HTTP mocks, a fake Django (`../tests/fake_django.py`) and an in-memory EVM.

| File | Covers |
|---|---|
| `test_matching.py` | gallery parsing/validation, threshold, margin, session-first decision, banned members |
| `test_tracker.py` | IoU tracking, 3-of-5 voting, one report per label per track, track expiry |
| `test_worker.py` | full frame → event path, cooldowns, model mismatch, no gallery, sink routing |
| `test_gallery.py` | refresh triggers (version/session change, staleness), keeping the old gallery on errors |
| `test_outbox.py` | ordering, original `captured_at`, 60 s backoff cap, dead events, survives restart, `401` retried, hook behaviour, chain → Django forwarding |
| `test_chain.py` | contract writes (unlocked account and private key), non-owner retried, imposter rejected, resume without double write, node down keeps queue, `deployed.json` |
| `test_status.py` | ok/degraded status, no identities exposed, dashboard page, failed probes, IP Webcam URL handling |
| `test_fake_django_contract.py` | the matcher's Django calls against the fake dashboard and `fixtures/attendance_contract.json` |

To run the matcher end to end without a real dashboard, see [root README §4](../README.md#4-developing-without-django-the-fake-dashboard).
