# Face Recognition Attendance: recognition services

Two independent services that run on two different machines. Attendance is written **straight to an Ethereum smart contract**; the Django dashboard is only used to download the registered faces:

```
 camera ──► matcher_service ──HTTP──► embedding_service      (stateless: image in, embeddings out)
                  │ │
                  │ └──JSON-RPC──► Ethereum node (Ganache) ──► Attendance contract   (markAttendance(memberId))
                  │
                  └────HTTP──► Django dashboard               (gallery of members/embeddings, sessions, bans)
 Django admin registration camera ──HTTP──► embedding_service
```

| Service | What it does | Keeps state? |
|---|---|---|
| [`embedding_service/`](embedding_service/) | FastAPI. Finds faces with YuNet, returns 128-d SFace embeddings and a quality verdict. | No. Images are processed in memory only. |
| [`matcher_service/`](matcher_service/) | Worker. Reads a camera at `FPS`, gets embeddings, matches against the gallery downloaded from Django, votes over frames, writes attendance to the smart contract. | Only a local SQLite outbox of events not yet on chain. |

The wire formats are in [API_CONTRACT.md](API_CONTRACT.md); hand that same file to the Django side. [AUDIT.md](AUDIT.md) describes the code this replaced (the legacy prototype in `open_CV/` is kept until the new setup has been run on the real hardware).

Models (must be identical on every machine, including the dashboard): YuNet `face_detection_yunet_2023mar.onnx` + SFace `face_recognition_sface_2021dec.onnx`, model string `yunet2023mar+sface2021dec`, OpenCV `5.0.0` (`opencv-python-headless==5.0.0.93`). The `.onnx` files are never committed; `embedding_service/get_models.py` downloads them and checks their SHA-256.

Python 3.12 or newer.

---

## 1. Embedding service (machine A)

```bash
cd embedding_service
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python get_models.py                                 # -> ./models (or $MODEL_DIR)
cp .env.example .env                                 # set EMBED_API_TOKEN
export $(grep -v '^#' .env | xargs)                  # or use your process manager's env file
python -m app                                        # listens on $PORT (8000)
```

Or with Docker (models are downloaded during the build): `docker compose up -d --build embedding`.

| Env var | Default | Meaning |
|---|---|---|
| `EMBED_API_TOKEN` | – (**required**) | Shared secret; clients send `Authorization: Token <it>`. The service refuses to start without it unless `EMBED_ENV=dev`. |
| `EMBED_ENV` | `production` | `dev` allows running without a token (auth disabled). Never in production. |
| `PORT` | `8000` | Listen port. |
| `MODEL_DIR` | `embedding_service/models` | Where the two `.onnx` files are. |
| `MIN_FACE_PX` | `80` | Faces whose box's shorter side is smaller are flagged `face_too_small`. |
| `BLUR_THRESHOLD` | `60` | Laplacian variance of the 112×112 aligned face below this is flagged `blurry`. |
| `DETECT_SCORE_FLOOR` | `0.6` | Detector cut-off. Faces between this and `?min_score` are returned flagged `low_confidence`. |
| `MAX_REQUEST_BYTES` | `5242880` | Larger bodies get `413`. |
| `MAX_IMAGE_SIDE` | `2000` | Larger images are downscaled before detection. |
| `MAX_IMAGE_PIXELS` | `40000000` | Larger images (read from the header, before decoding) get `413`. |
| `MAX_BATCH_IMAGES` | `30` | Max images per `/v1/embed/batch`. |

Run a single uvicorn worker per container (the models are loaded once and inference is serialised by a lock); scale by running more containers behind a load balancer.

### curl examples

```bash
TOKEN=...; E=http://localhost:8000

curl $E/health
# {"status":"ok","model":"yunet2023mar+sface2021dec","dim":128}

curl -H "Authorization: Token $TOKEN" -F image=@face.jpg $E/v1/embed
# {"model":"yunet2023mar+sface2021dec","image":{"width":553,"height":555},
#  "faces":[{"box":[...],"score":0.93,"quality":{"ok":true,"reasons":[]},"embedding":[...128 floats...]}]}

curl -H "Authorization: Token $TOKEN" -F image=@empty_room.jpg $E/v1/embed     # -> "faces": []
curl -F image=@face.jpg $E/v1/embed                                            # -> 401

# JSON / base64 variant (a data: URL prefix is accepted)
curl -H "Authorization: Token $TOKEN" -H "Content-Type: application/json" \
     -d "{\"image_b64\":\"$(base64 -w0 face.jpg)\"}" "$E/v1/embed?max_faces=1&min_score=0.9"

# Registration batch (200 if >= min_usable usable frames, else 422 with the same body)
python - <<'PY' > batch.json
import base64, glob, json
print(json.dumps({"images_b64": [base64.b64encode(open(f, "rb").read()).decode() for f in sorted(glob.glob("frames/*.jpg"))],
                  "min_usable": 5}))
PY
curl -H "Authorization: Token $TOKEN" -H "Content-Type: application/json" -d @batch.json $E/v1/embed/batch
```

---

## 2. Matcher service (machine B, next to the camera)

```bash
cd matcher_service
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # set CAMERA_SOURCE, EMBED_URL/TOKEN, DJANGO_URL/TOKEN, CHAIN_* (section 3)
export $(grep -v '^#' .env | xargs)
python -m app
curl localhost:8090/status        # or open http://localhost:8090/ in a browser
```

Or `docker compose up -d --build matcher` (the outbox lives in the `matcher-data` volume; for a USB camera on Linux pass the device, see `docker-compose.yml`).

| Env var | Default | Meaning |
|---|---|---|
| `CAMERA_SOURCE` | `0` | Camera index, RTSP/HTTP stream URL (e.g. IP Webcam `http://phone:8080/video`) or a video file (looped; handy for testing). |
| `CAMERA_ID` | `camera-1` | Label sent to Django, e.g. `door-1`. |
| `EMBED_URL`, `EMBED_API_TOKEN` | – (required) | Embedding service. |
| `DJANGO_URL`, `DJANGO_API_TOKEN` | – (required) | Dashboard API (gallery and sessions). |
| `ATTENDANCE_SINK` | `chain` | `chain`: write attendance to the smart contract only (Django is **not** told). `django`: old behaviour, `POST /api/attendance/` only. `both`: write to the contract, then `POST /api/attendance/` with the real `tx_hash`/`block_number`; imposters and banned members, which the contract cannot store, go to Django directly. |
| `DJANGO_OUTBOX_PATH` | `matcher_outbox_django.sqlite3` | With `both`: separate outbox for the Django copies, so a Django outage never re-sends a chain transaction. |
| `CHAIN_RPC_URL` | – (required for `chain`) | Ethereum JSON-RPC endpoint, e.g. the Ganache dev tunnel `https://33wj9spz-8545.inc1.devtunnels.ms/`. |
| `CHAIN_CONTRACT_ADDRESS` | – | Address of the deployed `Attendance` contract. |
| `CHAIN_DEPLOYED_FILE` | – | Alternative to the address: path to the `deployed.json` written by `deploy.py`. One of the two is required. |
| `CHAIN_TUNNEL_TOKEN` | – | Only for a **private** Microsoft dev tunnel; sent as `X-Tunnel-Authorization: tunnel <token>`. |
| `CHAIN_PRIVATE_KEY` | – | Sign transactions locally with this key (needed on any real network). Must be the contract owner's key. |
| `CHAIN_FROM` | node's first account | Sender for unlocked-account nodes like Ganache, when no private key is set. |
| `CHAIN_RECEIPT_TIMEOUT` | `60` | Seconds to wait for a transaction to be mined before retrying. |
| `MATCH_THRESHOLD` | `0.363` | Minimum cosine similarity (SFace's published threshold). |
| `MATCH_MARGIN` | `0.05` | Best person must beat the second-best person by this much. |
| `FPS` | `4` | Frames analysed per second. |
| `GALLERY_REFRESH_SECONDS` | `60` | Full gallery re-download interval. |
| `SESSION_POLL_SECONDS` | `10` | How often `/api/sessions/active/` is polled to detect `gallery_version` or session changes (triggers an immediate re-download). |
| `MEMBER_COOLDOWN_SECONDS` | `300` | Same member is not re-sent within this time. |
| `IMPOSTER_COOLDOWN_SECONDS` | `20` | At most one imposter event per this time, per camera. |
| `FRAME_WIDTH` | `640` | Frames are downscaled to this width before being sent. |
| `OUTBOX_PATH` | `matcher_outbox.sqlite3` | Local queue for events not yet written to the chain (or acknowledged by Django). |
| `STATUS_HOST`, `STATUS_PORT` | `0.0.0.0`, `8090` | `GET /`: live dashboard (embedder / Django / blockchain connectivity, camera, gallery, outbox, counters). `GET /status`: the same as JSON (no identities, no images). |
| `HTTP_TIMEOUT` | `10` | Seconds, for the HTTP clients and each JSON-RPC request. |

How it decides (details in [API_CONTRACT.md §2.4](API_CONTRACT.md)):
1. Each good-quality face is compared with the **session gallery** (members of the running sessions + all banned members), then the **full gallery**; a match needs score ≥ threshold and a margin over the second-best person; otherwise it is an **imposter** (`member_id: null`). A face above the threshold but within the margin is ambiguous and is not reported.
2. Faces are tracked by box overlap; an identity is reported after winning 3 of the last 5 frames of that face.
3. Cooldowns (5 min per member, 20 s for imposters per camera). With the chain sink, **imposters and banned members are only logged** (and counted in `/status` as `not_written_imposter` / `not_written_banned`); the contract stores attendance only. Members go into the SQLite outbox.
4. A background thread writes each queued member to the contract with `markAttendance(memberId)`, oldest first, retrying with backoff (1 s … 60 s) while the node is unreachable. See section 3.
5. If the embedding service is down, the loop backs off and keeps retrying; if Django is down at gallery refresh time, the last gallery stays in use. Until a gallery has been loaded at least once, nothing is reported.
6. `app/hooks.py: on_recorded(record)` is called after each successful write with the `tx_hash` and `block_number` (e.g. to tell the dashboard later). It is a no-op by default.

---

## 3. Writing attendance to the blockchain

The matcher writes to the `Attendance` contract from the sibling `blockchain attendance` project (`contracts/attendance.sol`):

```solidity
function markAttendance(string memory memberID) public onlyOwner   // appends (sender, memberID, block.timestamp)
```

Setup:

1. Start Ganache and deploy the contract once: `cd "../blockchain attendance" && python deploy.py`. That writes `deployed.json`; put its `address` in `CHAIN_CONTRACT_ADDRESS`, or set `CHAIN_DEPLOYED_FILE` to the file. Redeploy (and update the address) whenever Ganache is restarted without a database, because the contract is gone.
2. Set `CHAIN_RPC_URL`. Ganache is currently exposed through the dev tunnel `https://33wj9spz-8545.inc1.devtunnels.ms/`. The tunnel must let the matcher in: either make port 8545 **public** (VS Code Ports view → right-click → Port Visibility → Public, or `devtunnel access create <tunnel-id> -p 8545 --anonymous`), or keep it private and set `CHAIN_TUNNEL_TOKEN` to the output of `devtunnel token <tunnel-id> --scopes connect` (tokens expire, so renew them). Without either, the node answers `401 Unauthorized` and events wait in the outbox.
3. Choose the sender. Only the contract **owner** (the deploying account) may write. On Ganache leave `CHAIN_PRIVATE_KEY` and `CHAIN_FROM` empty: the node's first unlocked account is used, the same one `deploy.py` used. On a real network set `CHAIN_PRIVATE_KEY` to the owner's key (keep it in `.env` only) and the transaction is signed inside the matcher.

At startup the matcher logs `chain: {chain_id, contract, sender, owner, records}`, and logs an error if the sender is not the owner. If the node is unreachable it only warns and keeps running.

Delivery guarantees:
- Each event's tx hash is saved in the outbox **before** waiting for the receipt. After a timeout, crash or restart, the matcher looks that transaction up first instead of sending the member again. If the node dropped it, or it reverted, it is sent again.
- Node down, tunnel `401`, a revert (e.g. the sender is not the owner) or a receipt timeout: the event stays pending and is retried with backoff. Nothing is dropped; `/status` shows `"delivery": {"sink": "chain", "error": ...}`.
- The record's time on chain is `block.timestamp`, the time it was mined, not `captured_at`. Events replayed after an outage get the replay time. The contract has no field for the capture time, camera or confidence; storing those would need a new contract version.
- Everything written to the chain is readable by anyone who can reach the node. Only opaque member ids are stored: no names, no embeddings.

Read the records back with the sibling project: `python demo.py`, or `AttendanceChain(url=...).get_records()`.

To go back to Django delivery, set `ATTENDANCE_SINK=django`; imposter and banned events are then sent to Django as before.

---

## 4. Developing without Django: the fake dashboard

`tests/fake_django.py` implements the contract's Django endpoints in memory (persisted to `fake_django_state.json` so it survives restarts), plus test-only `/fake/*` helpers.

```bash
# from the repo root, with the matcher dev requirements installed (fastapi + uvicorn)
export FAKE_DJANGO_TOKEN=dev-django-token
python -m uvicorn tests.fake_django:app --port 8001

# register people through the embedding service (camera frames or image files)
export EMBED_URL=http://localhost:8000 EMBED_API_TOKEN=... DJANGO_URL=http://localhost:8001 DJANGO_API_TOKEN=dev-django-token
python tests/register_face.py --member-id 202400004 --camera 0 --frames 14           # you
python tests/register_face.py --member-id 202400019 --images a.jpg b.jpg --min-usable 2 --banned

# open a session containing 202400004
curl -H "Authorization: Token dev-django-token" -H "Content-Type: application/json" \
     -d '{"id": 7, "name": "Week 45", "member_ids": ["202400004"], "active": true}' localhost:8001/fake/sessions

# watch what the matcher sent
curl -H "Authorization: Token dev-django-token" localhost:8001/fake/attendance
```

## 5. Tests and lint

```bash
pip install -r embedding_service/requirements-dev.txt -r matcher_service/requirements-dev.txt
flake8                                   # repo root, max line length 140 (legacy open_CV/ excluded)
(cd embedding_service && python get_models.py && pytest)   # uses the real models + tests/fixtures/face.jpg
(cd matcher_service && pytest)                             # synthetic vectors, respx mocks, fake Django, in-memory EVM
```
CI (`.github/workflows/ci.yml`) runs the same on pushes and PRs to `main` and `dev`.

## 6. Security and privacy

- **Put the services on a private network (VPN/VLAN) or behind HTTPS** (e.g. a reverse proxy with TLS in front of each service). Tokens and embeddings are otherwise sent in clear text. Biometric embeddings are personal data.
- Images exist only in memory: the embedding service never writes or logs them; the matcher never saves frames. Only embeddings leave the embedding service; only member ids, scores and timestamps leave the matcher.
- The Ganache dev tunnel exposes a node with **unlocked, funded accounts**: anyone who can reach a public tunnel can write to the contract as the owner. Prefer a private tunnel with `CHAIN_TUNNEL_TOKEN`, and never expose a real network's node or key this way.
- Tokens and the chain private key come from the environment only. Generate them with `python -c "import secrets; print(secrets.token_urlsafe(32))"`; nothing secret is committed (`.env` files are git-ignored; use the `.env.example` files).
- Camera URLs with credentials are masked in the matcher's logs.
