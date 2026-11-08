# API contract

One contract for the three parties:

```
 camera ─► matcher_service ──(1) /v1/embed──────────► embedding_service ◄──(1)── Django admin
                 │                                                               registration camera
                 └──(2) /api/sessions, /api/embeddings, /api/attendance ──► Django dashboard
```

Version: 1.0 (2026-11-08). All bodies are JSON (UTF-8) unless stated. All timestamps are ISO 8601 **with a UTC offset**.

## 0. Model identity

| Item | Value |
|---|---|
| Detector | YuNet `face_detection_yunet_2023mar.onnx` via `cv2.FaceDetectorYN` |
| Recogniser | SFace `face_recognition_sface_2021dec.onnx` via `cv2.FaceRecognizerSF` (`alignCrop` then `feature`) |
| Model string | `"yunet2023mar+sface2021dec"` |
| Embedding | 128 floats, **L2-normalised**, compared with cosine similarity (= dot product) |

Every response that carries embeddings carries `model`. Embeddings with a different `model` string must never be compared. The embedding service is the only component that runs the models; Django should store embeddings it got from the embedding service (`/v1/embed/batch`) together with the `model` string.

---

## 1. Embedding service

Auth: every endpoint except `/health` needs `Authorization: Token <EMBED_API_TOKEN>`. Missing/wrong token → `401`.

Errors always have the shape `{"error": {"code": "<machine_code>", "message": "<human text>"}}` (except the batch `422`, see below).

### `GET /health`
`200 {"status": "ok", "model": "yunet2023mar+sface2021dec", "dim": 128}`

### `POST /v1/embed`
One image. Either:
- `multipart/form-data` with a file field `image` (JPEG or PNG), or
- `application/json` `{"image_b64": "<base64>"}`; a `data:image/jpeg;base64,` (or png) prefix is allowed.

Query: `max_faces` (1–50, default 5), `min_score` (0–1, default 0.9).

`200`:
```json
{
  "model": "yunet2023mar+sface2021dec",
  "image": {"width": 640, "height": 480},
  "faces": [
    {"box": [212, 96, 151, 190], "score": 0.9812,
     "quality": {"ok": true, "reasons": []},
     "embedding": [0.0123, -0.0841, "... 128 floats"]}
  ]
}
```
- `image` is the size of the image **as sent**; `box` (`[x, y, w, h]`, integers) is in those coordinates, even if the service downscaled internally.
- Faces are sorted by `score`, best first, at most `max_faces`.
- No face → `200` with `"faces": []`.
- Faces detected with a score below `min_score` are still returned, flagged `low_confidence`.

`quality.reasons` (any combination, in this order):

| reason | rule (env var, default) |
|---|---|
| `face_too_small` | shorter side of `box` < `MIN_FACE_PX` (80) px |
| `blurry` | variance of the Laplacian of the 112×112 aligned face crop < `BLUR_THRESHOLD` (60) |
| `low_confidence` | detector score < `min_score` |

Only use faces with `quality.ok == true` for matching or registration.

### `POST /v1/embed/batch`
For registration. Body: `{"images_b64": ["<base64>", ...], "min_usable": 5}` (`min_usable` optional, default 5; at most `MAX_BATCH_IMAGES` = 30 images). Query: `min_score` (default 0.9).

An image is **usable** only if it has exactly one face with score ≥ `min_score` and that face passes the quality check.

`200` when `usable >= min_usable`, otherwise **`422` with the same body**:
```json
{
  "model": "yunet2023mar+sface2021dec",
  "total": 14, "usable": 11, "min_usable": 5,
  "embeddings": [[128 floats], "..."],
  "rejected": [
    {"index": 3, "reason": "multiple_faces"},
    {"index": 7, "reason": "blurry", "reasons": ["blurry", "low_confidence"]},
    {"index": 9, "reason": "invalid_image", "detail": "invalid_base64"}
  ]
}
```
`reason` ∈ `no_face`, `multiple_faces`, `face_too_small`, `blurry`, `low_confidence`, `invalid_image`. `index` is the position in `images_b64`. `embeddings` are in input order of the usable images.

### Errors
| status | code | when |
|---|---|---|
| 400 | `unsupported_format`, `invalid_image`, `empty_image`, `invalid_base64`, `missing_image`, `invalid_json`, `invalid_request` | bad input |
| 401 | `unauthorized` | missing/invalid token |
| 413 | `request_too_large` | body > 5 MB (`MAX_REQUEST_BYTES`) |
| 413 | `image_too_large` | image > 40 MP (`MAX_IMAGE_PIXELS`), checked from the header before decoding |
| 413 | `too_many_images` | batch > `MAX_BATCH_IMAGES` |
| 415 | `unsupported_media_type` | neither multipart nor JSON |

Images larger than 2000 px on the longer side are downscaled before detection (not an error).

### Privacy
The service holds images only in memory for the duration of the request; it writes nothing to disk and never logs image data or embeddings.

---

## 2. Django dashboard API (used by the matcher)

Auth: `Authorization: Token <DJANGO_API_TOKEN>` on every call. Base URL = `DJANGO_URL` (e.g. `https://dashboard.example.org`). Paths keep Django's trailing slash.

### 2.1 `GET /api/sessions/active/`
Sessions running right now. The matcher polls this often (every `SESSION_POLL_SECONDS`, default 10 s), so keep it cheap.
```json
{
  "gallery_version": "42",
  "sessions": [
    {"id": 7, "name": "Week 45 – Tuesday", "starts_at": "2026-11-09T09:00:00+02:00", "ends_at": "2026-11-09T11:00:00+02:00"}
  ]
}
```
- `gallery_version`: opaque string that **changes whenever any registered embedding, membership of a session, or ban changes**. When it changes (or the set of active session ids changes) the matcher re-downloads its galleries immediately instead of waiting for `GALLERY_REFRESH_SECONDS`.
- `sessions` may be empty (no session running: every recognised member is "outside their session").

### 2.2 `GET /api/embeddings/?session=<id>,<id>`  and  `GET /api/embeddings/?scope=all`
- `?session=7,9`: the members of those sessions **plus every banned member** (whether or not they belong to the sessions).
- `?scope=all`: every registered member with embeddings (banned included).

```json
{
  "model": "yunet2023mar+sface2021dec",
  "gallery_version": "42",
  "members": [
    {"member_id": "202400004", "banned": false,
     "embeddings": [[128 floats], [128 floats]]},
    {"member_id": "202400019", "banned": true,
     "embeddings": [[128 floats]]}
  ]
}
```
- `member_id` is a **string**.
- Each member has 1..N sample embeddings, each 128 floats from the embedding service (the matcher re-normalises them, so small rounding is fine).
- `model` is required. If it differs from the embedding service's `model`, the matcher refuses to match and reports the mismatch on its `/status`.
- Names and other personal data are not needed and should not be sent.

### 2.3 `POST /api/attendance/`
One event per recognised person (or imposter).
```json
{
  "member_id": "202400004",
  "confidence": 0.71,
  "camera_id": "door-1",
  "captured_at": "2026-11-09T09:12:30+02:00",
  "tx_hash": "",
  "block_number": null,
  "event_id": "5f0c2a8e-3a59-4b5e-9f3e-0c1b7e1d2a44"
}
```
| field | meaning |
|---|---|
| `member_id` | string, or `null` = **imposter** (face not in any gallery) |
| `confidence` | best cosine similarity for that member (0–1). For an imposter: the best similarity found against anyone (always below the threshold) |
| `camera_id` | the matcher's `CAMERA_ID` |
| `captured_at` | when the frame was captured. **Not** when it was sent: events queued during an outage keep their original time |
| `tx_hash`, `block_number` | reserved for blockchain; currently always `""` / `null` |
| `event_id` | UUID generated by the matcher, unique per event and **identical across retries**. Django may use it as an idempotency key (recommended); it can be ignored safely |

The matcher does **not** decide whether a member is in the current session or banned: it sends what it saw and Django decides.

Responses (any 2xx = delivered; the matcher never resends it):

| status | `status` field | meaning |
|---|---|---|
| 201 | `recorded` | attendance created (in session, or "outside their session"; Django may add `"outside_session": true`) |
| 201 | `imposter` | imposter event stored (`member_id` was null) |
| 201 | `banned` | the member is banned; event stored, no attendance |
| 200 | `already_marked` | attendance already exists for this member and session |
| 200 | `already_seen` | duplicate within Django's own dedup window, or same `event_id` again |

Body: `{"status": "recorded", "record": {"id": 913, "member_id": "202400004", "session_id": 7, ...}}` (`record` optional; passed to the matcher's `on_recorded` hook).

Non-2xx:
- `400`/`422` (payload rejected): the matcher keeps the event in its local outbox as **dead** (not retried) and logs it, so a bad deploy does not lose data silently.
- `401`/`403`, `404`, `408`, `429`, `5xx`, network errors: retried with exponential backoff (1 s doubling up to 60 s), in order, until delivered.

### 2.4 Matcher-side rules (informative)
- Session gallery first; if no accepted match, the full gallery; if still no match, an imposter.
- Accept a match only if best person score ≥ `MATCH_THRESHOLD` (0.363) **and** best − second-best person ≥ `MATCH_MARGIN` (0.05). A person's score is the max over their samples.
- A face whose best score is above the threshold but within the margin of another person is **ambiguous**: it is not reported at all (neither as a member nor as an imposter).
- Report only after the same identity wins 3 of the last 5 frames of one tracked face.
- Cooldowns: the same member is not re-sent within 5 min; at most one imposter event per 20 s per camera. Django should still deduplicate.
