# Embedding service

A small, stateless FastAPI service. You send it an image, and it returns every face it finds with a 128‑d embedding and a quality verdict. It is the **only** component that runs the face models. The matcher (live camera) and the Django dashboard (registration) both call it, so all embeddings come from byte-identical models.

- Detector: **YuNet** (`face_detection_yunet_2023mar.onnx`, via `cv2.FaceDetectorYN`)
- Recogniser: **SFace** (`face_recognition_sface_2021dec.onnx`, via `cv2.FaceRecognizerSF`)
- Model string: `yunet2023mar+sface2021dec`. Every response carries it, and embeddings with a different string must never be compared.

Setup, Docker, the full list of environment variables and curl examples are in the [root README](../README.md#1-embedding-service-machine-a). Exact wire formats are in [API_CONTRACT.md §1](../API_CONTRACT.md).

---

## How a request flows

```
HTTP request
  │
  ├─ BodyLimitMiddleware        reject > MAX_REQUEST_BYTES with 413 (with or without Content-Length)
  ├─ require_token              "Authorization: Token <EMBED_API_TOKEN>" (constant-time compare)
  │
  ├─ read_single_image / read_json      multipart field "image"  or  JSON "image_b64"
  │
  ├─ images.decode              sniff JPEG/PNG header → check pixel count → decode (reduced if huge)
  │                             → downscale so longest side ≤ MAX_IMAGE_SIDE
  │
  ├─ FaceEngine.analyze         (in a thread pool, under a lock)
  │     YuNet detect → sort by score → for each face: alignCrop (112×112) → SFace feature
  │     → L2-normalise → blur score → quality reasons → box mapped back to original coords
  │
  └─ JSON response              { model, image:{width,height}, faces:[{box, score, quality, embedding}] }
```

Images exist only in memory for the duration of the request. Nothing is written to disk, and image bytes and embeddings are never logged.

## Endpoints

| Method & path | Auth | Purpose |
|---|---|---|
| `GET /health` | no | `{"status":"ok","model":...,"dim":128}`. Used by Docker's healthcheck and the matcher's probe. |
| `POST /v1/embed` | yes | One image, all faces (`?max_faces=1..50`, default 5; `?min_score=0..1`, default 0.9). Used by the matcher for every frame. |
| `POST /v1/embed/batch` | yes | Registration: up to `MAX_BATCH_IMAGES` base64 images. Returns the embeddings of the *usable* ones, plus the reason each other image was rejected. `200` if `usable ≥ min_usable`, otherwise `422` with the same body. |

### Quality rules

Each face gets `quality.reasons` (an empty list means `ok: true`):

| Reason | Rule |
|---|---|
| `face_too_small` | Shorter side of the box (in original-image pixels) < `MIN_FACE_PX` (80) |
| `blurry` | Variance of the Laplacian of the **112×112 aligned crop** < `BLUR_THRESHOLD` (60). Measured on the aligned crop, so it does not depend on face size. |
| `low_confidence` | Detector score < `min_score`. The detector itself keeps faces down to `DETECT_SCORE_FLOOR` (0.6), so these faces are still returned, but flagged. |

A batch image counts as **usable** only if it has exactly one face with score ≥ `min_score` and that face passes all checks. Otherwise it is rejected as `no_face`, `multiple_faces`, `invalid_image`, or its first quality reason.

---

## Files

```
embedding_service/
├── app/
│   ├── __main__.py     entry point: `python -m app` → uvicorn, 1 worker, port $PORT
│   ├── config.py       Settings (from env) + model constants
│   ├── main.py         FastAPI app factory, middleware, auth, endpoints, error handling
│   ├── engine.py       FaceEngine: YuNet + SFace, quality checks
│   └── images.py       safe in-memory decoding of untrusted base64/JPEG/PNG
├── get_models.py       downloads both .onnx models and checks their SHA-256
├── models/             the .onnx files (not committed)
├── tests/
│   ├── conftest.py     fixtures: real engine, TestClient, face.jpg
│   ├── fixtures/face.jpg
│   └── test_api.py     end-to-end tests against the real models
├── Dockerfile          python:3.12-slim, models fetched at build time, non-root user
├── .env.example
├── requirements.txt    fastapi, uvicorn, python-multipart, opencv-python-headless==5.0.0.93, numpy
└── requirements-dev.txt  + pytest, httpx, flake8
```

### `app/config.py`
- **Constants:** `MODEL_NAME`, `EMBEDDING_DIM = 128`, `DETECTOR_FILE`, `RECOGNIZER_FILE`.
- **`Settings`** (frozen dataclass)
  - `Settings.from_env()` reads every env var, with defaults.
  - `validate()` **refuses to start** without `EMBED_API_TOKEN` unless `EMBED_ENV=dev`.
  - `dev_mode` property.

### `app/main.py`
- **`create_app(settings=None, engine=None)`** is the app factory. Tests inject a settings object and a pre-built engine. In production the `lifespan` handler loads `FaceEngine` once at startup. The factory also raises Starlette's multipart `spool_max_size` so uploads stay in memory instead of being spooled to a temp file.
- **`BodyLimitMiddleware`** is a raw ASGI middleware. It counts body bytes as they stream in, so a chunked request with no `Content-Length` is still capped. It then replays the buffered body to the app.
- **`require_token`** is a dependency on the `/v1` router. It uses `hmac.compare_digest` to avoid timing leaks. It is skipped only in dev mode with no token.
- **`run_engine(data, max_faces, min_score)`** chains decode → analyze and passes `scale = original_width / analysed_width` so boxes come back in the caller's coordinates.
- **`embed`** and **`embed_batch`** are the two endpoints. CPU work runs through `run_in_threadpool`, so the event loop is never blocked.
- **`run_batch`** applies the per-image "exactly one confident face, quality ok" rule and builds `embeddings` / `rejected`.
- **`read_single_image`** accepts multipart *or* JSON and returns `415` for anything else. `read_json` returns `400 invalid_json` for bodies that are not a JSON object.
- **Error handling:** `ApiError`, `ImageError` and `RequestValidationError` are all mapped to `{"error":{"code","message"}}`. The validation handler lists only the offending *field names* and never echoes input, which could be image data.

### `app/engine.py`
- **`Face`** dataclass: `box [x,y,w,h]`, `score`, `embedding` (128 float32, L2-normalised), `blur`, `reasons`. Its `ok` property is `not reasons`.
- **`FaceEngine(model_dir, detect_floor, ...)`** loads both ONNX models and fails early with a clear message if either file is missing. `cv2.FaceDetectorYN` keeps per-call state and is not thread-safe, so every call holds `self.lock`.
- **`analyze(img, *, max_faces, min_score, min_face_px, blur_threshold, scale)`**: `setInputSize` → `detect` → keep the best `max_faces` → for each face, `alignCrop` + `feature`.
- **`_build(...)`** L2-normalises the feature, computes the blur score on the aligned crop, applies the three quality rules, rescales the box to original coordinates and clips it to the image.

### `app/images.py`
The input is untrusted, so the header is checked before any pixels are decoded.
- **`b64_to_bytes(value)`** strips an optional `data:image/...;base64,` prefix and decodes strictly (`validate=True`).
- **`sniff(data)`** reads the format and dimensions from the header alone: the PNG `IHDR` chunk, or a walk over JPEG markers to the SOF segment. Anything other than JPEG or PNG (WebP, GIF, …) gets `unsupported_format`.
- **`decode(data, max_side, max_pixels)`** rejects images over `MAX_IMAGE_PIXELS` with `413` *before* decoding (decompression-bomb guard). For very large images it uses `cv2.IMREAD_REDUCED_COLOR_{2,4,8}` so libjpeg decodes at reduced size, then resizes so the longest side is ≤ `max_side`. It returns `(img, original_w, original_h)`.
- **`ImageError(code, message, status=400)`** carries the machine-readable error code.

### `get_models.py`
`python get_models.py [target_dir]` downloads both models from the OpenCV Zoo (with a fallback mirror). It verifies each file against a **pinned SHA-256**, writes it atomically (temp file → `os.replace`), and skips files that are already present and correct.

---

## Design notes

- **Stateless:** no database and no stored images, so you can scale by running more containers behind a load balancer.
- **One uvicorn worker per container.** The models are loaded once and inference is serialised by a lock. More workers in one container would only duplicate the models in memory.
- **Same models everywhere:** `opencv-python-headless==5.0.0.93` plus SHA-pinned ONNX files. An embedding from a different model or OpenCV build is not comparable.
- **Embeddings are L2-normalised,** so cosine similarity is just a dot product (that is what the matcher does).

## Tests

```bash
pip install -r requirements-dev.txt
python get_models.py      # tests use the real models
pytest
```

`tests/test_api.py` covers:
- auth, and dev mode without a token
- multipart, base64 and data-URL input
- JPEG and PNG giving matching embeddings
- no face, and the `max_faces` limit
- bad, truncated and WebP input
- size limits, with and without `Content-Length`
- downscaling and box coordinates
- the blurry, small and low-confidence flags
- batch acceptance and rejection, and `min_usable`
- nothing image-related appearing in the logs
