# Audit of the recognition repo (before decoupling)

Date: 2026-11-08. Branch audited: `dev` at `1e0ad74`. Work happens on `feature/decouple-services`.

## 1. What exists today

| Concern | What is there | Files |
|---|---|---|
| Camera capture | `cv.VideoCapture(...)` on a webcam index or an IP Webcam MJPEG URL, read from `.env` (`WEBCAM`, `IP_WEBCAM`, `VIDEO_SOURCE`). Displays an annotated window with `cv.imshow`. | `open_CV/computer-vision/sface-fr.py` |
| Detection | OpenCV `cv.FaceDetectorYN` (YuNet), model `face_detection_yunet_2026may.onnx`, score threshold 0.9, NMS 0.3. | `open_CV/computer-vision/sface-fr.py`, `open_CV/computer-vision/models/` |
| Recognition | OpenCV `cv.FaceRecognizerSF` (SFace), model `face_recognition_sface_2021dec.onnx`. `alignCrop` + `feature`, cosine score, threshold 0.363, best single sample wins (no margin check). | `open_CV/computer-vision/sface-fr.py` |
| Identity storage | Identities are **file names** of reference photos in a folder (`person1a.jpeg` → `person1`), re-embedded at every start. Unknown faces are auto-enrolled as `unknown#N`, with their embeddings in `.npy` files **and a face crop saved as `.jpg`**. | `open_CV/computer-vision/face_logger.py`, `open_CV/computer-vision/tests/images/`, `open_CV/computer-vision/unknown_faces/` |
| Attendance storage | Local JSON file `{name: {first_seen, last_seen, count}}`, rewritten at most every 120 s per name. | `open_CV/computer-vision/attendance_log.json`, `face_logger.py` |
| Web server / API | None. | – |
| Blockchain | None in this repo. No `attendance_chain.py`, no `markAttendance`, no web3. (There is a separate sibling folder `../blockchain attendance` outside this repo; not touched.) | – |
| Older prototype | `face_recognition` (dlib, 128-d, Euclidean distance) attendance script writing names to a text file, plus OpenCV learning scripts. Not used by the current pipeline. | `open_CV/concept_study/**` |
| Tests | None (the `tests/images` folder only holds reference photos). | – |
| CI / lint config | None. | – |

### Problems found

1. **`sface-fr.py` does not run as committed**: it uses `getenv`, `KNOWN_DIR` and `COSINE_THRESHOLD`, none of which are defined or imported in that file (`NameError` at start-up).
2. **Wrong YuNet version for the dashboard**: it uses `yunet_2026may`; the dashboard uses `yunet_2023mar`. Embeddings come from SFace either way, but boxes/landmarks (and therefore the aligned crop) differ, so we standardise on `2023mar`.
3. **Model binaries are committed** (`*.onnx`, in two places) – must be downloaded by script instead.
4. **`.env` is committed** even though `.gitignore` lists it (it was added before the ignore rule). It contains the local IP Webcam URL. It will be removed from the index (the local file is kept); it remains in git history.
5. **Face images and biometric data are committed and written to disk at run time**: `unknown_faces/unknown#1.jpg` / `.npy`, reference photos, `__pycache__`. The new services must never write frames to disk.
6. Identity model (file name = person, auto-created `unknown#N`) does not match the dashboard (member ids, sessions, bans, imposters).
7. Recogniser and detector are module globals with no locking; fine for a single loop, not for an API server.

## 2. Detector / recogniser and identity format

- Detector: YuNet via `cv.FaceDetectorYN.create(model, "", (320, 320), 0.9, 0.3, 5000)`, `setInputSize` per frame.
- Recogniser: SFace via `cv.FaceRecognizerSF.create(model, "")`; `alignCrop(img, face)` then `feature()` → 1×128 float32 (not normalised).
- Identities: `{name: [feature, ...]}` built in memory from photo file names; unknowns persisted as `.npy` arrays named `unknown#N`.

So the **model family already matches the spec (YuNet + SFace through OpenCV)**. No Haar/LBPH/dlib is used by the current pipeline (dlib appears only in the legacy `concept_study` prototype). No second model will be supported.

## 3. Reuse vs replace

| Keep / reuse | Replace |
|---|---|
| `FaceDetectorYN` + `FaceRecognizerSF` usage pattern (`setInputSize`, `detect`, `alignCrop`, `feature`) → moves into `embedding_service/app/engine.py`. | YuNet `2026may` → `2023mar`, downloaded by `get_models.py`. |
| Cosine threshold 0.363 (SFace's published threshold) → matcher default. | "best single sample" matching → per-person max + `MATCH_MARGIN` check, session gallery first, then full gallery. |
| Camera source handling (index or IP Webcam URL) → matcher `CAMERA_SOURCE`. | File-name identities, `unknown#N` auto-enrolment, `.npy`/`.jpg` writes → gallery downloaded from Django; unknown faces reported as imposters (`member_id: null`); nothing saved to disk. |
| Idea of a per-person cooldown (120 s) → matcher cooldown (5 min, per spec). | `attendance_log.json` → `POST /api/attendance/` to Django with a SQLite retry queue. |
| Reference photos `tests/images/person1a.jpeg` → one is copied to `embedding_service/tests/fixtures/face.jpg` for tests. | `cv.imshow` UI window → headless worker with `/status` endpoint. |

## 4. Migration plan

1. Branch `feature/decouple-services`; add `.gitignore` rules for `*.onnx`, `.venv`, `*.sqlite3`, `__pycache__`; untrack `.env`.
2. `embedding_service/`: FastAPI app (stateless, token auth, `/health`, `/v1/embed`, `/v1/embed/batch`, quality checks, size limits, single model load + lock), `get_models.py`, Dockerfile, tests with real models.
3. `API_CONTRACT.md`: one contract for embedding service ↔ callers and matcher ↔ Django.
4. `tests/fake_django.py`: in-memory fake of the Django endpoints (sessions, embeddings, attendance with dedup / banned / imposter answers) plus admin helpers to register faces.
5. `matcher_service/`: camera loop, embedding client, gallery cache with versioning and model check, matcher (threshold + margin, session → fallback → imposter), IoU tracker with 3-of-5 voting, cooldowns, SQLite outbox with backoff, `on_recorded()` hook, `/status`.
6. Root: `docker-compose.yml`, `.flake8`, CI workflow, `README.md`, `.env.example` per service.
7. Verify end to end (embedding service + fake Django + matcher against webcam/IP camera or a video file, including a Django outage).
8. Only after that: retire `open_CV/` (left in place in this branch; deletion is a separate decision for the owner).
