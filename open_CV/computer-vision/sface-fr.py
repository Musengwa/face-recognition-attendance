import os
import re
import time
from datetime import datetime
from dotenv import load_dotenv

import cv2 as cv

from face_logger import FaceLogger

load_dotenv()

# Initialize detector and recognizer
detector = cv.FaceDetectorYN.create(
    'models/face_detection_yunet_2026may.onnx',
    "", (320, 320), 0.9, 0.3, 5000
)

recognizer = cv.FaceRecognizerSF.create(
    'models/face_recognition_sface_2021dec.onnx', ""
)


def get_features(img, face):
    aligned = recognizer.alignCrop(img, face)
    return recognizer.feature(aligned)


def load_known_people(folder):
    """Build {name: [features, ...]} from images. 'person1a.jpeg' -> 'person1'."""
    known = {}
    for file in sorted(os.listdir(folder)):
        img = cv.imread(os.path.join(folder, file))
        if img is None:
            continue
        detector.setInputSize((img.shape[1], img.shape[0]))
        _, faces = detector.detect(img)
        if faces is None:
            print(f"[skip] no face found in {file}")
            continue
        name = re.sub(r'[a-z]?$', '', os.path.splitext(file)[0])
        known.setdefault(name, []).append(get_features(img, faces[0]))
        print(f"[load] {file} -> {name}")
    return known


def identify(feature, known):
    """Return (best_name, best_score); name is 'Unknown' below threshold."""
    best_name, best_score = "Unknown", 0.0
    for name, feats in known.items():
        for f in feats:
            score = recognizer.match(feature, f, cv.FaceRecognizerSF_FR_COSINE)
            if score > best_score:
                best_name, best_score = name, score
    if best_score < COSINE_THRESHOLD:
        best_name = "Unknown"
    return best_name, best_score


known_people = load_known_people(KNOWN_DIR)
logger = FaceLogger()
print(f"Loaded {len(known_people)} people: {', '.join(known_people)}")

cap = cv.VideoCapture(getenv("WEBCAM") or getenv("IP_WEBCAM"))  # ip webcam or 0 for native webcam
if not cap.isOpened():
    raise SystemExit("Could not open webcam")

prev_time = time.perf_counter()
fps = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Detection timing
    t0 = time.perf_counter()
    detector.setInputSize((frame.shape[1], frame.shape[0]))
    _, faces = detector.detect(frame)
    detect_ms = (time.perf_counter() - t0) * 1000

    # Recognition timing
    t1 = time.perf_counter()
    if faces is not None:
        for face in faces:
            x, y, w, h = face[:4].astype(int)
            feature = get_features(frame, face)
            name, score = identify(feature, known_people)
            if name == "Unknown":
                name = logger.resolve_unknown(feature, frame, face)
                color = (0, 165, 255) if name != "Unknown" else (0, 0, 255)
            else:
                logger.record(name)
                color = (0, 200, 0)
            cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv.putText(frame, f"{name} ({score:.2f})", (x, y - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    recog_ms = (time.perf_counter() - t1) * 1000

    # FPS (smoothed)
    now = time.perf_counter()
    fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
    prev_time = now

    # Clock / measurement overlay
    n_faces = 0 if faces is None else len(faces)
    lines = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        f"FPS: {fps:.1f}",
        f"Detect: {detect_ms:.1f} ms",
        f"Recognize: {recog_ms:.1f} ms ({n_faces} face{'s' if n_faces != 1 else ''})",
    ]
    cv.rectangle(frame, (5, 5), (300, 15 + 22 * len(lines)), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        cv.putText(frame, text, (10, 25 + 22 * i),
                   cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    cv.imshow("Face Recognition", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
