"""YuNet detection + SFace embedding, loaded once and guarded by a lock."""
import os
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import DETECTOR_FILE, EMBEDDING_DIM, RECOGNIZER_FILE


@dataclass
class Face:
    box: list            # [x, y, w, h] in the caller's original image coordinates
    score: float
    embedding: np.ndarray  # (128,), L2-normalised float32
    blur: float
    reasons: list = field(default_factory=list)

    @property
    def ok(self):
        return not self.reasons


class FaceEngine:
    """Not thread-safe internally (cv2.FaceDetectorYN keeps per-call state), so every call holds `lock`."""

    def __init__(self, model_dir, detect_floor=0.6, nms_threshold=0.3, top_k=5000):
        det_path = os.path.join(model_dir, DETECTOR_FILE)
        rec_path = os.path.join(model_dir, RECOGNIZER_FILE)
        for path in (det_path, rec_path):
            if not os.path.isfile(path):
                raise RuntimeError(f"model file missing: {path} (run get_models.py)")
        self.detector = cv2.FaceDetectorYN.create(det_path, "", (320, 320), detect_floor, nms_threshold, top_k)
        self.recognizer = cv2.FaceRecognizerSF.create(rec_path, "")
        self.lock = threading.Lock()

    def analyze(self, img, *, max_faces, min_score, min_face_px, blur_threshold, scale=1.0):
        """Detect faces (best score first, at most `max_faces`) and embed each one.

        `scale` = original width / analysed width, when the caller downscaled the image first.
        Boxes and the face-size check use original-image pixels.
        """
        h, w = img.shape[:2]
        with self.lock:
            self.detector.setInputSize((w, h))
            _, raw = self.detector.detect(img)
            if raw is None:
                return []
            raw = sorted(raw, key=lambda r: float(r[-1]), reverse=True)[:max_faces]
            faces = []
            for row in raw:
                aligned = self.recognizer.alignCrop(img, row)
                feat = self.recognizer.feature(aligned).reshape(-1).astype(np.float32)
                faces.append(self._build(row, aligned, feat, w, h, scale, min_score, min_face_px, blur_threshold))
        return faces

    @staticmethod
    def _build(row, aligned, feat, img_w, img_h, scale, min_score, min_face_px, blur_threshold):
        assert feat.shape == (EMBEDDING_DIM,)
        feat = feat / max(float(np.linalg.norm(feat)), 1e-12)
        x, y, bw, bh = (float(v) * scale for v in row[:4])
        img_w, img_h = img_w * scale, img_h * scale
        score = float(row[-1])
        # Sharpness measured on the 112x112 aligned crop, so it does not depend on face size.
        blur = float(cv2.Laplacian(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())

        reasons = []
        if min(bw, bh) < min_face_px:
            reasons.append("face_too_small")
        if blur < blur_threshold:
            reasons.append("blurry")
        if score < min_score:
            reasons.append("low_confidence")

        x0, y0 = max(0.0, x), max(0.0, y)
        x1, y1 = min(img_w, x + bw), min(img_h, y + bh)
        box = [round(x0), round(y0), round(max(0.0, x1 - x0)), round(max(0.0, y1 - y0))]
        return Face(box=box, score=score, embedding=feat, blur=blur, reasons=reasons)
