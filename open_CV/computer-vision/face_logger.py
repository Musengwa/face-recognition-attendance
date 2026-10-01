import json
import os
import time
from datetime import datetime

import cv2 as cv
import numpy as np

LOG_FILE = 'attendance_log.json'
UNKNOWN_DIR = 'unknown_faces'

LOG_INTERVAL = 120          # seconds before the same person's entry is updated again
COSINE_THRESHOLD = 0.363    # same threshold SFace uses for known people
CONFIRM_SECONDS = 1.5       # a new face must stay in view this long to become unknown#
CONFIRM_FRAMES = 10         # ...and be seen in at least this many frames
PENDING_TIMEOUT = 1.0       # drop a candidate if it disappears for this long
MAX_FEATURES = 15           # embeddings kept per unknown (different angles/lighting)


def cosine(a, b):
    a, b = a.ravel(), b.ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def best_match(feature, gallery):
    """gallery: {name: [features]} -> (name, score) or (None, 0)."""
    best_name, best_score = None, 0.0
    for name, feats in gallery.items():
        for f in feats:
            s = cosine(feature, f)
            if s > best_score:
                best_name, best_score = name, s
    return best_name, best_score


class FaceLogger:
    def __init__(self, log_file=LOG_FILE, unknown_dir=UNKNOWN_DIR):
        self.log_file = log_file
        self.unknown_dir = unknown_dir
        os.makedirs(unknown_dir, exist_ok=True)

        self.log = self._load_log()
        self.unknowns = self._load_unknowns()   # {"unknown#1": [features]}
        self.pending = []                       # unconfirmed new faces

    # ---------- persistence ----------
    def _load_log(self):
        if os.path.exists(self.log_file):
            with open(self.log_file) as f:
                return json.load(f)
        return {}

    def _save_log(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.log, f, indent=2)

    def _load_unknowns(self):
        unknowns = {}
        for file in os.listdir(self.unknown_dir):
            if file.endswith('.npy'):
                name = os.path.splitext(file)[0]
                unknowns[name] = list(np.load(os.path.join(self.unknown_dir, file)))
        return unknowns

    def _save_unknown(self, name, crop=None):
        np.save(os.path.join(self.unknown_dir, f"{name}.npy"),
                np.array(self.unknowns[name]))
        if crop is not None:
            cv.imwrite(os.path.join(self.unknown_dir, f"{name}.jpg"), crop)

    def _next_unknown_name(self):
        nums = [int(n.split('#')[1]) for n in self.unknowns if '#' in n]
        return f"unknown#{max(nums, default=0) + 1}"

    # ---------- logging ----------
    def record(self, name):
        """Log a sighting; only writes if LOG_INTERVAL passed since the last one."""
        now = datetime.now()
        entry = self.log.get(name)
        if entry:
            last = datetime.fromisoformat(entry['last_seen'])
            if (now - last).total_seconds() < LOG_INTERVAL:
                return
            entry['last_seen'] = now.isoformat(timespec='seconds')
            entry['count'] += 1
        else:
            stamp = now.isoformat(timespec='seconds')
            self.log[name] = {'first_seen': stamp, 'last_seen': stamp, 'count': 1}
        self._save_log()
        print(f"[log] {name} @ {now:%H:%M:%S}")

    # ---------- unknown handling ----------
    def resolve_unknown(self, feature, frame, face):
        """Map an unrecognised face to an existing unknown#, or track it as a
        candidate until it has been in view long enough to get a new unknown#.
        Returns the display name ('unknown#N' or 'Unknown' while pending)."""
        now = time.time()
        feature = feature.ravel().copy()

        # 1) Already a saved unknown?
        name, score = best_match(feature, self.unknowns)
        if name and score >= COSINE_THRESHOLD:
            feats = self.unknowns[name]
            if len(feats) < MAX_FEATURES and score < 0.7:   # learn new angles only
                feats.append(feature)
                self._save_unknown(name)
            self.record(name)
            return name

        # 2) Match against pending candidates (drop stale ones first)
        self.pending = [p for p in self.pending if now - p['last'] < PENDING_TIMEOUT]
        cand = None
        for p in self.pending:
            if max(cosine(feature, f) for f in p['features']) >= COSINE_THRESHOLD:
                cand = p
                break
        if cand is None:
            cand = {'features': [], 'first': now, 'last': now, 'frames': 0,
                    'crop': None, 'det_score': 0.0}
            self.pending.append(cand)

        cand['last'] = now
        cand['frames'] += 1
        if len(cand['features']) < MAX_FEATURES:
            cand['features'].append(feature)

        # keep the most confident crop as the saved image
        det_score = float(face[-1])
        if det_score >= cand['det_score']:
            cand['det_score'] = det_score
            cand['crop'] = self._crop(frame, face)

        # 3) Promote once seen long enough
        if now - cand['first'] >= CONFIRM_SECONDS and cand['frames'] >= CONFIRM_FRAMES:
            name = self._next_unknown_name()
            self.unknowns[name] = cand['features']
            self._save_unknown(name, cand['crop'])
            self.pending.remove(cand)
            print(f"[new] {name} saved to {self.unknown_dir}/")
            self.record(name)
            return name

        return "Unknown"

    @staticmethod
    def _crop(frame, face, margin=0.3):
        x, y, w, h = face[:4]
        mx, my = w * margin, h * margin
        x1, y1 = int(max(x - mx, 0)), int(max(y - my, 0))
        x2 = int(min(x + w + mx, frame.shape[1]))
        y2 = int(min(y + h + my, frame.shape[0]))
        return frame[y1:y2, x1:x2].copy()
