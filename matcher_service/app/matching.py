"""Pure matching logic: galleries, per-person scores, threshold + margin, session-first decision."""
from dataclasses import dataclass

import numpy as np

IMPOSTER = "__imposter__"     # track label for "not in any gallery"; sent to Django as member_id null
DIM = 128


def normalise(vectors):
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


class GalleryError(ValueError):
    pass


@dataclass(frozen=True)
class Gallery:
    model: str
    version: str
    member_ids: tuple          # one per person
    banned: frozenset
    samples: np.ndarray        # (n_samples, 128), L2-normalised
    owner: np.ndarray          # (n_samples,), index into member_ids

    @classmethod
    def from_payload(cls, payload):
        """Build from the Django /api/embeddings/ response body. Raises GalleryError on bad data."""
        model = payload.get("model")
        if not isinstance(model, str) or not model:
            raise GalleryError("gallery response has no model")
        ids, banned, rows, owner = [], set(), [], []
        for member in payload.get("members") or []:
            mid = member.get("member_id")
            embs = member.get("embeddings") or []
            if mid is None or not embs:
                continue
            mid = str(mid)
            arr = np.asarray(embs, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != DIM or not np.isfinite(arr).all():
                raise GalleryError(f"member {mid}: embeddings must be lists of {DIM} finite floats")
            idx = len(ids)
            ids.append(mid)
            if member.get("banned"):
                banned.add(mid)
            rows.append(normalise(arr))
            owner.extend([idx] * len(arr))
        samples = np.vstack(rows) if rows else np.zeros((0, DIM), dtype=np.float32)
        return cls(model=model, version=str(payload.get("gallery_version", "")), member_ids=tuple(ids),
                   banned=frozenset(banned), samples=samples, owner=np.asarray(owner, dtype=np.int64))

    @classmethod
    def empty(cls, model, version=""):
        return cls.from_payload({"model": model, "gallery_version": version, "members": []})

    def __len__(self):
        return len(self.member_ids)

    def person_scores(self, embedding):
        """Cosine similarity per person = max over that person's samples."""
        if not len(self.member_ids):
            return np.zeros(0, dtype=np.float32)
        sims = self.samples @ normalise(embedding)[0]
        per = np.full(len(self.member_ids), -1.0, dtype=np.float32)
        np.maximum.at(per, self.owner, sims)
        return per


@dataclass(frozen=True)
class Match:
    kind: str                  # "match" | "ambiguous" | "none"
    member_id: str = None
    score: float = 0.0         # best person score (also set for "none": how close the nearest person was)
    second: float = -1.0


def match(embedding, gallery, threshold, margin):
    scores = gallery.person_scores(embedding)
    if scores.size == 0:
        return Match("none", score=0.0)
    order = np.argsort(scores)[::-1]
    best = float(scores[order[0]])
    second = float(scores[order[1]]) if scores.size > 1 else -1.0
    if best < threshold:
        return Match("none", score=best, second=second)
    if best - second < margin:
        return Match("ambiguous", member_id=gallery.member_ids[order[0]], score=best, second=second)
    return Match("match", member_id=gallery.member_ids[order[0]], score=best, second=second)


@dataclass(frozen=True)
class Decision:
    label: str                 # member id, IMPOSTER, or None (no decision: ambiguous)
    score: float
    source: str                # "session" | "full" | "imposter" | "ambiguous"
    banned: bool = False


def decide(embedding, session, full, threshold, margin):
    """Session gallery first, then the full gallery, else imposter.

    A face that is above the threshold but within the margin of another person (in the gallery
    that would decide it) is "ambiguous": it is not reported, because calling a registered
    member an imposter is worse than waiting for a clearer frame.
    """
    if session is not None and len(session):
        m = match(embedding, session, threshold, margin)
        if m.kind == "match":
            return Decision(m.member_id, m.score, "session", m.member_id in session.banned)
    m = match(embedding, full, threshold, margin)
    if m.kind == "match":
        return Decision(m.member_id, m.score, "full", m.member_id in full.banned)
    if m.kind == "ambiguous":
        return Decision(None, m.score, "ambiguous")
    return Decision(IMPOSTER, m.score, "imposter")
