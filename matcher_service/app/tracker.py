"""Flicker reduction: follow faces across frames by box overlap and vote on their identity."""
import itertools
from collections import Counter, deque
from dataclasses import dataclass, field


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    id: int
    box: tuple
    last_seen: float
    votes: deque               # (label or None, score)
    reported: set = field(default_factory=set)


@dataclass(frozen=True)
class Winner:
    label: str
    confidence: float
    track_id: int


class Tracker:
    """A label (member id or IMPOSTER) wins when it has >= `needed` of the last `window` votes of one track.

    `None` votes (low-quality face, ambiguous match) take up a slot in the window but never win.
    Each track reports a given label once.
    """

    def __init__(self, window=5, needed=3, min_iou=0.3, max_age=2.0):
        self.window, self.needed, self.min_iou, self.max_age = window, needed, min_iou, max_age
        self.tracks = []
        self._ids = itertools.count(1)

    def update(self, detections, now):
        """detections: list of (box, label_or_None, score). Returns the list of new Winners."""
        self.tracks = [t for t in self.tracks if now - t.last_seen <= self.max_age]
        pairs = sorted(
            ((iou(t.box, d[0]), ti, di) for ti, t in enumerate(self.tracks) for di, d in enumerate(detections)),
            reverse=True,
        )
        used_t, used_d, assigned = set(), set(), {}
        for overlap, ti, di in pairs:
            if overlap < self.min_iou:
                break
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            assigned[di] = self.tracks[ti]

        winners = []
        for di, (box, label, score) in enumerate(detections):
            track = assigned.get(di)
            if track is None:
                track = Track(next(self._ids), tuple(box), now, deque(maxlen=self.window))
                self.tracks.append(track)
            track.box, track.last_seen = tuple(box), now
            track.votes.append((label, score))
            winner = self._winner(track)
            if winner is not None:
                track.reported.add(winner.label)
                winners.append(winner)
        return winners

    def _winner(self, track):
        counts = Counter(label for label, _ in track.votes if label is not None)
        if not counts:
            return None
        label, n = counts.most_common(1)[0]
        if n < self.needed or label in track.reported:
            return None
        scores = [s for lab, s in track.votes if lab == label]
        return Winner(label, sum(scores) / len(scores), track.id)
