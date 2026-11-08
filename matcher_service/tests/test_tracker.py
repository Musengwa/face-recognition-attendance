import pytest

from app.matching import IMPOSTER
from app.tracker import Tracker, iou

BOX = (100, 100, 120, 150)


def moved(box, dx):
    x, y, w, h = box
    return (x + dx, y, w, h)


def test_iou():
    assert iou(BOX, BOX) == 1.0
    assert iou(BOX, (500, 500, 10, 10)) == 0.0


def feed(tracker, labels, box=BOX, t0=0.0):
    out = []
    for i, label in enumerate(labels):
        out.extend(tracker.update([(moved(box, i * 3), label, 0.7)], t0 + i * 0.25))
    return out


def test_three_of_five_needed():
    t = Tracker()
    assert feed(t, ["A", "A"]) == []
    wins = feed(Tracker(), ["A", "A", "A"])
    assert [w.label for w in wins] == ["A"]


def test_flicker_does_not_report():
    assert feed(Tracker(), ["A", "B", "A", "B", None, "C", "A", "B"]) == []


def test_votes_only_count_within_last_five():
    # A appears 3 times, but never 3 times within any 5 consecutive frames.
    assert feed(Tracker(), ["A", None, None, None, "A", None, None, None, "A"]) == []
    wins = feed(Tracker(), ["A", None, "A", None, "A"])
    assert [w.label for w in wins] == ["A"]


def test_none_votes_never_win():
    assert feed(Tracker(), [None] * 6) == []


def test_imposter_needs_votes_too():
    assert feed(Tracker(), [IMPOSTER, "A", IMPOSTER]) == []
    wins = feed(Tracker(), [IMPOSTER, "A", IMPOSTER, IMPOSTER])
    assert [w.label for w in wins] == [IMPOSTER]


def test_reported_once_per_track():
    wins = feed(Tracker(), ["A"] * 10)
    assert len(wins) == 1
    assert wins[0].confidence == pytest.approx(0.7)


def test_two_faces_are_tracked_separately():
    t = Tracker()
    left, right = (50, 100, 100, 120), (400, 100, 100, 120)
    wins = []
    for i in range(3):
        wins += t.update([(left, "A", 0.6), (right, "B", 0.8)], i * 0.25)
    assert sorted(w.label for w in wins) == ["A", "B"]
    assert len({w.track_id for w in wins}) == 2


def test_track_expires_after_max_age():
    t = Tracker(max_age=2.0)
    t.update([(BOX, "A", 0.7)], 0.0)
    t.update([(BOX, "A", 0.7)], 0.25)
    assert t.update([(BOX, "A", 0.7)], 10.0) == []    # new track: only one vote
