import numpy as np
import pytest
from conftest import MODEL, gallery, member, toward, unit

from app.matching import IMPOSTER, Gallery, GalleryError, decide, match

T, M = 0.363, 0.05


def test_threshold():
    a = unit(1)
    g = gallery(member("A", a), member("B", unit(2)))
    assert match(toward(a, unit(9), 0.40), g, T, M).kind == "match"
    below = match(toward(a, unit(9), 0.35), g, T, M)
    assert below.kind == "none" and below.score == pytest.approx(0.35, abs=1e-4)


def test_exact_threshold_is_accepted():
    a = unit(1)
    g = gallery(member("A", a))
    assert match(toward(a, unit(9), T + 1e-4), g, T, M).kind == "match"


def test_margin_against_second_person():
    a, b = unit(1), unit(2)
    probe = toward(a, unit(9), 0.60)
    # B has cosine 0.57 with the probe: within the margin -> ambiguous
    b_close = toward(probe, b, 0.57)
    m = match(probe, gallery(member("A", a), member("B", b_close)), T, M)
    assert m.kind == "ambiguous" and m.member_id == "A"
    # B at 0.50: margin 0.10 -> match
    b_far = toward(probe, b, 0.50)
    assert match(probe, gallery(member("A", a), member("B", b_far)), T, M).kind == "match"


def test_person_score_is_max_over_samples_and_own_samples_do_not_compete():
    a = unit(1)
    probe = toward(a, unit(9), 0.7)
    # Three near-identical samples of A must not make A "ambiguous with itself".
    g = gallery(member("A", toward(a, unit(3), 0.99), a, toward(a, unit(4), 0.2)), member("B", unit(2)))
    m = match(probe, g, T, M)
    assert m.kind == "match" and m.member_id == "A"
    assert m.score == pytest.approx(0.7, abs=1e-3)


def test_unnormalised_gallery_vectors_are_normalised():
    a = unit(1)
    g = gallery(member("A", a * 37.0))
    assert match(a, g, T, M).score == pytest.approx(1.0, abs=1e-5)


def test_session_first_then_full_then_imposter():
    a, b, c = unit(1), unit(2), unit(3)
    session = gallery(member("A", a))
    full = gallery(member("A", a), member("B", b), member("C", c))

    d = decide(toward(a, unit(9), 0.8), session, full, T, M)
    assert (d.label, d.source) == ("A", "session")

    d = decide(toward(b, unit(9), 0.8), session, full, T, M)       # registered, not in session
    assert (d.label, d.source) == ("B", "full")

    d = decide(unit(42), session, full, T, M)                        # nobody
    assert (d.label, d.source) == (IMPOSTER, "imposter")
    assert d.score < T


def test_empty_session_gallery_uses_full():
    a = unit(1)
    d = decide(a, Gallery.empty(MODEL), gallery(member("A", a)), T, M)
    assert (d.label, d.source) == ("A", "full")


def test_empty_full_gallery_means_imposter():
    d = decide(unit(1), Gallery.empty(MODEL), Gallery.empty(MODEL), T, M)
    assert d.label == IMPOSTER


def test_banned_member_in_session_gallery_is_reported_by_id():
    a, banned = unit(1), unit(2)
    session = gallery(member("A", a), member("X", banned, banned=True))
    full = gallery(member("A", a), member("X", banned, banned=True))
    d = decide(toward(banned, unit(9), 0.75), session, full, T, M)
    assert d.label == "X" and d.banned is True and d.source == "session"


def test_banned_member_only_in_full_gallery():
    banned = unit(2)
    d = decide(banned, gallery(member("A", unit(1))), gallery(member("A", unit(1)), member("X", banned, banned=True)), T, M)
    assert d.label == "X" and d.banned


def test_ambiguous_is_not_reported_as_imposter():
    a, b = unit(1), unit(2)
    probe = toward(a, unit(9), 0.6)
    full = gallery(member("A", a), member("B", toward(probe, b, 0.58)))
    d = decide(probe, Gallery.empty(MODEL), full, T, M)
    assert d.label is None and d.source == "ambiguous"


def test_session_ambiguous_but_full_gallery_decisive():
    a, b, c = unit(1), unit(2), unit(3)
    probe = toward(c, unit(9), 0.75)
    session = gallery(member("A", toward(probe, a, 0.50)), member("B", toward(probe, b, 0.49)))
    full = gallery(member("A", toward(probe, a, 0.50)), member("B", toward(probe, b, 0.49)), member("C", c))
    d = decide(probe, session, full, T, M)
    assert d.label == "C" and d.source == "full"


@pytest.mark.parametrize("payload", [
    {"members": []},                                                    # no model
    {"model": MODEL, "members": [{"member_id": "A", "embeddings": [[0.1] * 127]}]},
    {"model": MODEL, "members": [{"member_id": "A", "embeddings": [[float("nan")] * 128]}]},
])
def test_bad_gallery_payload(payload):
    with pytest.raises(GalleryError):
        Gallery.from_payload(payload)


def test_gallery_parse_skips_members_without_embeddings_and_stringifies_ids():
    g = Gallery.from_payload({"model": MODEL, "gallery_version": 7, "members": [
        {"member_id": 202400004, "embeddings": [list(unit(1))]},
        {"member_id": "B", "embeddings": []},
    ]})
    assert g.member_ids == ("202400004",) and g.version == "7"
    assert np.allclose(np.linalg.norm(g.samples, axis=1), 1.0)
