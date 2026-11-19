from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from app.config import Settings
from app.matching import Gallery

MODEL = "yunet2023mar+sface2021dec"
TZ = timezone(timedelta(hours=2))


def unit(seed):
    v = np.random.default_rng(seed).normal(size=128)
    return v / np.linalg.norm(v)


def toward(base, target, sim):
    """A unit vector with cosine `sim` to `base`, in the plane of base and target."""
    base = np.asarray(base, dtype=np.float64)
    ortho = np.asarray(target, dtype=np.float64) - np.dot(target, base) * base
    ortho /= np.linalg.norm(ortho)
    return sim * base + np.sqrt(1 - sim ** 2) * ortho


def member(mid, *vectors, banned=False):
    return {"member_id": mid, "banned": banned, "embeddings": [list(map(float, v)) for v in vectors]}


def gallery(*members, model=MODEL, version="1"):
    return Gallery.from_payload({"model": model, "gallery_version": version, "members": list(members)})


def face(embedding, box=(100, 100, 120, 150), ok=True):
    return {"box": list(box), "score": 0.98, "quality": {"ok": ok, "reasons": [] if ok else ["blurry"]},
            "embedding": [float(x) for x in embedding]}


def captured(seconds=0):
    return datetime(2026, 11, 9, 9, 12, 30, tzinfo=TZ) + timedelta(seconds=seconds)


@pytest.fixture
def settings(tmp_path):
    return Settings(camera_source="0", camera_id="door-1", embed_url="http://embed.test", embed_token="et",
                    django_url="http://django.test", django_token="dt", attendance_sink="django",
                    outbox_path=str(tmp_path / "outbox.sqlite3"))


def with_(settings, **kw):
    return replace(settings, **kw)
