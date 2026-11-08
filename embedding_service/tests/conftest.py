import base64
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

HERE = Path(__file__).parent
MODEL_DIR = os.environ.get("MODEL_DIR", str(HERE.parent / "models"))
TOKEN = "test-token"
AUTH = {"Authorization": f"Token {TOKEN}"}


def make_settings(**overrides):
    base = dict(
        api_token=TOKEN, env="test", model_dir=MODEL_DIR, min_face_px=80, blur_threshold=60.0,
        detect_floor=0.6, max_request_bytes=5 * 1024 * 1024, max_image_side=2000,
        max_image_pixels=40_000_000, max_batch_images=30, port=8000,
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture(scope="session")
def engine():
    from app.engine import FaceEngine
    if not os.path.isfile(os.path.join(MODEL_DIR, "face_recognition_sface_2021dec.onnx")):
        pytest.fail(f"models not found in {MODEL_DIR}; run `python get_models.py` first")
    return FaceEngine(MODEL_DIR, detect_floor=0.6)


@pytest.fixture(scope="session")
def client(engine):
    with TestClient(create_app(make_settings(), engine=engine)) as c:
        yield c


@pytest.fixture(scope="session")
def face_img():
    img = cv2.imread(str(HERE / "fixtures" / "face.jpg"))
    assert img is not None
    return img


def jpeg(img, quality=90):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return buf.tobytes()


def png(img):
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def b64(data, data_url=False):
    s = base64.b64encode(data).decode()
    return f"data:image/jpeg;base64,{s}" if data_url else s


def blank(w=640, h=480, value=128):
    return np.full((h, w, 3), value, dtype=np.uint8)
