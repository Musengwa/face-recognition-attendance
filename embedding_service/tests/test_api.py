import cv2
import numpy as np
import pytest
from conftest import AUTH, TOKEN, b64, blank, jpeg, make_settings, png
from fastapi.testclient import TestClient

from app.config import MODEL_NAME, Settings
from app.main import create_app


# ---------- health & auth ----------

def test_health_needs_no_token(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model": MODEL_NAME, "dim": 128}


@pytest.mark.parametrize("headers", [
    {}, {"Authorization": "Token wrong"}, {"Authorization": f"Bearer {TOKEN}"}, {"Authorization": "Token"},
])
def test_embed_rejects_missing_or_bad_token(client, face_img, headers):
    r = client.post("/v1/embed", files={"image": ("f.jpg", jpeg(face_img), "image/jpeg")}, headers=headers)
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "unauthorized"


def test_batch_rejects_missing_token(client):
    r = client.post("/v1/embed/batch", json={"images_b64": ["x"]})
    assert r.status_code == 401


def test_refuses_to_start_without_token_outside_dev():
    with pytest.raises(RuntimeError, match="EMBED_API_TOKEN"):
        create_app(make_settings(api_token="", env="production"))


def test_dev_mode_without_token_allows_requests(engine, face_img):
    with TestClient(create_app(make_settings(api_token="", env="dev"), engine=engine)) as c:
        r = c.post("/v1/embed", files={"image": ("f.jpg", jpeg(face_img), "image/jpeg")})
    assert r.status_code == 200


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("MIN_FACE_PX", "100")
    monkeypatch.setenv("BLUR_THRESHOLD", "12.5")
    s = Settings.from_env()
    assert s.min_face_px == 100 and s.blur_threshold == 12.5


# ---------- single embed ----------

def assert_embedding(vec):
    assert len(vec) == 128
    assert all(isinstance(v, float) for v in vec)
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-4


def test_embed_face_multipart(client, face_img):
    r = client.post("/v1/embed", files={"image": ("f.jpg", jpeg(face_img), "image/jpeg")}, headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == MODEL_NAME
    assert body["image"] == {"width": face_img.shape[1], "height": face_img.shape[0]}
    assert len(body["faces"]) == 1
    face = body["faces"][0]
    assert face["quality"] == {"ok": True, "reasons": []}
    assert face["score"] >= 0.9
    assert len(face["box"]) == 4 and all(isinstance(v, int) for v in face["box"])
    assert_embedding(face["embedding"])


@pytest.mark.parametrize("data_url", [False, True])
def test_embed_face_json_base64(client, face_img, data_url):
    r = client.post("/v1/embed", json={"image_b64": b64(png(face_img), data_url=data_url)}, headers=AUTH)
    assert r.status_code == 200
    assert len(r.json()["faces"]) == 1


def test_same_face_jpeg_and_png_embeddings_agree(client, face_img):
    a = client.post("/v1/embed", files={"image": ("a.jpg", jpeg(face_img, 95), "image/jpeg")}, headers=AUTH).json()
    b = client.post("/v1/embed", files={"image": ("b.png", png(face_img), "image/png")}, headers=AUTH).json()
    sim = float(np.dot(a["faces"][0]["embedding"], b["faces"][0]["embedding"]))
    assert sim > 0.9


@pytest.mark.parametrize("img", [blank(), np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)])
def test_no_face_returns_empty_list(client, img):
    r = client.post("/v1/embed", files={"image": ("x.jpg", jpeg(img), "image/jpeg")}, headers=AUTH)
    assert r.status_code == 200
    assert r.json()["faces"] == []


def test_max_faces_limits_result(client, face_img):
    two = np.hstack([face_img, face_img])
    r = client.post("/v1/embed?max_faces=1", files={"image": ("x.jpg", jpeg(two), "image/jpeg")}, headers=AUTH)
    assert len(r.json()["faces"]) == 1
    r = client.post("/v1/embed", files={"image": ("x.jpg", jpeg(two), "image/jpeg")}, headers=AUTH)
    assert len(r.json()["faces"]) == 2


# ---------- bad input ----------

@pytest.mark.parametrize("payload, code", [
    (b"not an image at all", "unsupported_format"),
    (b"GIF89a" + b"\0" * 100, "unsupported_format"),
    (b"\xff\xd8\xff\xe0" + b"\0" * 50, "invalid_image"),     # JPEG magic, garbage after
    (b"", "empty_image"),
])
def test_bad_image_bytes_400(client, payload, code):
    r = client.post("/v1/embed", files={"image": ("x.jpg", payload, "image/jpeg")}, headers=AUTH)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == code


def test_truncated_jpeg_400(client, face_img):
    data = jpeg(face_img)[:300]
    r = client.post("/v1/embed", files={"image": ("x.jpg", data, "image/jpeg")}, headers=AUTH)
    assert r.status_code == 400


def test_bad_base64_400(client):
    r = client.post("/v1/embed", json={"image_b64": "@@@not base64@@@"}, headers=AUTH)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_base64"


def test_missing_field_and_wrong_content_type(client):
    assert client.post("/v1/embed", json={"img": "x"}, headers=AUTH).status_code == 400
    assert client.post("/v1/embed", content=b"abc", headers={**AUTH, "Content-Type": "text/plain"}).status_code == 415
    assert client.post("/v1/embed", content=b"{not json", headers={**AUTH, "Content-Type": "application/json"}).status_code == 400


def test_invalid_query_param_400_without_echo(client, face_img):
    r = client.post("/v1/embed?max_faces=0", files={"image": ("x.jpg", jpeg(face_img), "image/jpeg")}, headers=AUTH)
    assert r.status_code == 400
    assert "max_faces" in r.json()["error"]["message"]


# ---------- size limits ----------

def test_oversize_body_413(client):
    big = b"\xff\xd8\xff" + b"\0" * (5 * 1024 * 1024 + 10)
    r = client.post("/v1/embed", files={"image": ("big.jpg", big, "image/jpeg")}, headers=AUTH)
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "request_too_large"


def test_oversize_body_413_without_content_length(client):
    def gen():
        for _ in range(6):
            yield b"\0" * (1024 * 1024)
    r = client.post("/v1/embed", content=gen(), headers={**AUTH, "Content-Type": "application/json"})
    assert r.status_code == 413


def test_too_many_pixels_413(client):
    # A tiny PNG header claiming 50000x50000 pixels: rejected before any decoding.
    import struct
    import zlib
    ihdr = struct.pack(">IIBBBBB", 50000, 50000, 8, 2, 0, 0, 0)
    chunk = b"IHDR" + ihdr
    data = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", len(ihdr)) + chunk + struct.pack(">I", zlib.crc32(chunk))
    r = client.post("/v1/embed", files={"image": ("x.png", data, "image/png")}, headers=AUTH)
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "image_too_large"


def test_large_image_is_downscaled_and_boxes_use_original_coordinates(client, face_img):
    big = cv2.resize(face_img, (face_img.shape[1] * 5, face_img.shape[0] * 5))   # ~2765 px, above 2000
    r = client.post("/v1/embed", files={"image": ("x.jpg", jpeg(big, 70), "image/jpeg")}, headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["image"] == {"width": big.shape[1], "height": big.shape[0]}
    x, y, w, h = body["faces"][0]["box"]
    assert w > 600 and x + w <= big.shape[1]


# ---------- quality ----------

def test_blurry_face_rejected(client, face_img):
    blurred = cv2.GaussianBlur(face_img, (0, 0), 6)
    r = client.post("/v1/embed", files={"image": ("x.jpg", jpeg(blurred), "image/jpeg")}, headers=AUTH)
    faces = r.json()["faces"]
    assert len(faces) == 1
    assert faces[0]["quality"]["ok"] is False
    assert "blurry" in faces[0]["quality"]["reasons"]


def test_small_face_rejected(client, face_img):
    canvas = blank(640, 480, 200)
    small = cv2.resize(face_img, (120, 120), interpolation=cv2.INTER_AREA)   # face box ~70 px
    canvas[100:220, 100:220] = small
    r = client.post("/v1/embed", files={"image": ("x.png", png(canvas), "image/png")}, headers=AUTH)
    faces = r.json()["faces"]
    assert len(faces) == 1
    assert "face_too_small" in faces[0]["quality"]["reasons"]


def test_low_confidence_flag(client, face_img):
    r = client.post("/v1/embed?min_score=0.9999", files={"image": ("x.jpg", jpeg(face_img), "image/jpeg")}, headers=AUTH)
    faces = r.json()["faces"]
    assert faces and faces[0]["quality"] == {"ok": False, "reasons": ["low_confidence"]}


def test_min_face_px_from_settings(engine, face_img):
    with TestClient(create_app(make_settings(min_face_px=5000), engine=engine)) as c:
        r = c.post("/v1/embed", files={"image": ("x.jpg", jpeg(face_img), "image/jpeg")}, headers=AUTH)
    assert r.json()["faces"][0]["quality"]["reasons"] == ["face_too_small"]


# ---------- batch ----------

def variants(face_img, n):
    """n slightly different, good-quality frames of the same face."""
    out = []
    for i in range(n):
        shifted = np.roll(face_img, shift=i * 3, axis=1)
        out.append(b64(jpeg(cv2.convertScaleAbs(shifted, alpha=1.0, beta=i))))
    return out


def test_batch_all_usable(client, face_img):
    r = client.post("/v1/embed/batch", json={"images_b64": variants(face_img, 6), "min_usable": 5}, headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == MODEL_NAME
    assert body["total"] == 6 and body["usable"] == 6 and body["rejected"] == []
    assert len(body["embeddings"]) == 6
    for vec in body["embeddings"]:
        assert_embedding(vec)


def test_batch_rejections_and_min_usable(client, face_img):
    good = variants(face_img, 11)
    images = good[:3] + [
        b64(jpeg(np.hstack([face_img, face_img]))),          # 3: multiple_faces
        b64(jpeg(blank())),                                    # 4: no_face
        b64(jpeg(cv2.GaussianBlur(face_img, (0, 0), 6))),       # 5: blurry
        "!!!",                                                 # 6: invalid
    ] + good[3:]
    assert len(images) == 15

    r = client.post("/v1/embed/batch", json={"images_b64": images, "min_usable": 5}, headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 15 and body["usable"] == 11 and len(body["embeddings"]) == 11
    reasons = {x["index"]: x["reason"] for x in body["rejected"]}
    assert reasons == {3: "multiple_faces", 4: "no_face", 5: "blurry", 6: "invalid_image"}

    r = client.post("/v1/embed/batch", json={"images_b64": images, "min_usable": 12}, headers=AUTH)
    assert r.status_code == 422
    body = r.json()
    assert body["usable"] == 11 and body["min_usable"] == 12 and len(body["rejected"]) == 4


def test_batch_default_min_usable_is_5(client, face_img):
    r = client.post("/v1/embed/batch", json={"images_b64": variants(face_img, 4)}, headers=AUTH)
    assert r.status_code == 422
    assert r.json()["usable"] == 4


@pytest.mark.parametrize("body", [
    {}, {"images_b64": []}, {"images_b64": "x"}, {"images_b64": ["x"], "min_usable": 0}, {"images_b64": ["x"], "min_usable": "5"},
])
def test_batch_invalid_body_400(client, body):
    assert client.post("/v1/embed/batch", json=body, headers=AUTH).status_code == 400


def test_batch_too_many_images(client):
    r = client.post("/v1/embed/batch", json={"images_b64": ["x"] * 31}, headers=AUTH)
    assert r.status_code == 413


def test_nothing_image_related_is_logged(client, face_img, caplog):
    payload = b64(jpeg(face_img))
    with caplog.at_level("DEBUG"):
        client.post("/v1/embed", json={"image_b64": payload}, headers=AUTH)
        client.post("/v1/embed/batch", json={"images_b64": [payload]}, headers=AUTH)
    assert payload[100:160] not in caplog.text


def test_fixture_file_bytes_as_uploaded(client):
    """The raw fixture (a real JPEG file), exactly as curl would send it."""
    from conftest import HERE
    data = (HERE / "fixtures" / "face.jpg").read_bytes()
    r = client.post("/v1/embed", files={"image": ("face.jpg", data, "image/jpeg")}, headers=AUTH)
    assert r.status_code == 200 and len(r.json()["faces"]) == 1


def test_webp_is_rejected(client, face_img):
    ok, buf = cv2.imencode(".webp", face_img)
    r = client.post("/v1/embed", files={"image": ("x.jpg", buf.tobytes(), "image/jpeg")}, headers=AUTH)
    assert r.status_code == 400 and r.json()["error"]["code"] == "unsupported_format"
