"""Download the YuNet and SFace ONNX models from the OpenCV Zoo.

Usage:  python get_models.py [target_dir]     (default: $MODEL_DIR or ./models)

The .onnx files are never committed; run this once per machine / image build.
Each file is checked against a pinned SHA-256 so every machine (and the Django
dashboard) runs byte-identical models.
"""
import hashlib
import os
import sys
import tempfile
import urllib.request

MIRRORS = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models",
    "https://github.com/opencv/opencv_zoo/raw/main/models",
)
MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        "face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    "face_recognition_sface_2021dec.onnx": (
        "face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name, rel_path, expected, target_dir):
    dest = os.path.join(target_dir, name)
    if os.path.exists(dest) and sha256(dest) == expected:
        print(f"[ok]   {name} already present")
        return
    for mirror in MIRRORS:
        try:
            download(f"{mirror}/{rel_path}", dest, expected)
            print(f"[ok]   {name} ({os.path.getsize(dest)} bytes)")
            return
        except Exception as exc:  # try the next mirror
            print(f"[warn] {name} from {mirror}: {exc}")
    raise SystemExit(f"[fail] could not download {name}")


def download(url, dest, expected):
    print(f"[get]  {url}")
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dest), suffix=".part")
    os.close(fd)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "get_models.py"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
        actual = sha256(tmp)
        if actual != expected:
            raise ValueError(f"sha256 {actual} != expected {expected}")
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__) or ".", "models"))
    os.makedirs(target, exist_ok=True)
    for name, (rel_path, digest) in MODELS.items():
        fetch(name, rel_path, digest, target)


if __name__ == "__main__":
    main()
