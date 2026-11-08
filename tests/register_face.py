"""Dev helper: register a face in the fake Django, the way the dashboard's admin camera will.

Captures frames (from a camera or from image files), sends them to the embedding service's
/v1/embed/batch, and stores the returned embeddings in the fake Django. Frames are only held in memory.

  python tests/register_face.py --member-id 202400004 --camera 0 --frames 14
  python tests/register_face.py --member-id 202400019 --images a.jpg b.jpg --banned --min-usable 2

Env: EMBED_URL, EMBED_API_TOKEN, DJANGO_URL (fake), DJANGO_API_TOKEN.
"""
import argparse
import base64
import os
import sys
import time

import cv2
import httpx


def frames_from_camera(source, n, interval):
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        sys.exit(f"cannot open camera {source}")
    out = []
    try:
        while len(out) < n:
            ok, frame = cap.read()
            if not ok:
                sys.exit("camera read failed")
            out.append(frame)
            print(f"captured {len(out)}/{n} - move your head slightly", flush=True)
            time.sleep(interval)
    finally:
        cap.release()
    return out


def to_b64(frame):
    h, w = frame.shape[:2]
    if w > 960:
        frame = cv2.resize(frame, (960, round(h * 960 / w)))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--member-id", required=True)
    p.add_argument("--name")
    p.add_argument("--banned", action="store_true")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera")
    src.add_argument("--images", nargs="+")
    p.add_argument("--frames", type=int, default=14)
    p.add_argument("--interval", type=float, default=0.3)
    p.add_argument("--min-usable", type=int, default=5)
    a = p.parse_args()

    if a.images:
        frames = [cv2.imread(f) for f in a.images]
        if any(f is None for f in frames):
            sys.exit("could not read one of the images")
    else:
        frames = frames_from_camera(a.camera, a.frames, a.interval)

    embed = httpx.Client(base_url=os.environ["EMBED_URL"], timeout=60,
                         headers={"Authorization": f"Token {os.environ['EMBED_API_TOKEN']}"})
    r = embed.post("/v1/embed/batch", json={"images_b64": [to_b64(f) for f in frames], "min_usable": a.min_usable})
    body = r.json()
    print(f"embedding service: HTTP {r.status_code}, usable {body.get('usable')}/{body.get('total')}, rejected {body.get('rejected')}")
    if r.status_code != 200:
        sys.exit("not enough usable frames; nothing registered")

    django = httpx.Client(base_url=os.environ["DJANGO_URL"], timeout=30,
                          headers={"Authorization": f"Token {os.environ['DJANGO_API_TOKEN']}"})
    r = django.post("/fake/members", json={"member_id": a.member_id, "name": a.name or a.member_id, "banned": a.banned,
                                           "model": body["model"], "embeddings": body["embeddings"]})
    r.raise_for_status()
    print("registered:", r.json())


if __name__ == "__main__":
    main()
