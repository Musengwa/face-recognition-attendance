"""All configuration comes from environment variables."""
import os
from dataclasses import dataclass

MODEL_NAME = "yunet2023mar+sface2021dec"
EMBEDDING_DIM = 128
DETECTOR_FILE = "face_detection_yunet_2023mar.onnx"
RECOGNIZER_FILE = "face_recognition_sface_2021dec.onnx"


def _float(name, default):
    return float(os.environ.get(name, default))


def _int(name, default):
    return int(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    api_token: str
    env: str
    model_dir: str
    min_face_px: int
    blur_threshold: float
    detect_floor: float
    max_request_bytes: int
    max_image_side: int
    max_image_pixels: int
    max_batch_images: int
    port: int

    @property
    def dev_mode(self):
        return self.env == "dev"

    @classmethod
    def from_env(cls):
        default_models = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        return cls(
            api_token=os.environ.get("EMBED_API_TOKEN", ""),
            env=os.environ.get("EMBED_ENV", "production").strip().lower(),
            model_dir=os.environ.get("MODEL_DIR", default_models),
            min_face_px=_int("MIN_FACE_PX", 80),
            blur_threshold=_float("BLUR_THRESHOLD", 60.0),
            detect_floor=_float("DETECT_SCORE_FLOOR", 0.6),
            max_request_bytes=_int("MAX_REQUEST_BYTES", 5 * 1024 * 1024),
            max_image_side=_int("MAX_IMAGE_SIDE", 2000),
            max_image_pixels=_int("MAX_IMAGE_PIXELS", 40_000_000),
            max_batch_images=_int("MAX_BATCH_IMAGES", 30),
            port=_int("PORT", 8000),
        )

    def validate(self):
        if not self.api_token and not self.dev_mode:
            raise RuntimeError("EMBED_API_TOKEN is not set. Set it, or set EMBED_ENV=dev for local development only.")
