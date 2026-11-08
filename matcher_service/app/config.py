"""Matcher configuration, from environment variables only."""
import os
from dataclasses import dataclass


def _req(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"environment variable {name} is required")
    return value


@dataclass(frozen=True)
class Settings:
    camera_source: str
    camera_id: str
    embed_url: str
    embed_token: str
    django_url: str
    django_token: str
    match_threshold: float = 0.363
    match_margin: float = 0.05
    fps: float = 4.0
    gallery_refresh_seconds: float = 60.0
    session_poll_seconds: float = 10.0
    member_cooldown_seconds: float = 300.0
    imposter_cooldown_seconds: float = 20.0
    vote_window: int = 5
    votes_needed: int = 3
    frame_width: int = 640
    jpeg_quality: int = 85
    outbox_path: str = "matcher_outbox.sqlite3"
    status_host: str = "0.0.0.0"
    status_port: int = 8090
    http_timeout: float = 10.0

    @classmethod
    def from_env(cls):
        env = os.environ.get
        return cls(
            camera_source=env("CAMERA_SOURCE", "0"),
            camera_id=env("CAMERA_ID", "camera-1"),
            embed_url=_req("EMBED_URL").rstrip("/"),
            embed_token=_req("EMBED_API_TOKEN"),
            django_url=_req("DJANGO_URL").rstrip("/"),
            django_token=_req("DJANGO_API_TOKEN"),
            match_threshold=float(env("MATCH_THRESHOLD", 0.363)),
            match_margin=float(env("MATCH_MARGIN", 0.05)),
            fps=float(env("FPS", 4)),
            gallery_refresh_seconds=float(env("GALLERY_REFRESH_SECONDS", 60)),
            session_poll_seconds=float(env("SESSION_POLL_SECONDS", 10)),
            member_cooldown_seconds=float(env("MEMBER_COOLDOWN_SECONDS", 300)),
            imposter_cooldown_seconds=float(env("IMPOSTER_COOLDOWN_SECONDS", 20)),
            frame_width=int(env("FRAME_WIDTH", 640)),
            outbox_path=env("OUTBOX_PATH", "matcher_outbox.sqlite3"),
            status_host=env("STATUS_HOST", "0.0.0.0"),
            status_port=int(env("STATUS_PORT", 8090)),
            http_timeout=float(env("HTTP_TIMEOUT", 10)),
        )
