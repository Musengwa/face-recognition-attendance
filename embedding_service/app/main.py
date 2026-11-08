"""Embedding service: image in, face embeddings out. Stateless; images live only in memory.

Never log request bodies, image bytes, or embeddings from this module.
"""
import hmac
import json
import logging
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.formparsers import MultiPartParser

from .config import EMBEDDING_DIM, MODEL_NAME, Settings
from .engine import FaceEngine
from .images import ImageError, b64_to_bytes, decode

log = logging.getLogger("embedding_service")


class ApiError(Exception):
    def __init__(self, status, code, message, headers=None):
        super().__init__(message)
        self.status, self.code, self.message, self.headers = status, code, message, headers


def error_body(code, message):
    return {"error": {"code": code, "message": message}}


class BodyLimitMiddleware:
    """Reject request bodies over `limit` bytes with 413, whether or not Content-Length is sent."""

    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    too_big = int(value) > self.limit
                except ValueError:
                    too_big = False
                if too_big:
                    return await self._reject(scope, receive, send)
        chunks, size, more = [], 0, True
        while more:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > self.limit:
                return await self._reject(scope, receive, send)
            chunks.append(chunk)
            more = message.get("more_body", False)
        body = b"".join(chunks)
        sent = False

        async def replay():
            nonlocal sent
            if sent:
                return await receive()
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, replay, send)

    async def _reject(self, scope, receive, send):
        resp = JSONResponse(error_body("request_too_large", f"request body exceeds {self.limit} bytes"), status_code=413)
        await resp(scope, receive, send)


def create_app(settings=None, engine=None):
    settings = settings or Settings.from_env()
    settings.validate()
    if settings.dev_mode and not settings.api_token:
        log.warning("EMBED_ENV=dev and EMBED_API_TOKEN unset: authentication is DISABLED")

    @asynccontextmanager
    async def lifespan(app):
        app.state.engine = engine or FaceEngine(settings.model_dir, detect_floor=settings.detect_floor)
        log.info("models loaded: %s", MODEL_NAME)
        yield

    # Starlette spools multipart files above spool_max_size (1 MB) to a temp file on disk.
    # Bodies are capped at max_request_bytes, so raising the spool limit keeps uploads in memory.
    MultiPartParser.spool_max_size = max(MultiPartParser.spool_max_size, settings.max_request_bytes + 1)

    app = FastAPI(title="Face embedding service", version="1.0.0", lifespan=lifespan)
    app.state.settings = settings
    app.add_middleware(BodyLimitMiddleware, limit=settings.max_request_bytes)

    @app.exception_handler(ApiError)
    async def _api_error(request, exc):
        return JSONResponse(error_body(exc.code, exc.message), status_code=exc.status, headers=exc.headers)

    @app.exception_handler(ImageError)
    async def _image_error(request, exc):
        return JSONResponse(error_body(exc.code, exc.message), status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request, exc):
        # Do not echo the offending input back (it could be image data).
        fields = [".".join(str(p) for p in e.get("loc", ())) for e in exc.errors()]
        return JSONResponse(error_body("invalid_request", "invalid parameters: " + ", ".join(fields)), status_code=400)

    def require_token(request: Request):
        if not settings.api_token:      # dev mode only (validate() refuses this otherwise)
            return
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "token" or not hmac.compare_digest(token.strip().encode(), settings.api_token.encode()):
            raise ApiError(401, "unauthorized", "missing or invalid token", headers={"WWW-Authenticate": "Token"})

    @app.get("/health")
    def health():
        return {"status": "ok", "model": MODEL_NAME, "dim": EMBEDDING_DIM}

    api = APIRouter(prefix="/v1", dependencies=[Depends(require_token)])

    def run_engine(data, max_faces, min_score):
        img, orig_w, orig_h = decode(data, settings.max_image_side, settings.max_image_pixels)
        faces = app.state.engine.analyze(
            img, max_faces=max_faces, min_score=min_score, min_face_px=settings.min_face_px,
            blur_threshold=settings.blur_threshold, scale=orig_w / img.shape[1],
        )
        return faces, orig_w, orig_h

    @api.post("/embed")
    async def embed(request: Request,
                    max_faces: int = Query(5, ge=1, le=50),
                    min_score: float = Query(0.9, ge=0.0, le=1.0)):
        data = await read_single_image(request)
        faces, w, h = await run_in_threadpool(run_engine, data, max_faces, min_score)
        return {
            "model": MODEL_NAME,
            "image": {"width": w, "height": h},
            "faces": [face_json(f) for f in faces],
        }

    @api.post("/embed/batch")
    async def embed_batch(request: Request, min_score: float = Query(0.9, ge=0.0, le=1.0)):
        body = await read_json(request)
        images = body.get("images_b64")
        min_usable = body.get("min_usable", 5)
        if not isinstance(images, list) or not images:
            raise ApiError(400, "invalid_request", "images_b64 must be a non-empty list")
        if len(images) > settings.max_batch_images:
            raise ApiError(413, "too_many_images", f"at most {settings.max_batch_images} images per batch")
        if not isinstance(min_usable, int) or isinstance(min_usable, bool) or min_usable < 1:
            raise ApiError(400, "invalid_request", "min_usable must be a positive integer")

        result = await run_in_threadpool(run_batch, images, min_score)
        result["model"] = MODEL_NAME
        result["min_usable"] = min_usable
        status = 200 if result["usable"] >= min_usable else 422
        return JSONResponse(result, status_code=status)

    def run_batch(images, min_score):
        embeddings, rejected = [], []
        for index, item in enumerate(images):
            try:
                faces, _, _ = run_engine(b64_to_bytes(item), max_faces=10, min_score=min_score)
            except ImageError as exc:
                rejected.append({"index": index, "reason": "invalid_image", "detail": exc.code})
                continue
            confident = [f for f in faces if f.score >= min_score]
            if not faces:
                rejected.append({"index": index, "reason": "no_face"})
            elif len(confident) > 1:
                rejected.append({"index": index, "reason": "multiple_faces"})
            else:
                face = confident[0] if confident else faces[0]
                if face.ok:
                    embeddings.append(to_floats(face.embedding))
                else:
                    rejected.append({"index": index, "reason": face.reasons[0], "reasons": face.reasons})
        return {"total": len(images), "usable": len(embeddings), "embeddings": embeddings, "rejected": rejected}

    app.include_router(api)
    return app


def to_floats(vec):
    return [float(v) for v in vec]


def face_json(face):
    return {
        "box": face.box,
        "score": round(face.score, 4),
        "quality": {"ok": face.ok, "reasons": face.reasons},
        "embedding": to_floats(face.embedding),
    }


async def read_json(request):
    try:
        body = json.loads(await request.body())
    except (ValueError, UnicodeDecodeError):
        raise ApiError(400, "invalid_json", "body must be valid JSON") from None
    if not isinstance(body, dict):
        raise ApiError(400, "invalid_json", "body must be a JSON object")
    return body


async def read_single_image(request):
    ctype = request.headers.get("content-type", "").lower()
    if ctype.startswith("multipart/form-data"):
        try:
            form = await request.form()
        except Exception:
            raise ApiError(400, "invalid_multipart", "could not parse multipart body") from None
        upload = form.get("image")
        if upload is None or isinstance(upload, str):
            raise ApiError(400, "missing_image", "multipart field 'image' (a file) is required")
        data = await upload.read()
        await form.close()
        return data
    if ctype.startswith("application/json"):
        body = await read_json(request)
        if "image_b64" not in body:
            raise ApiError(400, "missing_image", "JSON field 'image_b64' is required")
        return b64_to_bytes(body["image_b64"])
    raise ApiError(415, "unsupported_media_type", "send multipart/form-data with field 'image' or JSON with 'image_b64'")
