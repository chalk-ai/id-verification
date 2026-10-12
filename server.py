"""Face-to-ID verification: does the person on camera match the photo on a (specimen) license?

Runs inside the container image that deploy.py builds for a Chalk ScalingGroup; the page at
`/` is index.html. Everything runs on CPU with OpenCV's YuNet face detector and SFace face
recognizer. Images arrive in the request body, are processed in memory, and are never written
to disk or logged.

API:
  POST /verify  {"license": "data:image/...;base64,...", "frames": ["data:image/jpeg;base64,...", ...]}
    200 {"verdict": "verified" | "no_match" | "not_live" | "no_face",
         "match": {"similarity", "threshold", "matched"},
         "liveness": {"passed", "yaw_range", "min_yaw_range"},
         "frames": [{"face": bool, "faces": int, "similarity" | null, "yaw" | null}],
         "license_face": "data:image/jpeg;base64,...", "best_frame_face": "data:..." | null,
         "ms": {"decode", "detect_embed", "total"}}
    400 {"error"}   malformed input, or no face found on the license
    413 {"error"}   too many frames or images too large
    429 {"error"}   the demo is over its per-minute budget
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import os
import statistics
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

MODELS_DIR = Path(os.environ.get("IDV_MODELS_DIR", "/models"))
DETECTOR_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
RECOGNIZER_MODEL = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
HERE = Path(__file__).resolve().parent
SAMPLES_DIR = Path(os.environ.get("IDV_SAMPLES_DIR", HERE / "samples"))
PORT = int(os.environ.get("PORT", "8080"))

# SFace's published cosine-similarity threshold for "same person" on LFW.
MATCH_THRESHOLD = 0.363
# Frames whose largest face scores below this are treated as having no face.
DETECT_SCORE = 0.8
# Fraction of frames that must contain a face for the video to count.
MIN_FACE_FRAMES = 0.6
# Spread of the yaw proxy (see `yaw_proxy`) across the clip that counts as a real head turn.
MIN_YAW_RANGE = 0.15
MAX_FRAMES = 24
MAX_IMAGE_BYTES = 600_000
# Downscale inputs so detection cost is bounded no matter what the client sends.
MAX_SIDE = 640
# The scaling-group gateway forwards only internal load-balancer addresses, so there is no
# trustworthy client IP; the budget is for the whole replica.
RATE_LIMIT = int(os.environ.get("IDV_RATE_LIMIT_PER_MINUTE", "60"))

log = logging.getLogger("idv")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class BadInput(ValueError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class Face:
    """The largest detected face in an image: YuNet's 15-value row plus its SFace embedding."""

    row: np.ndarray
    embedding: np.ndarray
    aligned: np.ndarray


def decode_image(data_url: Any, what: str) -> np.ndarray:
    if not isinstance(data_url, str) or "," not in data_url:
        raise BadInput(400, f"{what} must be a base64 data URL.")
    try:
        raw = base64.b64decode(data_url.split(",", 1)[1], validate=True)
    except (binascii.Error, ValueError) as e:
        raise BadInput(400, f"{what} is not valid base64.") from e
    if len(raw) > MAX_IMAGE_BYTES:
        raise BadInput(413, f"{what} is larger than {MAX_IMAGE_BYTES // 1000} KB.")
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise BadInput(400, f"{what} is not a decodable image.")
    scale = MAX_SIDE / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def yaw_proxy(row: np.ndarray) -> float:
    """Where the nose tip sits between the eyes, from 0 (left eye) to 1 (right eye).

    A frontal face reads about 0.5; turning the head moves the nose toward one eye. A flat
    photo rotated in front of the camera keeps this ratio nearly constant, because a planar
    projection roughly preserves where a point lies along a line, so the spread across a clip
    is a cheap signal that the face is three-dimensional.
    """
    right_eye_x, left_eye_x, nose_x = row[4], row[6], row[8]
    span = left_eye_x - right_eye_x
    if abs(span) < 1e-3:
        return 0.5
    return float((nose_x - right_eye_x) / span)


def to_jpeg_data_url(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


class Verifier:
    """YuNet + SFace. OpenCV DNN objects are not thread-safe, so calls hold `lock`."""

    def __init__(self) -> None:
        super().__init__()
        self.detector = cv2.FaceDetectorYN.create(str(DETECTOR_MODEL), "", (320, 320), DETECT_SCORE)
        self.recognizer = cv2.FaceRecognizerSF.create(str(RECOGNIZER_MODEL), "")
        self.lock = threading.Lock()

    def faces(self, img: np.ndarray) -> tuple[Face | None, int]:
        """The largest face in `img` and how many faces were detected."""
        h, w = img.shape[:2]
        self.detector.setInputSize((w, h))
        _, rows = self.detector.detect(img)
        if rows is None or len(rows) == 0:
            return None, 0
        row = max(rows, key=lambda r: r[2] * r[3])
        aligned = self.recognizer.alignCrop(img, row)
        embedding = self.recognizer.feature(aligned)
        return Face(row=row, embedding=embedding, aligned=aligned), len(rows)

    def similarity(self, a: Face, b: Face) -> float:
        return float(self.recognizer.match(a.embedding, b.embedding, cv2.FaceRecognizerSF_FR_COSINE))

    def verify(self, license_img: np.ndarray, frames: list[np.ndarray]) -> dict[str, Any]:
        with self.lock:
            license_face, _ = self.faces(license_img)
            if license_face is None:
                raise BadInput(400, "No face found on the license. Use a clearer photo.")
            per_frame: list[dict[str, Any]] = []
            best: tuple[float, Face] | None = None
            for img in frames:
                face, count = self.faces(img)
                if face is None:
                    per_frame.append({"face": False, "faces": 0, "similarity": None, "yaw": None})
                    continue
                sim = self.similarity(license_face, face)
                per_frame.append(
                    {"face": True, "faces": count, "similarity": round(sim, 4), "yaw": round(yaw_proxy(face.row), 4)}
                )
                if best is None or sim > best[0]:
                    best = (sim, face)

        found = [f for f in per_frame if f["face"]]
        yaws = [f["yaw"] for f in found]
        enough_faces = len(found) >= max(1, int(MIN_FACE_FRAMES * len(frames)))
        # Score identity on the more frontal half of the clip: the frames where the user is
        # mid head-turn are there for liveness, and naturally match a frontal ID photo less well.
        frontal = sorted(found, key=lambda f: abs(f["yaw"] - 0.5))[: max(1, (len(found) + 1) // 2)]
        similarity = statistics.median(f["similarity"] for f in frontal) if found else 0.0
        matched = enough_faces and similarity >= MATCH_THRESHOLD
        yaw_range = (max(yaws) - min(yaws)) if yaws else 0.0
        live = enough_faces and yaw_range >= MIN_YAW_RANGE
        if not enough_faces:
            verdict = "no_face"
        elif not matched:
            verdict = "no_match"
        elif not live:
            verdict = "not_live"
        else:
            verdict = "verified"
        return {
            "verdict": verdict,
            "match": {"similarity": round(similarity, 4), "threshold": MATCH_THRESHOLD, "matched": matched},
            "liveness": {"passed": live, "yaw_range": round(yaw_range, 4), "min_yaw_range": MIN_YAW_RANGE},
            "frames": per_frame,
            "license_face": to_jpeg_data_url(license_face.aligned),
            "best_frame_face": to_jpeg_data_url(best[1].aligned) if best else None,
        }


class RateLimiter:
    """At most `per_minute` admissions in any sliding one-minute window."""

    def __init__(self, per_minute: int) -> None:
        super().__init__()
        self.per_minute = per_minute
        self.hits: deque[float] = deque()

    def allow(self) -> bool:
        now = time.monotonic()
        while self.hits and now - self.hits[0] > 60:
            self.hits.popleft()
        if len(self.hits) >= self.per_minute:
            return False
        self.hits.append(now)
        return True


state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    state["verifier"] = await asyncio.to_thread(Verifier)
    state["limiter"] = RateLimiter(RATE_LIMIT)
    log.info("models loaded")
    yield


app = FastAPI(lifespan=lifespan)


def error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse((HERE / "index.html").read_text())


@app.get("/samples/{name}.webp")
async def sample(name: str) -> Response:
    path = SAMPLES_DIR / f"{name}.webp"
    if not name.isalnum() or not path.is_file():
        return error(404, "Unknown sample.")
    return FileResponse(path, media_type="image/webp")


@app.get("/status")
async def status() -> JSONResponse:
    return JSONResponse({"ok": True, "detector": DETECTOR_MODEL.name, "recognizer": RECOGNIZER_MODEL.name})


@app.post("/verify")
async def verify(request: Request) -> JSONResponse:
    if int(request.headers.get("content-length") or 0) > MAX_IMAGE_BYTES * (MAX_FRAMES + 2):
        return error(413, "Request is too large.")
    try:
        body = await request.json()
    except ValueError:
        return error(400, "Expected a JSON body.")
    frames_in = body.get("frames")
    if not isinstance(frames_in, list) or not frames_in:
        return error(400, "Send at least one video frame.")
    if len(frames_in) > MAX_FRAMES:
        return error(413, f"Send at most {MAX_FRAMES} frames.")
    if not state["limiter"].allow():
        return error(429, "The demo is at capacity right now. Try again in a minute.")

    verifier: Verifier = state["verifier"]

    def run() -> dict[str, Any]:
        t0 = time.perf_counter()
        license_img = decode_image(body.get("license"), "The license image")
        frames = [decode_image(f, f"Frame {i + 1}") for i, f in enumerate(frames_in)]
        t1 = time.perf_counter()
        result = verifier.verify(license_img, frames)
        t2 = time.perf_counter()
        result["ms"] = {
            "decode": round((t1 - t0) * 1000),
            "detect_embed": round((t2 - t1) * 1000),
            "total": round((t2 - t0) * 1000),
        }
        return result

    try:
        result = await asyncio.to_thread(run)
    except BadInput as e:
        return error(e.status, str(e))
    # Only aggregate numbers are logged; images never are.
    log.info(
        "verdict=%s similarity=%.3f yaw_range=%.3f frames=%d ms=%d",
        result["verdict"],
        result["match"]["similarity"],
        result["liveness"]["yaw_range"],
        len(result["frames"]),
        result["ms"]["total"],
    )
    return JSONResponse(result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT, access_log=False)
