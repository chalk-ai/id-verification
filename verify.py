#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx", "opencv-python-headless", "numpy"]
# ///
"""Verify a video against a license image with the deployed demo.

    ./verify.py https://<web_url> license.jpg selfie.mp4 [--frames 15]

Samples frames evenly across the video, the way the page samples its 5-second recording.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys

import cv2
import httpx
import numpy as np


def data_url(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        sys.exit("could not encode an image")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def sample_frames(path: str, count: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        sys.exit(f"could not read frames from {path}")
    frames = []
    for i in np.linspace(0, total - 1, count).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            scale = 640 / max(frame.shape[:2])
            frames.append(cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame)
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("license")
    parser.add_argument("video")
    parser.add_argument("--frames", type=int, default=15)
    args = parser.parse_args()

    license_img = cv2.imread(args.license)
    if license_img is None:
        sys.exit(f"could not read {args.license}")
    frames = sample_frames(args.video, args.frames)
    res = httpx.post(
        args.url.rstrip("/") + "/verify",
        json={"license": data_url(license_img), "frames": [data_url(f) for f in frames]},
        timeout=60,
    )
    body = res.json()
    if res.status_code != 200:
        sys.exit(f"{res.status_code}: {body.get('error')}")
    print(
        json.dumps(
            {k: body[k] for k in ("verdict", "match", "liveness", "ms")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
