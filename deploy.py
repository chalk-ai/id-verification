#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute==2.13.4"]
# ///
"""Deploy the ID verification demo (server.py + index.html) as a CPU ScalingGroup.

One replica serves the page and the /verify API, so the browser's requests are same-origin.
Face detection and recognition run on CPU with OpenCV, so no GPU is needed.

Credentials come from your existing `chalk login` (or the CHALK_* env vars):
    ./deploy.py
Then open the printed URL, pick a license, and record five seconds of video.
"""

from __future__ import annotations

import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS_DIR = "/models"
ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
MODELS = [
    f"{ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    f"{ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
]


def main() -> None:
    from chalkcompute import Image, ReadinessProbe, ScalingGroup

    image = (
        Image.debian_slim(python_version="3.12")
        .apt_install(["curl"])
        .pip_install(
            [
                "opencv-python-headless==4.14.0.94",
                "numpy==2.5.4",
                "fastapi==0.143.0",
                "uvicorn[standard]==0.54.0",
            ]
        )
        .env({"IDV_MODELS_DIR": MODELS_DIR})
        # YuNet (MIT) and SFace (Apache-2.0) are about 40 MB together, small enough to bake in.
        .run_commands(
            f"mkdir -p {MODELS_DIR}", *(f"curl -fsSL -o {MODELS_DIR}/{u.rsplit('/', 1)[1]} {u}" for u in MODELS)
        )
        .add_local_file(str(HERE / "server.py"), "/app/server.py", strategy="copy")
        .add_local_file(str(HERE / "index.html"), "/app/index.html", strategy="copy")
    )
    # Copied files share a 32 KiB budget that server.py and index.html mostly use, so the
    # sample portraits ride on a volume mounted at their own path. A volume shadows anything
    # copied into the same directory, which is why it is not under /app.
    image = image.add_local_dir(str(HERE / "samples"), "/samples", strategy="volume").workdir("/app")

    sg = ScalingGroup(
        image=image,
        name='id-verification',
        cpu="2",
        memory="4Gi",
        env={"PORT": "8080", "IDV_SAMPLES_DIR": "/samples"},
        entrypoint=["python", "/app/server.py"],
        port=8080,
        readiness_probe=ReadinessProbe.http("/status", period_seconds=5),
        min_replicas=1,
        max_replicas=1,
    )
    t0 = time.monotonic()
    try:
        sg.deploy(build_timeout=900.0, ready_timeout=900.0)
    finally:
        print(f"deploy returned after {time.monotonic() - t0:.0f}s")
    print("web_url:", sg.web_url)


if __name__ == "__main__":
    main()
