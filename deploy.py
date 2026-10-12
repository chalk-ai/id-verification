#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute==2.13.4"]
# ///
"""Deploy the ID verification demo (server.py + index.html) as a CPU ScalingGroup.

One replica serves the page and the /verify API, so the browser's requests are same-origin.
Face detection and recognition run on CPU with OpenCV, so no GPU is needed.

Credentials come from `chalk config` in the current directory (or the CHALK_* env vars):
    ./deploy.py [--env-id <environment>]
Then open the printed URL, pick a license, and record five seconds of video.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS_DIR = "/models"
ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
MODELS = [
    f"{ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    f"{ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
]


def load_chalk_env(env_id: str | None) -> None:
    if "CHALK_CLIENT_ID" not in os.environ:
        cfg = json.loads(subprocess.check_output(["chalk", "config", "--json"]))
        os.environ["CHALK_API_SERVER"] = cfg["apiServer"]["value"]
        os.environ["CHALK_CLIENT_ID"] = cfg["clientId"]["value"]
        os.environ["CHALK_CLIENT_SECRET"] = cfg["clientSecret"]["value"]
        os.environ.setdefault("CHALK_ENVIRONMENT_ID", cfg["environment"]["value"])
    if env_id:
        os.environ["CHALK_ENVIRONMENT_ID"] = env_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="id-verification")
    # Defaults to the environment in `chalk config` (or CHALK_ENVIRONMENT_ID).
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--ready-timeout", type=float, default=900.0)
    args = parser.parse_args()

    load_chalk_env(args.env_id)
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
        name=args.name,
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
        sg.deploy(build_timeout=900.0, ready_timeout=args.ready_timeout)
    finally:
        print(f"deploy returned after {time.monotonic() - t0:.0f}s")
    print("web_url:", sg.web_url)


if __name__ == "__main__":
    main()
