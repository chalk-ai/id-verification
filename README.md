# ID verification on Chalk

Does the person on camera match the photo on their license? This demo answers that from a
5-second webcam clip, using a clearly fictional "Chalkland" specimen license. The page, the API
and the models are served from a single CPU replica deployed with
[`chalkcompute`](https://pypi.org/project/chalkcompute/). No GPU is needed.

```
license photo ─┐
               ├─▶ YuNet face detection ─▶ SFace embedding ─▶ cosine similarity ─┐
15 video frames┘                                                                  ├─▶ verdict
                    YuNet landmarks ─▶ head-turn range across the clip ──────────┘
```

1. **Choose a license.** Take a webcam snapshot to make a specimen license with your own face, or
   pick one of four samples. The sample portraits are AI-generated and do not depict real people.
2. **Record 5 seconds.** The page captures 15 frames while asking you to look straight, then turn
   your head left and right.
3. **Verify.** The server finds the largest face on the license and in each frame, embeds each one
   with SFace, and compares them. It also measures how far the head turned across the clip.

## Files

| File | Role |
| --- | --- |
| `deploy.py` | Builds the image (OpenCV, with both models baked in) and deploys a `chalkcompute.ScalingGroup` on 2 CPUs. |
| `server.py` | FastAPI app: `GET /` serves the page, `POST /verify` compares a license image with video frames. |
| `index.html` | The page: draws the specimen license, records the clip, and shows the result frame by frame. |
| `samples/` | Four AI-generated portraits for the sample licenses. |
| `verify.py` | Sends a license image and a video file to `/verify` from the command line. |

## Run it

You need [`uv`](https://docs.astral.sh/uv/) and a Chalk environment. Credentials come from
the SDK’s ambient credentials from `chalk login`. You can also set `CHALK_API_SERVER`, `CHALK_CLIENT_ID`,
`CHALK_CLIENT_SECRET` and `CHALK_ENVIRONMENT_ID` yourself.

```bash
./deploy.py
```

The deploy takes a few minutes. Open the printed `web_url`, choose a license, and record. Browsers
only allow camera access over HTTPS, which the scaling-group URL already uses.

From the command line:

```bash
./verify.py https://<web_url> license.jpg clip.mp4
```

## How the decision is made

- **Match.** The score is the median cosine similarity between the license face and the more
  frontal half of the frames; head-turn frames are there for liveness and match a frontal photo
  less well. A score of at least 0.363, SFace's published threshold, counts as the same person.
- **Liveness.** For each frame the server places the nose tip between the two eyes on a 0–1 scale,
  about 0.5 when facing the camera. A real head turn moves it well toward one eye; a flat photo
  tilted in front of the camera barely moves it. A spread of at least 0.15 across the clip passes.
- **Verdict.** `verified` needs both. Otherwise the result is `no_match`, `not_live`, or `no_face`
  when a face is visible in fewer than 60% of the frames.

## This is a demo, not an identity product

- The head-turn check is a single cheap signal. It stops a still photo, not a video replay, a
  mask, or a deepfake. Production identity verification adds document authentication, active
  challenges and dedicated presentation-attack detection.
- Face recognition accuracy varies across demographics, lighting and cameras. Calibrate the
  threshold on your own data before relying on it.
- Only use faces with the person's consent, and follow the biometric privacy laws that apply to
  you.

## Privacy

Images arrive in the request body, are decoded and compared in memory, and are discarded when the
response is sent. The server never writes them to disk or logs them; it logs only the verdict,
the scores and the timing. Each request is limited to 24 frames of at most 600 KB each, and the
replica admits at most `IDV_RATE_LIMIT_PER_MINUTE` verifications a minute (default 60).

## Clean up

```bash
uv run --with chalkcompute python -c \
  "from chalkcompute import ScalingGroup; ScalingGroup.from_name('id-verification').delete()"
```

## License

The code is Apache-2.0. The YuNet face detector is MIT and the SFace face recognizer is
Apache-2.0, both from the [OpenCV model zoo](https://github.com/opencv/opencv_zoo).

## Local authentication

The SDK reads credentials, API host, and the selected environment from your existing
`chalk login` in `~/.chalk.yml`, including project-scoped and default logins.
For unattended runs, `CHALK_CLIENT_ID` and `CHALK_CLIENT_SECRET` remain supported.
The examples use the current environment selected by your Chalk login.
