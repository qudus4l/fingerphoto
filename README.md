# fingerphoto

Fingerprint recognition from phone-camera photos: enroll a finger from
photos, then identify it in any later photo. Photo-to-photo matching — no
contact scanner involved. Ships as an HTTP service (FastAPI) with a CLI for
local use.

## How it works

```
video / burst ─▶ score every sampled frame (ridge detector, no Gabor, ~0.5 s)
              ─▶ keep the best few
photo ─▶ skin-mask distal crop ─▶ grayscale
      ─▶ ridge-frequency scale normalization, closed-loop (pseudo-500dpi)
      ─▶ RIDGE-QUALITY GATE: refuse crops with no resolvable ridges
      ─▶ invert polarity ─▶ bilateral + CLAHE ─▶ oriented Gabor enhancement
      ─▶ mask-out background ─▶ SourceAFIS templates at 3 scale variants
      ─▶ set-vs-set max-score matching (threshold ≈ 40)
```

Scale normalization is the make-or-break step: a photo has unknown DPI, so we
estimate the ridge period (grid-median FFT — a single global FFT gets captured
by creases/specular highlights), rescale until ridges sit ~8 px apart, then
re-measure on the enhanced binary map and correct (closed loop). Residual
error is absorbed by extracting each photo at ±13% scale variants and scoring
identity sets by the max over the cross product.

The gate is the reason a bad photo now fails loudly. The Gabor stage is an
oriented band-pass filter: fed an out-of-focus finger it emits ridge-like
texture and SourceAFIS finds 50-100 minutiae in the hallucination, so a
minutiae count cannot tell a print from an invention. `fingerphoto/quality.py`
measures ridge structure on the scale-normalized grayscale *before*
enhancement, combining band-passed orientation coherence, edge density and
ridge-band spectral energy. On this project's crops it separates usable from
unusable perfectly (AUC 1.00, scores 0.73-1.00 against 0.00-0.07); see
`scripts/quality_calibration.py`, which is also the regression test for it.

Best-frame selection is the same measurement used twice. A few seconds of
video carries dozens of chances at a readable print where a photo carries one,
but extracting a frame costs 10-20 s while scoring one costs 0.3-1.0 s, because
scoring stops before the Gabor stage. So `fingerphoto/frames.py` samples a clip
at 3 fps, scores every sampled frame with `quality.ridge_quality` plus the same
geometric sanity rules the gate applies, and hands only the best few to the
pipeline. A frame the selector picks is therefore one the gate will accept, and
`tests/test_frame_score_corpus.py` pins the two together on the real corpus.
Frames are scored at FULL RESOLUTION: ridges sit 8-20 px apart in a 4K frame,
so downscaling destroys the signal being measured (at a 2560 px long side one
corpus photo drops from 0.91 to 0.00 and a failed probe climbs past the gate).
If scoring is ever too slow, sample fewer frames, never smaller ones. The
frames that get enrolled are what is archived as the originals, not the clip.

Research behind the design: see "The Fingerphoto Dossier" artifact
(three-track survey of tooling, academic SOTA, and practitioner pipelines,
2026-08). Key sources: Priesnitz 2022 (Sensors), C2CL (IEEE TIFS 2022),
Kauba/Ruzicka enhancement parameters (IET Biometrics 2023).

## Setup

```sh
make install          # venv + deps + SourceAFIS jars (needs Java 11+ locally)
cp .env.example .env  # set FP_API_KEYS (and ideally FP_ENCRYPTION_KEY)
```

## Run the service

```sh
make run              # uvicorn on 127.0.0.1:8000
# or in Docker:
make docker-up        # builds image, mounts a named volume at /data
```

The API is asynchronous where the pipeline is slow (10–60 s per photo):
enroll/identify/verify return `202 Accepted` with a `Location` header and a
job id; poll `GET /v1/jobs/{id}` until `succeeded`/`failed`. Auth is an
`X-API-Key` header (`FP_API_KEYS`, comma-separated). Interactive docs at
`/docs`.

| Endpoint | What it does |
|---|---|
| `POST /v1/identities` | Create identity `{name, consent: true}` (consent is mandatory) |
| `GET /v1/identities` · `GET /v1/identities/{uuid}` | List / inspect |
| `POST /v1/identities/{uuid}/photos` | Enroll photos or a video (multipart `files`, optional `finger`, `best_of`) → 202 |
| `POST /v1/identify` | 1:N search of a photo, video, or frame set against everyone → 202 |
| `POST /v1/identities/{uuid}/verify` | 1:1 check against a claimed identity → 202 |
| `POST /v1/quality` | Ridge score of one frame, synchronous — the capture UI live indicator |
| `GET /v1/jobs/{id}` | Poll a job; per-photo results carry machine-readable `reason_code`s |
| `DELETE /v1/identities/{uuid}` | Hard delete: templates, rows, original photos |
| `GET /healthz` · `GET /readyz` · `GET /v1/version` | Liveness / readiness (JVM+DB+queue) / versions |

Finger positions use ANSI/NIST-ITL names: `right_thumb` … `left_little`,
default `unknown`. JPEG/PNG/HEIC accepted (iPhone HEIC works natively).
MP4, MOV, WebM and MKV are accepted wherever a photo is, and their
frames are scored and selected server-side (`FP_VIDEO_FRAMES_ENROLL`, default
3; `FP_VIDEO_FRAMES_IDENTIFY`, default 2). `best_of=N` instead pools every
upload in one request and keeps the N best overall — that is how the capture
page sends its five-frame burst.
Uploads over `FP_MAX_UPLOAD_MB` (default 64, since a 5 s 4K clip is 15-35 MB)
are rejected with 413. Optional
`Idempotency-Key` header makes a repeated submission return the same job.

Example session:

```sh
curl -s -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"name":"Q","consent":true}' localhost:8000/v1/identities
curl -s -H "X-API-Key: $KEY" -F files=@thumb1.jpg -F files=@thumb2.jpg \
  -F finger=right_thumb localhost:8000/v1/identities/<uuid>/photos
curl -s -H "X-API-Key: $KEY" localhost:8000/v1/jobs/<job_id>
curl -s -H "X-API-Key: $KEY" -F file=@probe.jpg localhost:8000/v1/identify
# a clip: frames are sampled, scored, and the best few extracted
curl -s -H "X-API-Key: $KEY" -F files=@thumb.mov localhost:8000/v1/identify
# is this frame worth capturing at all?
curl -s -H "X-API-Key: $KEY" -F file=@frame.jpg localhost:8000/v1/quality
```

## CLI (same store as the API)

```sh
./.venv/bin/python scripts/enroll.py NAME photo1.jpg photo2.jpg
./.venv/bin/python scripts/identify.py probe.jpg
./.venv/bin/python scripts/evaluate.py              # measure the whole corpus
./.venv/bin/python scripts/quality_calibration.py   # check the quality gate
./.venv/bin/python scripts/rebuild_gallery.py       # re-enrol from the manifest
```

## Operations

**Process model.** Exactly one uvicorn worker (pinned in the Docker CMD).
The JVM starts once per process and can never restart; the in-process job
store needs polls to land where the job runs; SQLite gets a single writer.
CPU parallelism comes from a thread pool (`FP_CPU_WORKERS`, default 2) —
OpenCV and JPype release the GIL. The job queue is bounded
(`FP_MAX_QUEUE`, default 8): when full, submissions get 503 + Retry-After
and `/readyz` flips, so a load balancer stops routing. Jobs are in-memory
by design — a restart loses in-flight jobs and clients re-submit.

**Storage.** SQLite (WAL — keep the data volume on a local filesystem, not
NFS) at `FP_DB_PATH`; original photos under `FP_DATA_DIR/enrolled/<uuid>/`.
Originals are kept deliberately: SourceAFIS templates have **no backward
compatibility across versions** (per its own docs), so upgrading the matcher
means re-extracting from originals. Set `FP_STORE_ORIGINALS=false` to opt
out — accepting that a SourceAFIS upgrade then means re-photographing
everyone. Templates and photos are encrypted at rest with
`FP_ENCRYPTION_KEY` (Fernet; generate with
`python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`).
Back the key up separately from the data volume — losing it makes originals
unrecoverable. Rotation: `scripts/reencrypt.py --old <oldkey>` (server
stopped).

**Biometric-data compliance (GDPR Art. 9 / BIPA).** Templates derived from
photos are regulated biometric identifiers (see *Sosa v. Onfido*). Built in:
consent is a mandatory literal at identity creation (recorded with
timestamp); `DELETE /v1/identities/{uuid}` is a hard delete of rows, photos,
and WAL remnants (`secure_delete` is on); every enroll/identify/verify/delete
writes an `audit_log` row (metadata only — no image data or scores of
uncharged parties); logs are structured JSON and never carry image bytes or
templates. Operator obligations that live outside the code: present a
written notice + release before capture (BIPA), publish a retention policy,
and run `scripts/retention_sweep.py` (dry-run by default, `--delete` to
execute) on a schedule — BIPA requires destruction within 3 years of the
last interaction, or sooner when the purpose is satisfied.

**Threshold honesty.** `FP_THRESHOLD=40` maps to ~0.01% false-match rate
*per SourceAFIS's own calibration on contact-scanner data* — it has behaved
exactly as advertised on this project's photos (zero false accepts across
every experiment), but the cohort is two people. Calibrate on more subjects
before trusting the FMR number in production.

**Quality gating.** Two gates run per crop. `FP_MIN_RIDGE_QUALITY` (default
0.40) is the ridge-structure score described above; `FP_MIN_MINUTIAE` (default
15) still applies to whatever survives it. A crop is also refused on geometry
alone when its ridge period falls outside the physical band implied by the crop
width, or when the texture localizer returns more than 30% of the frame — both
mean the crop is a hand or a scene, not a fingertip. Set
`FP_MIN_RIDGE_QUALITY=0` to disable the score gate; the geometric rules always
apply. NFIQ2 is deliberately not used: NIST SP 500-334r1 disallows it on
contactless imagery, and the features that do predict contactless utility are
the ones measured here.

**Versioning.** Every template row is stamped with `pipeline_version` +
`sourceafis_version`; `GET /v1/version` reports both. Bump
`PIPELINE_VERSION` in `fingerphoto/version.py` whenever extraction
parameters change, then re-extract stale templates from originals.

## Tests

```sh
make test-fast   # unit + API-shape tests, no JVM (~5 s, 114 tests)
make test        # + golden scores, quality-gate corpus, API e2e (minutes)
node scripts/capture_sim.js   # the capture page against a simulated camera (~60 s)
```

Golden bands encode this project's validated numbers: genuine thumb trio
≥ 100 (observed 144–195), impostor sister_6214 < 40 (observed ≤ 21.8), the
two no-ridge photos asserted to be *refused*, plus a two-thread
concurrent-extraction test (JVM lock + enhancer thread-safety).

`scripts/capture_sim.js` runs `capture.html`'s own script in a VM against a
fake Android camera — a lens with a true focus distance, ridges that fade as it
defocuses, a torch that helps — and checks the adaptive sweep converges on that
distance, picks the better light, restores the camera afterwards, and still
falls back to the plain burst on a browser with no camera controls. It needs
only node. It is the only automated coverage the capture page has, because the
alternative is a phone in someone's hand.

**Measuring a change.** `scripts/evaluate.py` runs the labelled corpus
(`data/samples/manifest.csv`) and reports genuine/impostor distributions, EER,
failure-to-acquire and leave-one-out 1:N identification, caching template sets
so a re-run after a non-pipeline change is seconds rather than minutes. Use
`--compare out/eval/latest.json` for deltas and `--write-baseline` to accept a
change; `tests/test_evaluation_slow.py` guards the checked-in baseline. Numbers
at pipeline 1.2.0 on 12 labelled photos, threshold 40:

| | 1.1.0 | 1.2.0 (gate) |
|---|---|---|
| genuine median | 12.8 | 75.5 |
| genuine max | 194.6 | 194.6 |
| impostor max | 29.0 | 21.8 |
| FNMR | 0.67 | 0.50 |
| EER | 0.40 | 0.33 |
| rank-1 identification | 0.43 | 0.75 |
| failure to acquire | 1 | 10 |

The rise in failure-to-acquire is the point: ten of the twelve labelled photos
have no resolvable ridges at full resolution, and they were previously enrolling
50–100 hallucinated minutiae each. Genuine scores for real captures are
unchanged, because an accepted crop is extracted exactly as before.

## Capture protocol (matters enormously)

**On an iPhone, record a video from the Camera app.** Its video mode is the
only route that gives a continuous torch, tap-to-focus and 4K at once; a web
page gets none of the three from `getUserMedia` on Safari, which is why live
browser captures have failed here. Record 3-5 seconds, filling the outline,
rocking the phone slowly nearer and further so some frame lands in focus, and
upload it — the server keeps the frames with real ridge structure and discards
the rest.

**On Android Chrome, just press Capture.** Chrome is the only browser on any
platform that exposes real camera control to a web page — manual focus distance,
torch, zoom, exposure and full-resolution stills — so there the page searches
for the best shot instead of asking you to find it. Pressing Capture sweeps the
lens from its near limit out to 30 cm (uniformly in diopters, since that is what
lens position is linear in), scores every step with the same ridge detector the
enrolment gate uses, refines around the peak, then tries the torch on and off
and keeps whichever measures better. It uploads six candidates around that
optimum — three focus distances, two preview frames, and one shot under the
other light — and the server picks the winner. About 15-20 seconds of camera
work; the result card names the settings that won, and the capture trace under
it lists every measurement.

Validated: finger pointing roughly up, pad FACING the camera, close enough
to fill most of the frame (~15–20 cm), flash on, tap-to-focus on the
fingertip. Litmus test: ridges visible when you zoom the photo. Out-of-
protocol shots (pad averted, arm's length, dim) fail safe — no ridges to
extract → rejected with `QUALITY_TOO_LOW`/`NO_FINGER_FOUND`, not
misidentified. Pose beyond ±45° from enrollment does not match (enroll the
poses you expect to probe).

## Status

- [x] Matching core proven on FVC2000 (EER ~2%, 40/40 1:N) and on real
      thumbs (genuine 104–195 vs threshold 40; zero false accepts ever,
      including cross-person and 4-finger per-position trials)
- [x] 4-finger splitter (k-curvature fingertip detection) + per-position
      impostor validation; fusion scoring in `scripts/hands_test.py`
- [x] Persistent store: SQLite + originals, uuid identities, consent,
      audit log, versioned templates, optional Fernet encryption at rest
- [x] Production API: FastAPI, async jobs (202+poll), API-key auth, body
      limits, HEIC, problem+json errors with reason codes, health/readiness,
      JSON logs with request ids, Docker single-worker deployment
- [ ] 4+1 genuine fusion validation — needs a second protocol 4-finger
      capture per person; then wire finger-position fusion into identify
- [x] Ridge-structure quality gate (band-passed orientation coherence + edge
      density + normalized-scale spectral energy) refusing no-ridge crops
      before extraction; calibrated AUC 1.00 on 23 labelled crops
- [x] Evaluation harness: labelled manifest, cached template sets, EER/FTA/1:N
      report with baseline comparison (`scripts/evaluate.py`)
- [ ] More subjects → data-driven threshold calibration
- [ ] pyfing A/B for frequency estimation + enhancement
- [x] Video / multi-frame capture: server-side frame scoring and best-frame
      selection (`fingerphoto/frames.py`), torch where the browser exposes it,
      live ridge indicator on the capture page (`POST /v1/quality`)
- [x] Adaptive capture on Android Chrome: diopter-uniform focus sweep scored by
      the ridge detector, fine pass around the peak, torch A/B, conditional
      zoom, full-resolution stills, and a visible capture trace
- [ ] MediaRecorder clips recorded in-page on Android; a native shell for iOS
      (Capacitor plus a camera-preview plugin) to bring that same adaptive
      capture to iPhone, where WebKit exposes no focus or torch control
- [ ] Trained fingertip detector for cluttered/free-form shots
