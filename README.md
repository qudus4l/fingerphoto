# fingerphoto

Fingerprint recognition from phone-camera photos: enroll a finger from
photos, then identify it in any later photo. Photo-to-photo matching — no
contact scanner involved. Ships as an HTTP service (FastAPI) with a CLI for
local use.

## How it works

```
photo ─▶ skin-mask distal crop ─▶ grayscale
      ─▶ ridge-frequency scale normalization, closed-loop (pseudo-500dpi)
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
| `POST /v1/identities/{uuid}/photos` | Enroll photos (multipart `files`, optional `finger` position) → 202 |
| `POST /v1/identify` | 1:N search of one photo against everyone → 202 |
| `POST /v1/identities/{uuid}/verify` | 1:1 check of a photo against a claimed identity → 202 |
| `GET /v1/jobs/{id}` | Poll a job; per-photo results carry machine-readable `reason_code`s |
| `DELETE /v1/identities/{uuid}` | Hard delete: templates, rows, original photos |
| `GET /healthz` · `GET /readyz` · `GET /v1/version` | Liveness / readiness (JVM+DB+queue) / versions |

Finger positions use ANSI/NIST-ITL names: `right_thumb` … `left_little`,
default `unknown`. JPEG/PNG/HEIC accepted (iPhone HEIC works natively).
Uploads over `FP_MAX_UPLOAD_MB` (default 20) are rejected with 413. Optional
`Idempotency-Key` header makes a repeated submission return the same job.

Example session:

```sh
curl -s -H "X-API-Key: $KEY" -H 'Content-Type: application/json' \
  -d '{"name":"Q","consent":true}' localhost:8000/v1/identities
curl -s -H "X-API-Key: $KEY" -F files=@thumb1.jpg -F files=@thumb2.jpg \
  -F finger=right_thumb localhost:8000/v1/identities/<uuid>/photos
curl -s -H "X-API-Key: $KEY" localhost:8000/v1/jobs/<job_id>
curl -s -H "X-API-Key: $KEY" -F file=@probe.jpg localhost:8000/v1/identify
```

## CLI (same store as the API)

```sh
./.venv/bin/python scripts/enroll.py NAME photo1.jpg photo2.jpg
./.venv/bin/python scripts/identify.py probe.jpg
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
before trusting the FMR number in production. Quality gating is
minutiae-count only (`FP_MIN_MINUTIAE=15`); a marginal photo can still
enroll a weak template — harmless under set-max scoring, but a real quality
score (NFIQ2 is miscalibrated for contactless) is future work.

**Versioning.** Every template row is stamped with `pipeline_version` +
`sourceafis_version`; `GET /v1/version` reports both. Bump
`PIPELINE_VERSION` in `fingerphoto/version.py` whenever extraction
parameters change, then re-extract stale templates from originals.

## Tests

```sh
make test-fast   # unit + API-shape tests, no JVM (~3 s)
make test        # + golden-score regression on real photos and full API e2e (minutes)
```

Golden bands encode this project's validated numbers: genuine thumb trio
≥ 100 (observed 104–195), impostors < 30 (observed ≤ 29), plus a
two-thread concurrent-extraction test (JVM lock + enhancer thread-safety).
The print-style deep-regression scripts still exist
(`scripts/subjects_test.py`, `fvc_benchmark.py`, `hands_test.py`, ...).

## Capture protocol (matters enormously)

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
- [ ] More subjects → data-driven threshold calibration
- [ ] pyfing A/B for frequency estimation + enhancement
- [ ] Guided capture UX (the single biggest quality lever left)
- [ ] Trained fingertip detector for cluttered/free-form shots
