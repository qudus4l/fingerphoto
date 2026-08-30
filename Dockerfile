FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
      default-jre-headless curl bash libxcb1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# fingerprint-enhancer drags in GL-linked opencv-python; strip it so the
# headless wheel's cv2 wins (no X/GL libraries in this image)
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y opencv-python \
    && pip install --no-cache-dir --force-reinstall opencv-python-headless

COPY scripts/fetch_jars.sh scripts/
RUN bash scripts/fetch_jars.sh

COPY fingerphoto/ fingerphoto/

RUN useradd -r app && mkdir /data && chown app /data
USER app
ENV FP_DATA_DIR=/data FP_DB_PATH=/data/identities.db

HEALTHCHECK --interval=30s --timeout=3s --start-period=30s \
  CMD curl -sf http://localhost:8000/healthz || exit 1
EXPOSE 8000

# Exactly ONE worker: the JVM starts once per process and cannot restart,
# the in-process job store needs polls to hit the process running the job,
# and SQLite wants a single writer. Scale-out = external queue, not workers.
CMD ["uvicorn", "fingerphoto.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
