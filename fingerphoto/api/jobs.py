"""In-process job store for the 202+poll pattern.

Deliberate trade-off (documented in the README): jobs live in this process
only and are lost on restart — clients re-submit. That is what keeps the
deployment single-process, which is what keeps the one-JVM/SQLite model
correct. Scale-out later means an external queue.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from ..errors import CodedError, ReasonCode, StoreError
from .logging import request_id_var

logger = logging.getLogger("fingerphoto.jobs")


@dataclass
class Job:
    id: str
    kind: str
    status: str = "pending"  # pending | running | succeeded | failed
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: dict | None = None
    error: dict | None = None
    request_id: str | None = None


class JobStore:
    def __init__(self, executor: ThreadPoolExecutor, max_queue: int, ttl_s: int):
        self._executor = executor
        self._max_queue = max_queue
        self._ttl_s = ttl_s
        self._jobs: dict[str, Job] = {}
        self._by_idempotency: dict[str, str] = {}
        self._lock = threading.Lock()

    def _sweep_locked(self) -> None:
        cutoff = time.time() - self._ttl_s
        dead = [
            jid
            for jid, j in self._jobs.items()
            if j.finished_at is not None and j.finished_at < cutoff
        ]
        for jid in dead:
            del self._jobs[jid]
        self._by_idempotency = {
            k: v for k, v in self._by_idempotency.items() if v in self._jobs
        }

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.status in ("pending", "running"))

    def saturated(self) -> bool:
        return self.active_count() >= self._max_queue

    def submit(
        self, kind: str, fn: Callable[[], dict], idempotency_key: str | None = None
    ) -> Job:
        rid = request_id_var.get()
        with self._lock:
            self._sweep_locked()
            if idempotency_key:
                existing = self._by_idempotency.get(idempotency_key)
                if existing:
                    return self._jobs[existing]
            active = sum(1 for j in self._jobs.values() if j.status in ("pending", "running"))
            if active >= self._max_queue:
                raise StoreError(
                    ReasonCode.QUEUE_FULL,
                    f"{active} jobs already queued; retry after the current work drains",
                )
            job = Job(id=uuid.uuid4().hex, kind=kind, request_id=rid)
            self._jobs[job.id] = job
            if idempotency_key:
                self._by_idempotency[idempotency_key] = job.id

        def run() -> None:
            token = request_id_var.set(job.request_id)
            job.status, job.started_at = "running", time.time()
            try:
                job.result = fn()
                job.status = "succeeded"
            except CodedError as e:
                job.error = {"reason_code": e.code.value, "detail": str(e)}
                job.status = "failed"
            except Exception:
                logger.exception("job failed", extra={"job_id": job.id, "kind": job.kind})
                job.error = {
                    "reason_code": ReasonCode.INTERNAL_ERROR.value,
                    "detail": "unexpected error; see server logs",
                }
                job.status = "failed"
            finally:
                job.finished_at = time.time()
                logger.info(
                    "job finished",
                    extra={
                        "job_id": job.id,
                        "kind": job.kind,
                        "status": job.status,
                        "duration_s": round(job.finished_at - (job.started_at or job.finished_at), 3),
                    },
                )
                request_id_var.reset(token)

        self._executor.submit(run)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            self._sweep_locked()
            return self._jobs.get(job_id)
