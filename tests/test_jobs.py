from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from fingerphoto.api.jobs import JobStore
from fingerphoto.errors import AcquisitionError, ReasonCode, StoreError


def _wait(job, timeout=5.0):
    deadline = time.time() + timeout
    while job.status in ("pending", "running"):
        if time.time() > deadline:
            raise TimeoutError(job.status)
        time.sleep(0.01)
    return job


@pytest.fixture()
def jobs():
    executor = ThreadPoolExecutor(max_workers=1)
    yield JobStore(executor, max_queue=2, ttl_s=3600)
    executor.shutdown(wait=False, cancel_futures=True)


def test_success(jobs):
    job = _wait(jobs.submit("t", lambda: {"answer": 42}))
    assert job.status == "succeeded" and job.result == {"answer": 42}
    assert jobs.get(job.id) is job


def test_coded_failure(jobs):
    job = _wait(
        jobs.submit("t", lambda: (_ for _ in ()).throw(
            AcquisitionError(ReasonCode.QUALITY_TOO_LOW, "too few minutiae")))
    )
    assert job.status == "failed"
    assert job.error == {"reason_code": "QUALITY_TOO_LOW", "detail": "too few minutiae"}


def test_unexpected_failure_masked(jobs):
    job = _wait(jobs.submit("t", lambda: 1 / 0))
    assert job.status == "failed"
    assert job.error["reason_code"] == "INTERNAL_ERROR"
    assert "ZeroDivision" not in job.error["detail"]  # internals stay in server logs


def test_queue_full(jobs):
    release = threading.Event()
    jobs.submit("t", lambda: release.wait(5) or {})  # occupies the single worker
    jobs.submit("t", lambda: {})  # queued
    with pytest.raises(StoreError) as e:
        jobs.submit("t", lambda: {})
    assert e.value.code is ReasonCode.QUEUE_FULL
    release.set()


def test_idempotency_key(jobs):
    a = jobs.submit("t", lambda: {}, idempotency_key="abc")
    b = jobs.submit("t", lambda: {}, idempotency_key="abc")
    assert a.id == b.id


def test_ttl_sweep():
    executor = ThreadPoolExecutor(max_workers=1)
    jobs = JobStore(executor, max_queue=2, ttl_s=0)
    job = _wait(jobs.submit("t", lambda: {}))
    time.sleep(0.02)
    assert jobs.get(job.id) is None
    executor.shutdown(wait=False)
