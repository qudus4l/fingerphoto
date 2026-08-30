"""FastAPI application factory.

Process model (deliberate, load-bearing): exactly ONE uvicorn worker process.
The JVM starts once per process and can never restart; the in-process job
store needs every poll to land in the process that runs the job; SQLite gets
a single writer process. CPU parallelism comes from the thread pool — OpenCV
and JPype release the GIL for the heavy work. Scaling out means an external
queue + shared DB, not more workers.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import numpy as np
from fastapi import Depends, FastAPI

from .. import afis, store, version
from .errors import install_handlers
from .jobs import JobStore
from .logging import BodyLimitMiddleware, RequestIdMiddleware, setup_logging
from .routes import identities, recognition, system
from .security import require_api_key
from .settings import Settings, get_settings


def _export_core_env(settings: Settings) -> None:
    """Core modules (store, afis) read FP_* from the environment so the CLI
    works without this package; push .env-sourced values back so both layers
    resolve identical config."""
    os.environ["FP_DB_PATH"] = str(settings.db_path)
    os.environ["FP_DATA_DIR"] = str(settings.data_dir)
    os.environ["FP_THRESHOLD"] = str(settings.threshold)
    os.environ["FP_MIN_MINUTIAE"] = str(settings.min_minutiae)
    os.environ["FP_STORE_ORIGINALS"] = "true" if settings.store_originals else "false"
    os.environ["FP_JVM_HEAP"] = settings.jvm_heap
    if settings.encryption_key:
        os.environ["FP_ENCRYPTION_KEY"] = settings.encryption_key


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if not settings.keys() and not settings.dev_no_auth:
        raise RuntimeError(
            "FP_API_KEYS is empty — set at least one key, or FP_DEV_NO_AUTH=1 for local dev"
        )
    _export_core_env(settings)

    con = store.connect()  # run migrations before serving traffic
    con.close()

    afis._ensure_jvm()  # JVM starts here: once per process, after any fork
    try:  # warm the extractor classes so the first real request doesn't pay for them
        afis.extract_template(np.full((96, 96), 128, np.uint8))
    except Exception:
        pass
    app.state.jvm_ready = True

    app.state.executor = ThreadPoolExecutor(
        max_workers=settings.cpu_workers, thread_name_prefix="fp-cpu"
    )
    app.state.jobs = JobStore(app.state.executor, settings.max_queue, settings.job_ttl_s)
    yield
    app.state.executor.shutdown(wait=False, cancel_futures=True)


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings.log_level)
    app = FastAPI(title="fingerphoto", version=version.APP_VERSION, lifespan=lifespan)
    app.include_router(system.public_router)
    app.include_router(system.router, dependencies=[Depends(require_api_key)])
    app.include_router(identities.router, dependencies=[Depends(require_api_key)])
    app.include_router(recognition.router, dependencies=[Depends(require_api_key)])
    install_handlers(app)
    app.add_middleware(BodyLimitMiddleware, max_bytes=settings.max_upload_mb * 1024 * 1024)
    app.add_middleware(RequestIdMiddleware)  # added last -> outermost
    return app


app = create_app()
