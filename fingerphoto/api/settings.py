"""Environment-driven configuration (FP_ prefix).

The core modules (store, afis) read the same FP_* environment variables
directly so the CLI works without this package; Settings is the API layer's
typed view of them plus the API-only knobs.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FP_", env_file=".env", extra="ignore")

    db_path: Path = ROOT / "data" / "identities.db"
    data_dir: Path = ROOT / "data"
    threshold: float = 40.0
    min_minutiae: int = 15
    min_ridge_quality: float = 0.40  # keep in sync with quality.MIN_RIDGE_QUALITY
    api_keys: str = ""  # comma-separated; empty refuses to start unless dev_no_auth
    dev_no_auth: bool = False
    cpu_workers: int = 2
    max_queue: int = 8
    # 64 MB, not 20: a 5 s 4K clip from an iPhone is 15-35 MB and the video
    # capture path is the one that actually produces matchable frames.
    max_upload_mb: int = 64
    video_frames_enroll: int = 3  # frames kept per enrollment clip (each costs a full extraction)
    video_frames_identify: int = 2
    store_originals: bool = True
    encryption_key: str = ""
    job_ttl_s: int = 3600
    jvm_heap: str = "512m"
    log_level: str = "INFO"

    def keys(self) -> list[str]:
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
