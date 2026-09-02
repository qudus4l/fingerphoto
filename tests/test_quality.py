"""Ridge detector on synthetic images: a clean ridge pattern must score high,
and every way a real capture fails (blur, noise, flat skin, wrong scale) must
score low. No JVM, no photos on disk."""
from __future__ import annotations

import json
import time

import cv2
import numpy as np
import pytest

from fingerphoto import quality

N = 256


def sine(period: float = 8.0, angle: float = 0.0, amp: float = 60, n: int = N) -> np.ndarray:
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    t = np.radians(angle)
    return np.uint8(np.clip(128 + amp * np.sin(2 * np.pi * (x * np.cos(t) + y * np.sin(t)) / period), 0, 255))


def whorl(period: float = 8.0, n: int = N) -> np.ndarray:
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    r = np.hypot(x - n / 2, y - n / 2)
    return np.uint8(np.clip(128 + 60 * np.sin(2 * np.pi * r / period), 0, 255))


@pytest.mark.parametrize("angle", [0, 30, 90])
def test_sinusoid_ridges_score_high(angle):
    q = quality.ridge_quality(sine(angle=angle))
    assert q.valid and q.n_blocks > 100
    assert q.coherence_frac >= 0.9
    assert q.spectral_frac >= 0.8
    assert q.edge_density >= 0.08
    assert q.score >= 0.9


def test_whorl_scores_high():
    q = quality.ridge_quality(whorl())
    assert q.coherence_frac >= 0.7 and q.score >= 0.7


def test_blurred_ridges_score_low():
    """The failure this whole module exists for: the Gabor stage would still
    emit ridges here, so the score has to fall before extraction."""
    q = quality.ridge_quality(cv2.GaussianBlur(sine(), (0, 0), 4))
    assert q.score <= 0.1
    assert q.coherence_frac <= 0.2  # the energy floor, not the coherence itself


def test_noise_scores_low():
    rng = np.random.default_rng(0)
    q = quality.ridge_quality(rng.integers(0, 256, (N, N), dtype=np.uint8))
    assert q.coherence_frac <= 0.2 and q.spectral_frac <= 0.2 and q.score <= 0.15


def test_flat_skin_scores_zero():
    q = quality.ridge_quality(np.full((N, N), 128, np.uint8))
    assert q.score == 0.0 and q.energy_median == 0.0


def test_wrong_scale_scores_low():
    """Ridges at 24 px are outside the normalized band: the crop was rescaled
    against a wrong period estimate and is unusable to a 500-dpi matcher."""
    assert quality.ridge_quality(sine(period=24)).score <= 0.25


def test_empty_mask_is_invalid():
    q = quality.ridge_quality(sine(), np.zeros((N, N), np.uint8))
    assert q.valid is False and q.n_blocks == 0 and q.score == 0.0


def test_tiny_roi_is_invalid():
    q = quality.ridge_quality(sine(n=32))
    assert q.valid is False and q.n_blocks < quality.MIN_BLOCKS and q.score == 0.0


def test_mask_restricts_blocks():
    img = sine()
    left = np.zeros((N, N), np.uint8)
    left[:, : N // 2] = 255
    q = quality.ridge_quality(img, left)
    assert q.n_blocks == (N // quality.BLOCK) * (N // quality.BLOCK) // 2
    assert q.score >= 0.9


def test_none_mask_equals_full_mask():
    img = sine()
    assert (quality.ridge_quality(img).as_dict()
            == quality.ridge_quality(img, np.full((N, N), 255, np.uint8)).as_dict())


def test_mask_is_resized_to_the_image():
    img = sine()
    small = np.full((N // 2, N // 2), 255, np.uint8)
    assert quality.ridge_quality(img, small).valid


def test_accepts_colour_input():
    img = sine()
    assert quality.ridge_quality(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)).score >= 0.9


def test_as_dict_is_json_safe():
    d = quality.ridge_quality(sine()).as_dict()
    assert json.loads(json.dumps(d))["score"] == d["score"]
    assert set(d) == {"coherence_mean", "coherence_frac", "ocl_mean", "edge_density",
                      "spectral_frac", "energy_median", "n_blocks", "score", "valid"}


def test_combine_is_an_and_of_three_ramps():
    assert quality.combine(1.0, 1.0, 1.0) == 1.0
    assert quality.combine(0.0, 1.0, 1.0) == 0.0      # any factor at zero sinks it
    assert quality.combine(1.0, 1.0, 0.0) == 0.0
    assert quality.combine(1.0, 1.0, 0.0, use_spectral=False) == 1.0
    mid = quality.combine((quality.CF_LO + quality.CF_HI) / 2, 1.0, 1.0)
    assert abs(mid - 0.5) < 1e-6


def test_runtime_budget():
    """Has to be cheap enough to run on every crop, and later every video frame."""
    img = cv2.resize(sine(n=480), (640, 480))
    quality.ridge_quality(img)  # warm the scipy import
    t0 = time.perf_counter()
    quality.ridge_quality(img)
    assert time.perf_counter() - t0 < 0.25
