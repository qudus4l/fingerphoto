"""Identity layer: photos -> multi-scale template sets -> 1:N identification.

An enrolled identity is a SET of templates: several photos, each extracted at
the estimated ridge period and at +/-13% scale variants. Set-vs-set score is
the max over the cross product — multi-sample enrollment is the single biggest
accuracy lever in the contactless literature, and the scale sweep absorbs the
residual error the per-image period estimator can't remove.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from . import afis
from .errors import AcquisitionError, ReasonCode
from .pipeline import photo_to_ridge_image
from .quality import MIN_RIDGE_QUALITY
from .scale import ridge_evidence, ridge_period_band_for_fingertip
from .segment import crop_distal

SCALE_SWEEP = (0.87, 1.0, 1.15)
MIN_MINUTIAE = 15  # refuse templates with fewer minutiae
# A localizer crop covering more of the frame than this is a hand or a scene,
# not a fingertip. Only applied to the texture-localizer path: the colour crop
# legitimately fills the frame on a close-up.
MAX_CROP_FRAME_FRAC = 0.30
NO_RIDGES_PREFIX = "no resolvable ridge structure"
GATE_REASONS = (
    "ridge_quality", "period_outside_band", "period_at_band_edge",
    "crop_too_large", "enhancement_failed",
)
# Minutiae count does NOT prove a print was captured: the Gabor stage is an
# oriented band-pass filter, so fed an out-of-focus finger it emits ridge-like
# texture and SourceAFIS finds minutiae in the hallucination. That is what
# quality.ridge_quality + crop_sanity_reason below are for; they run on the
# scale-normalized grayscale BEFORE enhancement. scale.ridge_evidence was an
# earlier attempt that did not separate (see its docstring) and is kept only as
# a logged diagnostic.


def crop_sanity_reason(
    info: dict,
    strategy: str,
    crop_shape: tuple[int, int],
    frame_shape: tuple[int, int] | None,
) -> str | None:
    """Geometric sanity of a crop, independent of its ridge quality.

    The ridge-period band is derived from the crop width (a fingertip pad is
    ~25-90 ridges across), so a seed period outside it means the crop is not a
    fingertip at all, and a final period pinned to a limit means the scale loop
    was still pushing outwards when it stopped."""
    if info.get("period_clamped"):
        return "period_outside_band"
    if info.get("period_at_band_edge"):
        return "period_at_band_edge"
    if strategy == "pad" and frame_shape is not None:
        frame_area = float(frame_shape[0] * frame_shape[1])
        if frame_area > 0 and crop_shape[0] * crop_shape[1] > MAX_CROP_FRAME_FRAC * frame_area:
            return "crop_too_large"
    return None


def _templates_for_crop(
    bgr: np.ndarray,
    mask: np.ndarray | None,
    hint: float | None,
    sweep: tuple[float, ...],
    out_dir: Path | None,
    min_minutiae: int = MIN_MINUTIAE,
    seen: list[float] | None = None,
    min_quality: float = MIN_RIDGE_QUALITY,
    strategy: str = "distal",
    frame_shape: tuple[int, int] | None = None,
    report: dict | None = None,
) -> list[bytes]:
    """Templates for one crop, or [] if the crop is refused.

    The refusal happens before the scale sweep, so a crop with no ridges costs
    one enhancement pass instead of three."""
    if report is None:
        report = {}
    band = ridge_period_band_for_fingertip(bgr.shape[1])
    if seen is not None:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
        seen.append(ridge_evidence(gray, band[0], band[1])[1])
    ridge, info = photo_to_ridge_image(
        bgr, out_dir=out_dir, period_band=band, mask=mask, initial_period=hint
    )
    q = info.get("quality") or {}
    report.update(
        strategy=strategy,
        quality=q,
        period_px=info.get("ridge_period_px"),
        initial_period_px=info.get("initial_period_px"),
        band_px=info.get("period_band_px"),
        scale_converged=info.get("scale_converged"),
        crop_shape=list(bgr.shape[:2]),
        frame_frac=(
            round(bgr.shape[0] * bgr.shape[1] / float(frame_shape[0] * frame_shape[1]), 3)
            if frame_shape else None
        ),
    )

    reason = crop_sanity_reason(info, strategy, bgr.shape[:2], frame_shape)
    if reason is None and min_quality > 0 and (
        not q.get("valid") or q.get("score", 0.0) < min_quality
    ):
        reason = "ridge_quality"
    if reason is not None:
        report["rejected"] = reason
        return []

    base = info["ridge_period_px"]
    templates = []
    for m in sweep:
        try:
            if m == 1.0:
                r = ridge
            else:
                r, _ = photo_to_ridge_image(bgr, mask=mask, period=base * m, quality=False)
            t = afis.extract_template(r)
        except ValueError:
            continue  # this scale variant degenerated; others may survive
        n = afis.minutiae_count(t)
        if n is None or n >= min_minutiae:
            templates.append(t)
    report["templates"] = len(templates)
    return templates


def _quality_detail(crops: dict, min_quality: float) -> str:
    """Message a person can act on: which crop failed, how, and by how much."""
    parts = []
    for name, r in crops.items():
        q = r.get("quality") or {}
        parts.append(
            f"{name} score {q.get('score', 0.0):.2f} (coherence {q.get('coherence_frac', 0.0):.2f},"
            f" edges {q.get('edge_density', 0.0):.3f}, spectral {q.get('spectral_frac', 0.0):.2f},"
            f" period {r.get('period_px')} px, reason {r.get('rejected')})"
        )
    return (
        f"{NO_RIDGES_PREFIX} in any crop (min score {min_quality}): " + "; ".join(parts)
        + ". Move closer, tap the fingertip to focus, and add light."
    )


def photo_template_set(
    bgr: np.ndarray,
    crop: bool = True,
    sweep: tuple[float, ...] = SCALE_SWEEP,
    out_dir: Path | None = None,
    min_minutiae: int = MIN_MINUTIAE,
    stats: dict | None = None,
    min_quality: float = MIN_RIDGE_QUALITY,
) -> list[bytes]:
    """All templates for one photo — the union over crop strategies (upright
    color crop AND texture localization, when both work) and scale variants.
    Set-based max scoring downstream picks whichever crop was right, so a
    photo that suits one strategy never loses to the other's failure.

    Crops with no resolvable ridge structure are refused before extraction
    (min_quality=0 disables that; the geometric sanity rules always apply).
    Raises AcquisitionError (a ValueError) if nothing survives; `stats`, when
    given, is filled with per-stage wall times, per-crop reports and counts."""
    if stats is None:
        stats = {}
    ev: list[float] = []  # ridge-frequency energy seen per crop strategy
    crops: dict[str, dict] = {}
    stats["crops"] = crops
    quality_per_template: list[float] = []
    frame_shape = bgr.shape[:2]

    def finish(templates: list[bytes]) -> list[bytes]:
        stats["templates"] = len(templates)
        stats["ridge_evidence"] = round(max(ev, default=0.0), 2)
        stats["template_quality"] = quality_per_template
        stats["quality"] = round(
            max((r.get("quality", {}).get("score", 0.0)
                 for r in crops.values() if "rejected" not in r), default=0.0), 3
        )
        stats["rejections"] = {k: r["rejected"] for k, r in crops.items() if "rejected" in r}
        return templates

    def run(name: str, c: np.ndarray, m: np.ndarray | None, hint: float | None,
            dest: Path | None) -> list[bytes]:
        rep: dict = {}
        crops[name] = rep
        got = _templates_for_crop(
            c, m, hint, sweep, dest, min_minutiae, seen=ev, min_quality=min_quality,
            strategy=name, frame_shape=frame_shape, report=rep,
        )
        quality_per_template.extend(
            [rep.get("quality", {}).get("score", 0.0)] * len(got)
        )
        return got

    if not crop:
        t0 = time.monotonic()
        try:
            templates = run("full", bgr, None, None, out_dir)
        except (ValueError, cv2.error):
            crops.setdefault("full", {})["rejected"] = "enhancement_failed"
            templates = []
        stats["full_frame_s"] = round(time.monotonic() - t0, 3)
        if not templates:
            finish(templates)
            raise AcquisitionError(
                ReasonCode.QUALITY_TOO_LOW,
                _quality_detail(crops, min_quality)
                if _all_gated(crops)
                else f"no scale variant produced >= {min_minutiae} minutiae",
            )
        return finish(templates)

    templates: list[bytes] = []
    crops_found = 0
    t0 = time.monotonic()
    try:  # proven path for protocol (finger-up) captures
        c, m = crop_distal(bgr)
        crops_found += 1
        templates += run("distal", c, m, None, out_dir)
    except (ValueError, cv2.error):
        if "distal" in crops:  # the crop existed; the pipeline gave up on it
            crops["distal"].setdefault("rejected", "enhancement_failed")
    stats["crop_distal_s"] = round(time.monotonic() - t0, 3)
    t0 = time.monotonic()
    try:  # orientation-free texture localization (tilted/sideways fingers)
        from .locate import crop_pad

        c, m, hint = crop_pad(bgr)
        crops_found += 1
        templates += run("pad", c, m, hint, out_dir / "pad" if out_dir else None)
    except (ValueError, cv2.error):
        if "pad" in crops:
            crops["pad"].setdefault("rejected", "enhancement_failed")
    stats["crop_pad_s"] = round(time.monotonic() - t0, 3)

    if not templates:
        finish(templates)
        if crops_found == 0:
            raise AcquisitionError(
                ReasonCode.NO_FINGER_FOUND, "no crop strategy located a fingertip"
            )
        if _all_gated(crops):
            raise AcquisitionError(
                ReasonCode.QUALITY_TOO_LOW, _quality_detail(crops, min_quality)
            )
        raise AcquisitionError(
            ReasonCode.QUALITY_TOO_LOW,
            f"no crop strategy produced >= {min_minutiae} minutiae"
            f" (best ridge quality {stats['quality']})",
        )
    return finish(templates)


def _all_gated(crops: dict) -> bool:
    """True when every crop was refused by the ridge gate or a sanity rule,
    rather than merely running short of minutiae."""
    return bool(crops) and all(r.get("rejected") in GATE_REASONS for r in crops.values())


def set_score(probe_set: list[bytes], enrolled_set: list[bytes]) -> float:
    return max(max(afis.match_many(p, enrolled_set)) for p in probe_set)


def verify(
    probe_set: list[bytes],
    enrolled_set: list[bytes],
    threshold: float = afis.THRESHOLD,
) -> tuple[bool, float]:
    """1:1 verification: does the probe belong to this enrolled identity?"""
    score = set_score(probe_set, enrolled_set)
    return score >= threshold, score


def enroll(photos: list[np.ndarray], crop: bool = True) -> list[bytes]:
    """Identity = union of template sets from several photos of one finger."""
    identity: list[bytes] = []
    for bgr in photos:
        identity.extend(photo_template_set(bgr, crop=crop))
    return identity


def identify(
    probe_set: list[bytes],
    gallery: dict[str, list[bytes]],
    threshold: float = afis.THRESHOLD,
) -> tuple[str | None, float]:
    """Best (identity, score) if the score clears the threshold, else (None, best)."""
    best_name, best = None, -1.0
    for name, enrolled in gallery.items():
        s = set_score(probe_set, enrolled)
        if s > best:
            best_name, best = name, s
    if best < threshold:
        return None, best
    return best_name, best
