"""Learned finger localization: SAM masks + CLIP text selection.

EXPERIMENTAL — measured WORSE than the color skin gate on our photos. Kept
because it is a documented negative result and may suit other capture styles;
do not enable it expecting an improvement without re-running the A/B.

SAM proposes every object in the frame, CLIP picks the one best matching "a
human finger", and the result is rotated so the fingertip points up, which is
what segment.crop_distal assumes. Feed the output straight into
segment.crop_distal(img, mask=mask).

Measured on the 8 real sample photos (scripts/ab_localizer.py, 2026-08-31),
SourceAFIS held constant, crop_distal path only:

    localizer     genuine (Q trio)      impostor max   photos yielding templates
    color gate    194.6 / 153.8 / 104.6        23.8    7 of 8
    SAM + CLIP     51.5 / 135.0 /  43.0        11.7    5 of 8

SAM suppresses impostors better (11.7 vs 23.8 — its crops carry no background
clutter to spawn spurious matches) but costs far more genuine margin
(104.6 -> 43.0 worst case, near the threshold of 40) and fails outright on the
dim photos the color gate still handles. Net: color gate wins on both margin
(4.4x vs 3.7x) and robustness, and is ~40 s/photo faster.

Approach credit: Ridgeformer's ISPFD preprocessing (Pandey et al., 2025) uses
SAM+CLIP for smartphone fingerphoto segmentation. The libraries here are
permissively licensed (SAM: Apache-2.0, CLIP: MIT) and the orchestration and
orientation logic below are this project's own, so nothing in this path
carries their CC-BY-NC terms.

Cost: SAM ViT-H is ~40 s/image on CPU and needs a 2.4 GB checkpoint.
Enable by setting FP_SAM_CHECKPOINT to the sam_vit_h_4b8939.pth path;
install extras with `pip install -r requirements-sam.txt`.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from .errors import AcquisitionError, ReasonCode

TEXT_PROMPT = "a photo of a human finger"
MAX_SIDE = 1600  # SAM sees plenty of finger at this scale; keeps runtime sane

_state: dict = {}


def available() -> bool:
    return bool(checkpoint_path() and checkpoint_path().exists())


def checkpoint_path() -> Path | None:
    p = os.environ.get("FP_SAM_CHECKPOINT")
    return Path(p) if p else None


def _load():
    if not _state:
        import torch
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        from transformers import AutoProcessor, CLIPModel

        ckpt = checkpoint_path()
        if not ckpt or not ckpt.exists():
            raise AcquisitionError(
                ReasonCode.NO_FINGER_FOUND,
                "SAM checkpoint not configured (set FP_SAM_CHECKPOINT)",
            )
        # SAM's mask generator casts to float64, which MPS rejects; CPU unless CUDA.
        sam_dev = "cuda" if torch.cuda.is_available() else "cpu"
        clip_dev = sam_dev if sam_dev == "cuda" else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        sam = sam_model_registry["vit_h"](checkpoint=str(ckpt)).to(device=sam_dev)
        _state["gen"] = SamAutomaticMaskGenerator(sam, output_mode="binary_mask")
        _state["clip"] = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(clip_dev)
        _state["proc"] = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _state["clip_dev"] = clip_dev
        _state["torch"] = torch
    return _state


def _best_finger_mask(rgb: np.ndarray) -> np.ndarray:
    """SAM proposals scored by CLIP against the finger prompt."""
    s = _load()
    torch = s["torch"]
    from PIL import Image as PILImage

    masks = s["gen"].generate(rgb)
    if not masks:
        raise AcquisitionError(ReasonCode.NO_FINGER_FOUND, "SAM found no regions")

    frame_area = rgb.shape[0] * rgb.shape[1]
    best, best_score = None, None
    for md in masks:
        m = md["segmentation"]
        frac = m.sum() / frame_area
        if not 0.05 <= frac <= 0.95:  # ignore specks and whole-frame masks
            continue
        obj = np.zeros_like(rgb)
        obj[m] = rgb[m]
        inputs = s["proc"](text=[TEXT_PROMPT], images=PILImage.fromarray(obj),
                           return_tensors="pt", padding=True).to(s["clip_dev"])
        with torch.no_grad():
            score = float(s["clip"](**inputs).logits_per_image.cpu().numpy().ravel()[0])
        if best_score is None or score > best_score:
            best, best_score = m, score
    if best is None:
        raise AcquisitionError(ReasonCode.NO_FINGER_FOUND, "no finger-like region found")
    return (best.astype(np.uint8)) * 255


def _upright(bgr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate so the finger enters from the bottom — i.e. the tip points up,
    which is what crop_distal assumes. Entry side = the border the finger
    crosses most (same idea as hand_split._entry)."""
    h, w = mask.shape
    strip = max(4, int(0.01 * max(h, w)))
    counts = {
        "bottom": cv2.countNonZero(mask[h - strip:, :]),
        "top": cv2.countNonZero(mask[:strip, :]),
        "left": cv2.countNonZero(mask[:, :strip]),
        "right": cv2.countNonZero(mask[:, w - strip:]),
    }
    side = max(counts, key=counts.get)
    if counts[side] == 0:  # finger fully inside the frame: use centroid vs tip
        ys, xs = np.nonzero(mask)
        side = "bottom" if ys.mean() > h / 2 else "top"
    rot = {
        "bottom": None,  # already tip-up
        "top": cv2.ROTATE_180,
        "left": cv2.ROTATE_90_CLOCKWISE,
        "right": cv2.ROTATE_90_COUNTERCLOCKWISE,
    }[side]
    if rot is None:
        return bgr, mask
    return cv2.rotate(bgr, rot), cv2.rotate(mask, rot)


def find_finger(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Locate the finger and return (upright BGR image, matching mask).
    Feed straight into segment.crop_distal(img, mask=mask).

    SAM runs on a downscaled copy for speed, but the mask is scaled back up
    and applied to the FULL-RESOLUTION image: ridge extraction needs every
    pixel it can get, and cropping from the downscaled copy costs ~2.5x of
    the ridge detail the matcher depends on."""
    h, w = bgr.shape[:2]
    scale = MAX_SIDE / max(h, w)
    work = cv2.resize(bgr, (int(w * scale), int(h * scale))) if scale < 1 else bgr
    rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
    mask = _best_finger_mask(rgb)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    if mask.shape[:2] != (h, w):  # back to full resolution
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return _upright(bgr, mask)
