"""4-finger fusion validation with MANUAL finger axes (decoupled from the
auto-splitter): per-finger templates for Q's and sister's protocol hand shots,
then per-position impostor matrix + fused scores, plus the genuine tests the
current data supports (sister's index close-ups vs her hand's index).
Axes read off the photos by hand: (tip, distal joint) in original px.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fingerphoto import afis, identity
from fingerphoto.locate import crop_guided
from fingerphoto.pipeline import photo_to_ridge_image

PHOTO_DIR = ROOT / "data" / "samples" / "photo"
OUT = ROOT / "out" / "fusion"

HAND_PHOTOS = ["hand_c", "sis_hand_c"]  # both: fingers pointing up
FINGER_NAMES = ["index", "middle", "ring", "little"]
BAND = (6.0, 24.0)


def fingers_up_axes(skin: "np.ndarray"):
    """k-curvature fingertip detection: fingertip caps are sustained
    high-curvature CONVEX arcs of the contour (webbings are concave, the
    seams between touching fingers are straight). Convexity test: the chord
    midpoint lies inside the mask. Returns [(tip, joint, width)] ordered
    left-to-right. No manual coordinates, works with touching fingers."""
    import numpy as np
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks

    contours, _ = cv2.findContours(skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
    n = len(cnt)
    k = 220  # ~px along the arc; sides of a ~400px-wide fingertip cap
    if n < 3 * k:
        raise ValueError("contour too short")

    a = np.roll(cnt, k, axis=0)
    c = np.roll(cnt, -k, axis=0)
    v1 = a - cnt
    v2 = c - cnt
    cosang = (v1 * v2).sum(axis=1) / (
        np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9
    )
    sharp = np.pi - np.arccos(np.clip(cosang, -1, 1))

    mid = ((a + c) / 2).astype(int)
    h, w = skin.shape
    mid[:, 0] = np.clip(mid[:, 0], 0, w - 1)
    mid[:, 1] = np.clip(mid[:, 1], 0, h - 1)
    convex = skin[mid[:, 1], mid[:, 0]] > 0
    # exclude the frame edges AND the bottom strip: the hand enters from the
    # bottom in fingers-up photos, so high-curvature corners there are wrist/
    # knuckle artifacts, not fingertips
    inbounds = (
        (cnt[:, 0] > 12)
        & (cnt[:, 0] < w - 12)
        & (cnt[:, 1] > 12)
        & (cnt[:, 1] < h - 0.18 * h)
    )
    score = np.where(convex & inbounds, sharp, 0.0)
    score = uniform_filter1d(score, size=31, mode="wrap")

    peaks, props = find_peaks(
        np.concatenate([score, score[:k]]), prominence=0.25, distance=2 * k
    )
    peaks = np.unique(peaks % n)
    if len(peaks) == 0:
        raise ValueError("no fingertip curvature peaks")
    order = np.argsort(-score[peaks])[:4]
    tips_idx = peaks[order]

    axes = []
    for i in tips_idx:
        tip = cnt[i]
        chord_mid = (a[i] + c[i]) / 2
        width = float(np.linalg.norm(a[i] - c[i]))
        u = tip - chord_mid
        u = u / (np.linalg.norm(u) + 1e-9)
        joint = tip - u * 1.2 * width
        axes.append((tuple(tip), tuple(joint), width))
    axes.sort(key=lambda t: t[0][0])  # left-to-right
    return axes


def finger_templates(bgr, tip, joint, width, out_dir: Path) -> list[bytes]:
    import numpy as np

    crop, mask = crop_guided(bgr, tip, joint, width=width)
    # fingers touch: restrict the mask to the central finger strip so the
    # neighbors' slivers can't contribute (shared) minutiae
    cx = crop.shape[1] // 2
    half = int(0.42 * width)
    strip = np.zeros_like(mask)
    strip[:, max(0, cx - half) : cx + half] = 255
    mask = cv2.bitwise_and(mask, strip)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "0_crop.png"), crop)
    ridge, info = photo_to_ridge_image(crop, out_dir=out_dir, period_band=BAND, mask=mask)
    base = info["ridge_period_px"]
    out = []
    for m in identity.SCALE_SWEEP:
        try:
            r = ridge if m == 1.0 else photo_to_ridge_image(crop, mask=mask, period=base * m)[0]
            t = afis.extract_template(r)
        except ValueError:
            continue
        n = afis.minutiae_count(t)
        if n is None or n >= identity.MIN_MINUTIAE:
            out.append(t)
    return out


def main() -> None:
    from fingerphoto.segment import skin_mask

    hands: dict[str, dict[str, list[bytes]]] = {}
    for name in HAND_PHOTOS:
        bgr = cv2.imread(str(PHOTO_DIR / f"{name}.jpg"))
        axes = fingers_up_axes(skin_mask(bgr))
        hands[name] = {}
        for finger, (tip, joint, width) in zip(FINGER_NAMES, axes):
            s = finger_templates(bgr, tip, joint, width, OUT / f"{name}_{finger}")
            if s:
                hands[name][finger] = s
            print(f"  {name}/{finger}: tip=({tip[0]:.0f},{tip[1]:.0f}) w={width:.0f} {len(s)} templates")

    thr = afis.THRESHOLD
    q, sis = hands["hand_c"], hands["sis_hand_c"]

    print(f"\nper-position Q vs sister (ALL must be impostors < {thr}):")
    fused_impostor = 0.0
    for finger in ("index", "middle", "ring", "little"):
        if finger in q and finger in sis:
            s = identity.set_score(q[finger], sis[finger])
            fused_impostor += s
            print(f"  {finger:6s}: {s:6.1f}  {'ok' if s < thr else 'FALSE ACCEPT'}")
    print(f"  fused impostor score: {fused_impostor:.1f}")

    print("\nsame hand, DIFFERENT fingers (Q internal impostors):")
    fingers = [f for f in ("index", "middle", "ring", "little") if f in q]
    from itertools import combinations
    for a, b in combinations(fingers, 2):
        s = identity.set_score(q[a], q[b])
        print(f"  {a} vs {b}: {s:6.1f}  {'ok' if s < thr else 'FALSE ACCEPT'}")

    print("\ngenuine: sister's index close-ups vs her hand's index:")
    for n in ("sis_finger_a", "sis_finger_b"):
        bgr = cv2.imread(str(PHOTO_DIR / f"{n}.jpg"))
        try:
            probe = identity.photo_template_set(bgr, out_dir=OUT / n)
        except ValueError as e:
            print(f"  {n}: unusable ({e})")
            continue
        for finger in ("index", "middle", "ring", "little"):
            if finger in sis:
                s = identity.set_score(probe, sis[finger])
                mark = " <-- should be highest" if finger == "index" else ""
                print(f"  {n} vs sister {finger:6s}: {s:6.1f}{mark}")

    print("\ngenuine (weak-capture caveat): Q hand_c vs old soft captures, best per-finger:")
    from fingerphoto.hand_split import split_fingers
    for old in ("hand_a", "hand_b"):
        bgr = cv2.imread(str(PHOTO_DIR / f"{old}.jpg"))
        try:
            old_fingers = split_fingers(bgr)
        except ValueError:
            continue
        best_overall = 0.0
        for _, crop, mask in old_fingers:
            try:
                ridge, info = photo_to_ridge_image(crop, period_band=BAND, mask=mask)
                t = afis.extract_template(ridge)
            except ValueError:
                continue
            n = afis.minutiae_count(t)
            if n is not None and n < identity.MIN_MINUTIAE:
                continue
            for finger, fs in q.items():
                best_overall = max(best_overall, max(afis.match_many(t, fs)))
        print(f"  {old} best single score vs hand_c: {best_overall:.1f}")


if __name__ == "__main__":
    main()
