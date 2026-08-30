"""Impostor test: sister's thumb vs Q's enrolled identity.

The sister's photos are outside the guided-capture protocol (tilted, distant,
pad partly averted), so the pad is cropped with manually specified tip/joint
points (crop_guided) — we are testing the MATCHER's impostor behavior here,
not the auto-localizer. Q's photos go through the normal auto pipeline.
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
from fingerphoto.scale import ridge_period_band_for_fingertip

PHOTO_DIR = ROOT / "data" / "samples" / "photo"
OUT = ROOT / "out"

# (tip, distal joint) in original pixel coords, read off the photos by hand
SISTER_AXES = {
    "sister_6175": ((1555, 970), (2262, 1676)),
    "sister_6176": ((1212, 788), (2182, 1576)),
}


def sister_template_set(name: str) -> list[bytes]:
    bgr = cv2.imread(str(PHOTO_DIR / f"{name}.jpg"))
    tip, joint = SISTER_AXES[name]
    crop, mask = crop_guided(bgr, tip, joint)
    out_dir = OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "0_crop.png"), crop)

    # absolute band: hand-picked axis points make the crop size unreliable,
    # so don't derive the search band from it
    band = (8.0, 22.0)
    ridge, info = photo_to_ridge_image(crop, out_dir=out_dir, period_band=band, mask=mask)
    base = info["ridge_period_px"]
    templates = []
    for m in identity.SCALE_SWEEP:
        try:
            r = ridge if m == 1.0 else photo_to_ridge_image(crop, mask=mask, period=base * m)[0]
            t = afis.extract_template(r)
        except ValueError:
            continue
        n = afis.minutiae_count(t)
        if n is None or n >= identity.MIN_MINUTIAE:
            templates.append(t)
    print(f"  {name}: base period {base}px, {len(templates)} usable templates")
    return templates


def main() -> None:
    print("Q's thumb (auto pipeline):")
    q_sets = {}
    for p in sorted(PHOTO_DIR.glob("thumb_*.jpg")):
        q_sets[p.stem] = identity.photo_template_set(cv2.imread(str(p)))
        print(f"  {p.stem}: {len(q_sets[p.stem])} templates")

    print("sister's thumb (guided crop):")
    s_sets = {name: sister_template_set(name) for name in SISTER_AXES}
    s_sets = {k: v for k, v in s_sets.items() if v}

    thr = afis.THRESHOLD
    print(f"\nsister vs sister (genuine), threshold {thr}:")
    names = sorted(s_sets)
    if len(names) >= 2:
        s = identity.set_score(s_sets[names[0]], s_sets[names[1]])
        print(f"  {names[0]} vs {names[1]}: {s:6.1f}  {'MATCH' if s >= thr else 'below threshold'}")

    print("\nsister vs Q (impostor — MUST stay below threshold):")
    worst = 0.0
    for sn, sset in sorted(s_sets.items()):
        for qn, qset in sorted(q_sets.items()):
            s = identity.set_score(sset, qset)
            worst = max(worst, s)
            print(f"  {sn} vs {qn}: {s:6.1f}  {'ok' if s < thr else 'FALSE ACCEPT'}")

    print("\n1:N: enroll Q (all 3 photos) + sister as identities, probe each sister photo:")
    q_identity = [t for ts in q_sets.values() for t in ts]
    for sn in sorted(s_sets):
        gallery = {"Q-thumb": q_identity}
        for other in s_sets:
            if other != sn:
                gallery["sister-thumb"] = s_sets[other]
        who, score = identity.identify(s_sets[sn], gallery)
        print(f"  probe {sn} -> {who} ({score:.1f})")

    print(f"\nworst impostor score: {worst:.1f} (threshold {thr})")


if __name__ == "__main__":
    main()
