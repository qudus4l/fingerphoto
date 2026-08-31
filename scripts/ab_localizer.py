"""Localizer A/B: our color skin mask vs SAM+CLIP, SourceAFIS held constant.

This isolates the one question that matters: does learned finger localization
(the part of Ridgeformer's approach worth keeping) make OUR matcher better?
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import cv2

warnings.filterwarnings("ignore")
FP = Path("/Users/Q_1/Projects/fingerphoto")
sys.path.insert(0, str(FP))

from fingerphoto import afis, identity, imaging, localize_sam  # noqa: E402
from fingerphoto.scale import ridge_period_band_for_fingertip  # noqa: E402
from fingerphoto.segment import crop_distal  # noqa: E402

SAMPLES = FP / "data/samples/photo"
OUT = Path(__file__).resolve().parent / "sam_crops"
PHOTOS = [
    ("thumb_6172", "Q"), ("thumb_6173", "Q"), ("thumb_6174", "Q"), ("thumb_6217", "Q"),
    ("sister_6175", "SA"), ("sister_6176", "SA"), ("sister_6214", "SB"),
    ("fingertip_macro", "X"),
]


def templates_from_crop(crop, mask):
    return identity._templates_for_crop(
        crop, mask, None, identity.SCALE_SWEEP, None, identity.MIN_MINUTIAE
    )


def main() -> None:
    OUT.mkdir(exist_ok=True)
    color_t, sam_t = {}, {}

    print("=" * 78)
    print("EXTRACTION — color skin mask vs SAM+CLIP localization")
    print("=" * 78)
    for name, _ in PHOTOS:
        path = SAMPLES / f"{name}.jpg"
        if not path.exists():
            continue
        bgr = imaging.decode_image(path.read_bytes(), path.name)

        try:
            c, m = crop_distal(bgr)
            color_t[name] = templates_from_crop(c, m)
            a = f"{len(color_t[name])} tmpl"
        except Exception as e:
            a = f"FAIL {type(e).__name__}"

        t0 = time.monotonic()
        try:
            up, mask = localize_sam.find_finger(bgr)
            c2, m2 = crop_distal(up, mask=mask)
            cv2.imwrite(str(OUT / f"{name}.png"), c2)
            sam_t[name] = templates_from_crop(c2, m2)
            b = f"{len(sam_t[name])} tmpl ({time.monotonic() - t0:.0f}s)"
        except Exception as e:
            b = f"FAIL {type(e).__name__}: {str(e)[:28]}"
        print(f"  {name:18s} color: {a:16s} SAM+CLIP: {b}")

    finger = dict(PHOTOS)
    names = [n for n, _ in PHOTOS]
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            gen = finger[a] == finger[b]
            cs = (identity.set_score(color_t[a], color_t[b])
                  if color_t.get(a) and color_t.get(b) else None)
            ss = (identity.set_score(sam_t[a], sam_t[b])
                  if sam_t.get(a) and sam_t.get(b) else None)
            rows.append({"pair": f"{a} vs {b}", "genuine": gen,
                         "sideways": "6217" in a or "6217" in b,
                         "color": cs, "sam": ss})

    def show(title, subset):
        print()
        print("=" * 78)
        print(title)
        print("=" * 78)
        print(f"{'pair':42s} {'color':>10s} {'SAM+CLIP':>10s}")
        print("-" * 78)
        for r in sorted(subset, key=lambda r: -(r["color"] or 0)):
            c = f"{r['color']:10.1f}" if r["color"] is not None else "       n/a"
            s = f"{r['sam']:10.1f}" if r["sam"] is not None else "       n/a"
            flag = ""
            if r["color"] is not None and r["sam"] is not None:
                if r["genuine"] and r["sam"] > r["color"]:
                    flag = "  better"
                elif r["genuine"] and r["sam"] < r["color"]:
                    flag = "  worse"
            print(f"{r['pair']:42s} {c} {s}{flag}")

    show("GENUINE (want HIGH, threshold 40)", [r for r in rows if r["genuine"]])
    show("IMPOSTOR (want LOW)", [r for r in rows if not r["genuine"]])

    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    for key, label in (("color", "color skin mask"), ("sam", "SAM+CLIP")):
        for scope, keep in (("all      ", lambda r: True),
                            ("upright  ", lambda r: not r["sideways"])):
            g = [r[key] for r in rows if r["genuine"] and keep(r) and r[key] is not None]
            i_ = [r[key] for r in rows if not r["genuine"] and keep(r) and r[key] is not None]
            if g and i_:
                print(f"{label:16s} {scope} genuine min {min(g):7.1f} | impostor max {max(i_):6.1f}"
                      f" | margin {min(g) / max(max(i_), 0.1):5.1f}x"
                      f" | matches@40: {sum(1 for x in g if x >= 40)}/{len(g)}"
                      f" | false accepts: {sum(1 for x in i_ if x >= 40)}")
    print(f"\ncrops: {OUT}")


if __name__ == "__main__":
    main()
