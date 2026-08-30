"""Multi-subject evaluation on real photos.

Photos in data/samples/photo named <subject>_<id>.jpg (e.g. thumb_6172 =
subject "thumb" (Q), sister_6175 = subject "sister"). Within-subject pairs are
genuine, cross-subject pairs are impostors. Also runs leave-one-out 1:N
identification with the macro finger and an FVC print as extra gallery
distractors.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fingerphoto import afis, identity

PHOTO_DIR = ROOT / "data" / "samples" / "photo"
OUT = ROOT / "out"


def main() -> None:
    skip = ("fingertip", "hand_", "sis_hand_", "sis_finger_")
    photos = sorted(
        p for p in PHOTO_DIR.glob("*_*.jpg") if not p.stem.startswith(skip)
    )
    subjects: dict[str, list[str]] = defaultdict(list)
    sets: dict[str, list[bytes]] = {}
    print("building template sets:")
    for p in photos:
        subject = p.stem.rsplit("_", 1)[0]
        try:
            sets[p.stem] = identity.photo_template_set(cv2.imread(str(p)), out_dir=OUT / p.stem)
        except ValueError as e:
            print(f"  {p.stem}: REJECTED by quality gate ({e})")
            continue
        subjects[subject].append(p.stem)
        print(f"  {p.stem} [{subject}]: {len(sets[p.stem])} templates")

    macro = identity.photo_template_set(
        cv2.imread(str(PHOTO_DIR / "fingertip_macro.jpg")), crop=False, out_dir=OUT / "fingertip_macro"
    )
    fvc = [
        afis.extract_template(
            cv2.createCLAHE(4.0, (15, 15)).apply(
                cv2.imread(str(ROOT / "data" / "fvc" / "101_1.tif"), cv2.IMREAD_GRAYSCALE)
            ),
            dpi=500,
        )
    ]

    thr = afis.THRESHOLD
    genuine_ok = impostor_ok = True

    print(f"\ngenuine pairs (same subject), threshold {thr}:")
    for subject, names in sorted(subjects.items()):
        for a, b in combinations(sorted(names), 2):
            s = identity.set_score(sets[a], sets[b])
            ok = s >= thr
            genuine_ok &= ok
            print(f"  {a} vs {b}: {s:6.1f}  {'MATCH' if ok else 'below threshold'}")

    print("\nimpostor pairs (different subjects):")
    subject_list = sorted(subjects)
    for sa, sb in combinations(subject_list, 2):
        for a in sorted(subjects[sa]):
            for b in sorted(subjects[sb]):
                s = identity.set_score(sets[a], sets[b])
                ok = s < thr
                impostor_ok &= ok
                print(f"  {a} vs {b}: {s:6.1f}  {'ok' if ok else 'FALSE ACCEPT'}")

    print("\nleave-one-out 1:N identification (gallery: all subjects + distractors):")
    ident_ok = True
    for subject, names in sorted(subjects.items()):
        if len(names) < 2:
            continue
        for probe_name in sorted(names):
            gallery: dict[str, list[bytes]] = {"macro-finger": macro, "fvc-101": fvc}
            for s2, n2 in subjects.items():
                enrolled = [t for n in n2 if n != probe_name for t in sets[n]]
                if enrolled:
                    gallery[s2] = enrolled
            who, score = identity.identify(sets[probe_name], gallery)
            ok = who == subject
            ident_ok &= ok
            print(f"  probe {probe_name} -> {who} ({score:6.1f})  {'ok' if ok else 'WRONG'}")

    print(f"\nsummary: genuine {'PASS' if genuine_ok else 'FAIL'} | "
          f"impostor {'PASS' if impostor_ok else 'FAIL'} | "
          f"identification {'PASS' if ident_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
