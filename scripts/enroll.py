"""Enroll an identity from finger photos.

    ./.venv/bin/python scripts/enroll.py NAME photo1.jpg [photo2.jpg ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fingerphoto import store


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit("usage: enroll.py NAME photo1.jpg [photo2.jpg ...]")
    name, photos = sys.argv[1], [Path(p) for p in sys.argv[2:]]
    result = store.enroll(name, photos)
    for photo, n in result["enrolled"]:
        print(f"enrolled {photo}: {n} templates")
    for photo, why in result["rejected"]:
        print(f"REJECTED {photo}: {why}")
    print("\ngallery now:")
    for row in store.list_identities():
        print(f"  {row['name']}: {row['photos']} photos, {row['templates']} templates")


if __name__ == "__main__":
    main()
