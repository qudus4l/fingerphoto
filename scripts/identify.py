"""Identify a finger photo, or a video, against all enrolled identities.

    ./.venv/bin/python scripts/identify.py photo.jpg
    ./.venv/bin/python scripts/identify.py clip.mov

A video's frames are sampled and scored, and only the best few are extracted.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fingerphoto import store
from fingerphoto.errors import AcquisitionError


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: identify.py photo.jpg|clip.mov")
    try:
        result = store.identify_photo(Path(sys.argv[1]))
    except AcquisitionError as e:
        # a refused photo is a re-shoot, not a stack trace
        sys.exit(f"{e.code.value}: {e}")
    if result.get("note"):
        sys.exit(result["note"])
    used = result.get("frames")
    if used and used["candidates"] > 1:
        print(f"frames: {used['used']} of {used['candidates']} used"
              + (f", ridge {used['scores']}" if used["scores"] else ""))
    print(f"scores (threshold {result['threshold']}):")
    for name, score in result["scores"].items():
        print(f"  {name}: {score:.1f}")
    if result["match"]:
        print(f"\nIDENTIFIED: {result['match']}")
    else:
        print(f"\nNO MATCH (best candidate {result['best_candidate']}, below threshold)")


if __name__ == "__main__":
    main()
