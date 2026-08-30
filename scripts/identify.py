"""Identify a finger photo against all enrolled identities.

    ./.venv/bin/python scripts/identify.py photo.jpg
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fingerphoto import store


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: identify.py photo.jpg")
    result = store.identify_photo(Path(sys.argv[1]))
    if result.get("note"):
        sys.exit(result["note"])
    print(f"scores (threshold {result['threshold']}):")
    for name, score in result["scores"].items():
        print(f"  {name}: {score:.1f}")
    if result["match"]:
        print(f"\nIDENTIFIED: {result['match']}")
    else:
        print(f"\nNO MATCH (best candidate {result['best_candidate']}, below threshold)")


if __name__ == "__main__":
    main()
