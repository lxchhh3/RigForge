"""Dump an ASCII FBX's limb-bone view as JSON, for authoring canonical_to_name.

Usage:
    PYTHONPATH=. python training/dump_avatar_bones.py <ascii_fbx_path> [--out PATH]

Produces a JSON file with one record per LimbNode bone:
    {model_id, name, parent_name, translation_xyz, child_names,
     has_skin_cluster, cluster_weight_count}

This is the same record shape Phase B sends to the LLM, but here it's used
manually to bootstrap a curated avatar's canonical_to_name mapping in
data/curated_avatars.json. Run it once per new curated avatar.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rigforge.ascii_fbx.lexer import parse as parse_fbx
from rigforge.ascii_fbx.sections import extract


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ascii_fbx", type=Path, help="path to an ASCII FBX")
    p.add_argument("--out", type=Path, help="output JSON path (default: alongside input)")
    args = p.parse_args(argv)

    if not args.ascii_fbx.exists():
        print(f"error: {args.ascii_fbx} not found", file=sys.stderr)
        return 1

    raw = args.ascii_fbx.read_bytes()
    view = extract(parse_fbx(raw))
    limbs = view.limb_bones()
    limbs.sort(key=lambda b: b.model_id)

    records = [b.to_json_record() for b in limbs]
    out_path = args.out or args.ascii_fbx.with_suffix(".bones.json")
    out_path.write_text(json.dumps(records, indent=2))
    print(f"wrote {len(records)} bone records → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
