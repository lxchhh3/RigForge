"""Phase 0 gate: does a NO-OP Blender round-trip preserve a clothing FBX well
enough to keep flowing through RigForge's assemble pipeline?

This is the kill-point for the whole headless-Blender taste-edit approach. It
does NOT edit geometry. It:

  1. drives Blender (headless) to import a real binary clothing and re-export
     it with the pinned exporter settings  -> report_in (+ the OUT fbx)
  2. re-imports OUT in Blender to measure it -> report_out
  3. checks geometry preservation (verts / skin weights / shape keys / mesh +
     armature counts) via rigforge.geometry.report
  4. converts IN and OUT to ASCII through fbx_env and parses both with
     SectionView for structural counts (meshes / LimbNodes / clusters /
     blendshape channels)
  5. runs the orientation monitor (compare_conventions + find_mixed) to catch
     the historical "rotated 90 degrees" axis-baking failure

GREEN  -> the Blender pre-step is viable; proceed to Phase 1 (the real op).
RED    -> tune the pinned exporter settings in blender_env/geometry_edit.py, or
          if orientation can't be made to match, reconsider (FBX SDK geometry
          API instead of Blender).

Usage:
    PYTHONPATH=. python training/smoke_blender_roundtrip.py [CLOTHING.fbx]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from rigforge.ascii_fbx.convert import bin_to_ascii
from rigforge.ascii_fbx.lexer import parse as parse_fbx
from rigforge.ascii_fbx.sections import SectionView, extract
from rigforge.diagnostics.orientation import (
    compare_conventions,
    dominant_rotation,
    find_mixed_orientation,
)
from rigforge.geometry.blender_edit import inspect, roundtrip_noop
from rigforge.geometry.report import report_totals, validate_roundtrip_report

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLOTHING = Path(
    "D:/2files/models/vrc/ASCII_models/clothing/Maya_SchoolUniform/Maya_SchoolUniform.fbx"
)


def _load_view(ascii_path: Path) -> SectionView:
    return extract(parse_fbx(ascii_path.read_bytes()))


def _struct_counts(view: SectionView) -> dict:
    meshes = sum(1 for b in view.bones.values() if b.type_class == "Mesh")
    limbs = sum(1 for b in view.bones.values() if b.type_class == "LimbNode")
    # Compare WEIGHTED clusters + total weight entries — the real skinning. The
    # total cluster count is NOT a preservation signal: Blender re-exports the
    # full bone x mesh grid, padding with zero-weight clusters (the same class
    # Maya's convention produces and the pipeline already filters). Tracked as
    # informational bloat, not a gate failure.
    return {
        "meshes": meshes,
        "limb_bones": limbs,
        "weighted_clusters": sum(1 for c in view.clusters.values() if c.weight_count > 0),
        "total_weight_count": sum(c.weight_count for c in view.clusters.values()),
        "blend_shape_channels": len(view.blend_shape_channels),
    }


def main() -> int:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CLOTHING
    if not src.is_file():
        print(f"[skip] clothing not found: {src}")
        return 2

    work = REPO_ROOT / "data" / "training" / "_blender_roundtrip"
    work.mkdir(parents=True, exist_ok=True)
    out_fbx = work / (src.stem + "_blender_roundtrip.fbx")

    print(f"[in ] {src}  ({src.stat().st_size/1e6:.1f} MB)")
    print(f"[out] {out_fbx}")

    # 1 + 2 — Blender no-op round-trip, then re-measure the output.
    t0 = time.time()
    print("[blender] import -> re-export (no-op) ...")
    report_in = roundtrip_noop(src, out_fbx)
    print(f"[blender] re-import OUT to re-measure ...")
    report_out = inspect(out_fbx, report_path=work / "out_inspect.json")
    print(f"[blender] done in {time.time()-t0:.1f}s "
          f"(blender {report_in.get('blender_version')})")

    # 3 — geometry preservation
    check = validate_roundtrip_report(report_in, report_out)
    print("\n=== geometry preservation (Blender measure) ===")
    print(f"  before: {report_totals(report_in)}")
    print(f"  after : {report_totals(report_out)}")
    for w in check.warnings:
        print(f"  WARN: {w}")
    for p in check.problems:
        print(f"  FAIL: {p}")

    # 4 — structural counts through fbx_env + SectionView
    print("\n=== structural parse (fbx_env -> SectionView) ===")
    in_ascii = bin_to_ascii(src, work / (src.stem + "_IN_ascii.fbx"))
    out_ascii = bin_to_ascii(out_fbx, work / (src.stem + "_OUT_ascii.fbx"))
    view_in = _load_view(in_ascii)
    view_out = _load_view(out_ascii)
    ci, co = _struct_counts(view_in), _struct_counts(view_out)
    struct_problems = []
    for k in ci:
        flag = "" if ci[k] == co[k] else "  <-- DRIFT"
        if ci[k] != co[k]:
            struct_problems.append(f"{k}: {ci[k]} -> {co[k]}")
        print(f"  {k:22} IN={ci[k]:6}  OUT={co[k]:6}{flag}")
    # Informational: total (incl. zero-weight) clusters — expected to grow.
    tot_in, tot_out = len(view_in.clusters), len(view_out.clusters)
    print(f"  {'total_clusters(info)':22} IN={tot_in:6}  OUT={tot_out:6}"
          f"  (+{tot_out - tot_in} zero-weight pad — benign, pipeline filters)")

    # 5 — orientation monitor (the axis-baking failure)
    print("\n=== orientation monitor ===")
    print(f"  dominant mesh-Model rot  IN={dominant_rotation(view_in)}  "
          f"OUT={dominant_rotation(view_out)}")
    orient_problems = []
    conv = compare_conventions(view_in, view_out)
    if conv:
        orient_problems.append(conv)
        print(f"  FAIL: {conv}")
    mixed = find_mixed_orientation(view_out)
    if mixed:
        orient_problems.append(mixed.message())
        print(f"  FAIL [mixed in OUT]: {mixed.message()}")
    if not orient_problems:
        print("  OK: clothing round-trips with a consistent rotation convention")

    # verdict
    print("\n" + "=" * 60)
    all_problems = check.problems + struct_problems + orient_problems
    if all_problems:
        print(f"GATE: RED — {len(all_problems)} problem(s):")
        for p in all_problems:
            print(f"  - {p}")
        print("Tune blender_env/geometry_edit.py pinned settings (esp. export "
              "axis/bake) and re-run, or reconsider the Blender approach.")
        return 1
    print("GATE: GREEN — no-op Blender round-trip preserves geometry, skin "
          "weights, shape keys, structure, and orientation.")
    print("Proceed to Phase 1 (the lower-soles op).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
