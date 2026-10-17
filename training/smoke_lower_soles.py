"""Phase 1 smoke: the lower-soles op (translate_slice) on a real clothing.

Runs the vertex-count-preserving slice translate on the school-uniform Shoes
mesh, then proves the edit is geometrically sound AND structurally safe:

  1. translate_slice: lower the bottom band of Shoes along world Z
  2. preservation: vert count / skin weights / shape keys unchanged
     (vertex-count-preserving is the safety boundary) and moved_count > 0
  3. structure + orientation survive: convert IN/OUT through fbx_env, parse
     with SectionView, confirm weighted clusters + weight count preserved and
     the orientation monitor stays clean
  4. point at the before/after preview PNGs to eyeball the change

Usage:
    PYTHONPATH=. python training/smoke_lower_soles.py [CLOTHING.fbx] [MESH]
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
from rigforge.geometry.blender_edit import translate_slice
from rigforge.geometry.report import report_totals, validate_roundtrip_report

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLOTHING = Path(
    "D:/2files/models/vrc/ASCII_models/clothing/Maya_SchoolUniform/Maya_SchoolUniform.fbx"
)
DEFAULT_MESH = "Shoes"


def _weighted(view: SectionView) -> tuple[int, int]:
    return (
        sum(1 for c in view.clusters.values() if c.weight_count > 0),
        sum(c.weight_count for c in view.clusters.values()),
    )


def main() -> int:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CLOTHING
    mesh = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MESH
    if not src.is_file():
        print(f"[skip] clothing not found: {src}")
        return 2

    work = REPO_ROOT / "data" / "training" / "_lower_soles"
    work.mkdir(parents=True, exist_ok=True)
    out_fbx = work / (src.stem + "_lowsole.fbx")
    preview = work / "preview"

    print(f"[in ] {src}")
    print(f"[op ] translate_slice mesh={mesh} axis=z from=0.0 to=0.2 delta_frac=-0.4 "
          "(lower bottom 20% by 40% of shoe height)")
    t0 = time.time()
    report = translate_slice(
        src, out_fbx, mesh=mesh, axis="z",
        from_pct=0.0, to_pct=0.2, delta_frac=-0.4,
        preview_dir=preview,
    )
    print(f"[blender] done in {time.time()-t0:.1f}s (blender {report.get('blender_version')})")

    problems: list[str] = []

    # 2 — geometry preservation + the edit actually happened
    edit = report.get("edit", {})
    print("\n=== the edit ===")
    print(f"  mesh={edit.get('mesh')}  world-Z range=[{edit.get('axis_min'):.4f}, "
          f"{edit.get('axis_max'):.4f}]  extent={edit.get('extent'):.4f}")
    print(f"  moved {edit.get('moved_count')}/{edit.get('total_verts')} verts "
          f"by Δz={edit.get('delta_world'):.4f} (world units)")
    if not edit.get("moved_count", 0) > 0:
        problems.append("edit moved 0 vertices — slice selected nothing")

    check = validate_roundtrip_report(report.get("before", {}), report.get("after", {}))
    print("\n=== geometry preservation (vertex-count-preserving?) ===")
    print(f"  before: {report_totals(report.get('before', {}))}")
    print(f"  after : {report_totals(report.get('after', {}))}")
    for w in check.warnings:
        print(f"  WARN: {w}")
    for p in check.problems:
        print(f"  FAIL: {p}")
    problems.extend(check.problems)

    # 3 — structure + orientation survive (fbx_env -> SectionView)
    print("\n=== structure + orientation (fbx_env -> SectionView) ===")
    in_ascii = bin_to_ascii(src, work / (src.stem + "_IN_ascii.fbx"))
    out_ascii = bin_to_ascii(out_fbx, work / (src.stem + "_OUT_ascii.fbx"))
    vi = extract(parse_fbx(in_ascii.read_bytes()))
    vo = extract(parse_fbx(out_ascii.read_bytes()))
    wi, swi = _weighted(vi)
    wo, swo = _weighted(vo)
    print(f"  weighted_clusters  IN={wi}  OUT={wo}")
    print(f"  total_weight_count IN={swi}  OUT={swo}")
    if wi != wo:
        problems.append(f"weighted clusters changed: {wi} -> {wo}")
    if swi != swo:
        problems.append(f"total weight count changed: {swi} -> {swo}")
    print(f"  dominant mesh-Model rot  IN={dominant_rotation(vi)}  OUT={dominant_rotation(vo)}")
    conv = compare_conventions(vi, vo)
    if conv:
        problems.append(conv)
        print(f"  FAIL: {conv}")
    mixed = find_mixed_orientation(vo)
    if mixed:
        problems.append(mixed.message())
        print(f"  FAIL [mixed in OUT]: {mixed.message()}")
    if not conv and not mixed:
        print("  OK: structure + orientation preserved through the edit")

    # 4 — previews
    print("\n=== preview ===")
    for k, v in (report.get("previews") or {}).items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    if problems:
        print(f"PHASE 1: RED — {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("PHASE 1: GREEN — lower-soles edit applied, geometry/weights/shape "
          "keys preserved, structure + orientation intact.")
    print(f"Eyeball the soles: {preview/'before.png'}  vs  {preview/'after.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
