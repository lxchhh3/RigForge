"""
Structural comparison of two FBX files via Autodesk FBX SDK.

For each file, walks the scene and collects:
  - Per-class object counts (Mesh, LimbNode, Material, Deformer, BlendShape, ...)
  - Per-mesh stats: control point count, polygon count, layer count,
                    skin cluster count, blend-shape channel count
  - Global settings (axis, unit, time mode)
  - Total skin-weight sum (sentinel for skin-data integrity)

Then prints a side-by-side diff. Exit code 0 if structurally identical,
1 if any drift is detected, 2 on load failure.

Usage:
    python fbx_compare.py <a.fbx> <b.fbx>
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import fbx
import FbxCommon


def fail(msg: str, code: int = 2) -> None:
    sys.stderr.write(f"FATAL: {msg}\n")
    sys.exit(code)


def load_scene(path: Path):
    manager, scene = FbxCommon.InitializeSdkObjects()
    importer = fbx.FbxImporter.Create(manager, "")
    if not importer.Initialize(str(path), -1, manager.GetIOSettings()):
        fail(f"could not open {path}: {importer.GetStatus().GetErrorString()}")
    try:
        v = importer.GetFileVersion()
    except Exception:
        v = (None, None, None)
    if not importer.Import(scene):
        fail(f"import failed for {path}: {importer.GetStatus().GetErrorString()}")
    importer.Destroy()
    return manager, scene, v


def walk_nodes(root):
    """Yield every node under `root` (depth-first)."""
    stack = [root]
    while stack:
        n = stack.pop()
        yield n
        for i in range(n.GetChildCount()):
            stack.append(n.GetChild(i))


def collect(scene) -> dict:
    root = scene.GetRootNode()
    stats: dict = {
        "node_count": 0,
        "attr_kinds": Counter(),
        "mesh_details": {},          # name -> { ctrl_pts, polys, layers, clusters, blendshape_channels }
        "material_names": [],
        "total_clusters": 0,
        "total_skin_weights": 0,
        "total_blendshape_channels": 0,
        "total_blendshape_shapes": 0,
        "axis_up": None,
        "axis_front": None,
        "unit_scale": None,
        "time_mode": None,
    }

    # Global settings
    gs = scene.GetGlobalSettings()
    axis = gs.GetAxisSystem()
    stats["axis_up"] = str(axis.GetUpVector())
    stats["axis_front"] = str(axis.GetFrontVector())
    stats["unit_scale"] = gs.GetSystemUnit().GetScaleFactor()
    stats["time_mode"] = str(gs.GetTimeMode())

    # Materials (scene-wide)
    for i in range(scene.GetMaterialCount()):
        stats["material_names"].append(scene.GetMaterial(i).GetName())
    stats["material_names"].sort()

    # Walk nodes
    for node in walk_nodes(root):
        stats["node_count"] += 1
        attr = node.GetNodeAttribute()
        if attr is None:
            stats["attr_kinds"]["<none>"] += 1
            continue
        kind = attr.GetAttributeType()
        # Map enum int to a human label
        type_name = {
            fbx.FbxNodeAttribute.EType.eNull: "Null",
            fbx.FbxNodeAttribute.EType.eMesh: "Mesh",
            fbx.FbxNodeAttribute.EType.eSkeleton: "Skeleton",
            fbx.FbxNodeAttribute.EType.eCamera: "Camera",
            fbx.FbxNodeAttribute.EType.eLight: "Light",
            fbx.FbxNodeAttribute.EType.eMarker: "Marker",
        }.get(kind, f"EType={kind}")
        stats["attr_kinds"][type_name] += 1

        if type_name != "Mesh":
            continue

        mesh = attr  # FbxMesh
        ctrl_pts = mesh.GetControlPointsCount()
        polys = mesh.GetPolygonCount()
        layers = mesh.GetLayerCount()

        # Skin deformers + cluster count + weight sum
        clusters = 0
        weight_sum = 0.0
        for di in range(mesh.GetDeformerCount(fbx.FbxDeformer.EDeformerType.eSkin)):
            skin = mesh.GetDeformer(di, fbx.FbxDeformer.EDeformerType.eSkin)
            for ci in range(skin.GetClusterCount()):
                clusters += 1
                cl = skin.GetCluster(ci)
                weights = cl.GetControlPointWeights()
                # GetControlPointWeights returns a list-like; sum it.
                for w in weights:
                    weight_sum += float(w)
        stats["total_clusters"] += clusters
        stats["total_skin_weights"] += weight_sum

        # Blend-shape deformers + shape count
        bs_channels = 0
        bs_shapes = 0
        for di in range(mesh.GetDeformerCount(fbx.FbxDeformer.EDeformerType.eBlendShape)):
            bs = mesh.GetDeformer(di, fbx.FbxDeformer.EDeformerType.eBlendShape)
            for ci in range(bs.GetBlendShapeChannelCount()):
                bs_channels += 1
                ch = bs.GetBlendShapeChannel(ci)
                bs_shapes += ch.GetTargetShapeCount()
        stats["total_blendshape_channels"] += bs_channels
        stats["total_blendshape_shapes"] += bs_shapes

        stats["mesh_details"][node.GetName()] = {
            "ctrl_pts": ctrl_pts,
            "polys": polys,
            "layers": layers,
            "clusters": clusters,
            "weight_sum": round(weight_sum, 6),
            "bs_channels": bs_channels,
            "bs_shapes": bs_shapes,
        }

    return stats


def diff_report(a: dict, b: dict, label_a: str, label_b: str) -> int:
    """Print diff, return 0 if identical, 1 if not."""
    drifted = 0
    print(f"\n{'='*78}")
    print(f"COMPARE: {label_a}  vs  {label_b}")
    print(f"{'='*78}")

    scalar_keys = [
        "node_count",
        "total_clusters",
        "total_blendshape_channels",
        "total_blendshape_shapes",
        "axis_up", "axis_front", "unit_scale", "time_mode",
    ]
    print(f"\n{'KEY':<32}{label_a:>22}{label_b:>22}  STATUS")
    print("-" * 78)
    for k in scalar_keys:
        va, vb = a[k], b[k]
        status = "OK " if va == vb else "DIFF"
        if va != vb:
            drifted += 1
        print(f"{k:<32}{str(va):>22}{str(vb):>22}  {status}")

    # Total skin weights: float tolerance
    wa, wb = a["total_skin_weights"], b["total_skin_weights"]
    rel = abs(wa - wb) / max(abs(wa), 1e-12)
    status = "OK " if rel < 1e-9 else "DIFF"
    if rel >= 1e-9:
        drifted += 1
    print(f"{'total_skin_weights':<32}{wa:>22.6f}{wb:>22.6f}  {status} (rel={rel:.2e})")

    # Attribute kind counts
    print(f"\nAttribute kinds:")
    all_kinds = sorted(set(a["attr_kinds"]) | set(b["attr_kinds"]))
    for k in all_kinds:
        va, vb = a["attr_kinds"].get(k, 0), b["attr_kinds"].get(k, 0)
        status = "OK " if va == vb else "DIFF"
        if va != vb:
            drifted += 1
        print(f"  {k:<30}{va:>22}{vb:>22}  {status}")

    # Materials
    print(f"\nMaterials ({len(a['material_names'])} vs {len(b['material_names'])}):")
    if a["material_names"] != b["material_names"]:
        drifted += 1
        print(f"  DIFF  A: {a['material_names']}")
        print(f"        B: {b['material_names']}")
    else:
        print(f"  OK    {a['material_names']}")

    # Per-mesh details
    mesh_names = sorted(set(a["mesh_details"]) | set(b["mesh_details"]))
    print(f"\nPer-mesh ({len(mesh_names)} unique names):")
    print(f"  {'mesh':<30}{'field':<15}{label_a:>14}{label_b:>14}  STATUS")
    print("  " + "-" * 76)
    for mn in mesh_names:
        ma = a["mesh_details"].get(mn, {})
        mb = b["mesh_details"].get(mn, {})
        for field in ("ctrl_pts", "polys", "layers", "clusters",
                      "weight_sum", "bs_channels", "bs_shapes"):
            va, vb = ma.get(field, "<missing>"), mb.get(field, "<missing>")
            if isinstance(va, float) and isinstance(vb, float):
                ok = abs(va - vb) < 1e-6
            else:
                ok = va == vb
            if not ok:
                drifted += 1
                print(f"  {mn:<30}{field:<15}{str(va):>14}{str(vb):>14}  DIFF")

    print(f"\n{'='*78}")
    if drifted == 0:
        print(f"RESULT: STRUCTURALLY IDENTICAL (no drift across all checks)")
        return 0
    print(f"RESULT: {drifted} field(s) drifted")
    return 1


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 64
    pa, pb = Path(argv[1]).resolve(), Path(argv[2]).resolve()
    if not pa.is_file(): fail(f"missing {pa}")
    if not pb.is_file(): fail(f"missing {pb}")

    _, scene_a, va = load_scene(pa)
    stats_a = collect(scene_a)
    _, scene_b, vb = load_scene(pb)
    stats_b = collect(scene_b)

    print(f"A: {pa}  schema={va}")
    print(f"B: {pb}  schema={vb}")
    schema_match = va == vb
    print(f"   schema match: {'OK' if schema_match else 'DIFF'}")

    rc = diff_report(stats_a, stats_b, "A", "B")
    if not schema_match:
        rc = max(rc, 1)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
