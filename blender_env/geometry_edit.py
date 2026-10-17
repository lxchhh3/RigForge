"""Headless Blender geometry backend for RigForge.

Runs INSIDE Blender's bundled Python (bpy available). Invoked by
`rigforge.geometry.blender_edit` as:

    blender --background --factory-startup --python blender_env/geometry_edit.py -- \
        --op <op> --in <src.fbx> [--out <dst.fbx>] --report <report.json> [op args...]

Ops:
  noop            import <in>, measure, re-export to <out> (no geometry change).
  inspect         import <in>, measure only (no export).
  translate_slice import <in>, select a normalized bbox slice of one mesh along
                  an axis, translate those verts along that axis by
                  delta_frac * axis_extent, re-export to <out>. Vertex-count-
                  preserving (the safety boundary: never adds/removes verts, so
                  skin-cluster vertex indexing stays valid). This is the
                  "lower-soles / shorten-cuff" family.
                  Args: --mesh <name> --axis x|y|z --from-pct F --to-pct T
                        --delta-frac D   (signed; negative = down/in)

Optional for any op that edits: --preview-dir <dir> renders before/after PNGs
(Workbench engine, ortho side camera framed on the target mesh). Non-fatal.

We DON'T trust Blender's process exit code (a script exception leaves it at 0),
so the wrapper keys off the JSON report file; we also sys.exit(1) on failure.

PINNED EXPORTER SETTINGS are the crux of orientation fidelity (Phase 0): a
Maya-exported clothing must round-trip with the SAME mesh-Model rotation
convention so it doesn't reintroduce the historical "rotated 90 degrees" bug.
"""
import json
import math
import os
import sys
import traceback

import bpy
import mathutils

# Sibling import: Blender puts this script's dir on sys.path, but be explicit
# so the import is robust regardless of how Blender was launched.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slice_select import select_slice_indices  # noqa: E402

_AXIS = {"x": 0, "y": 1, "z": 2}


def _argv():
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1:]
    return []


def _parse(argv):
    out = {}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            key = a[2:]
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                out[key] = argv[i + 1]
                i += 2
            else:
                out[key] = True
                i += 1
        else:
            i += 1
    return out


def _ensure_fbx_addon():
    import addon_utils
    try:
        addon_utils.enable("io_scene_fbx", default_set=False, persistent=False)
    except Exception:
        pass


def _reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _import_fbx(path):
    bpy.ops.import_scene.fbx(
        filepath=path,
        use_anim=False,
        ignore_leaf_bones=False,
        automatic_bone_orientation=False,
        primary_bone_axis="Y",
        secondary_bone_axis="X",
    )


def _export_fbx(path):
    bpy.ops.export_scene.fbx(
        filepath=path,
        use_selection=False,
        use_visible=False,
        use_active_collection=False,
        global_scale=1.0,
        apply_unit_scale=True,
        apply_scale_options="FBX_SCALE_NONE",
        use_space_transform=True,
        bake_space_transform=False,
        object_types={"ARMATURE", "MESH", "EMPTY"},
        use_mesh_modifiers=False,
        mesh_smooth_type="OFF",
        use_custom_props=True,
        add_leaf_bones=False,
        primary_bone_axis="Y",
        secondary_bone_axis="X",
        bake_anim=False,
        path_mode="AUTO",
        axis_forward="-Z",
        axis_up="Y",
    )


def _mesh_objects():
    return [o for o in bpy.data.objects if o.type == "MESH"]


def _measure():
    meshes = {}
    armatures = []
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE":
            armatures.append(obj.name)
        elif obj.type == "MESH":
            me = obj.data
            shape_keys = 0
            if me.shape_keys and me.shape_keys.key_blocks:
                shape_keys = max(0, len(me.shape_keys.key_blocks) - 1)
            weight_entries = sum(len(v.groups) for v in me.vertices)
            skinned = any(m.type == "ARMATURE" for m in obj.modifiers)
            corners = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
            zs = [c.z for c in corners] or [0.0]
            meshes[obj.name] = {
                "verts": len(me.vertices),
                "shape_keys": shape_keys,
                "vertex_groups": len(obj.vertex_groups),
                "weight_entries": weight_entries,
                "skinned": skinned,
                "wz_min": round(min(zs), 5),
                "wz_max": round(max(zs), 5),
            }
    return meshes, armatures


def _find_mesh(name):
    obj = bpy.data.objects.get(name)
    if obj is not None and obj.type == "MESH":
        return obj
    # Fall back to a case-insensitive / suffix-tolerant match (Blender may have
    # appended .001 on import-name collision).
    cand = [o for o in _mesh_objects()
            if o.name == name or o.name.split(".")[0] == name
            or o.name.lower() == name.lower()]
    return cand[0] if cand else None


def _world_axis_coords(obj, axis_idx):
    mw = obj.matrix_world
    return [(mw @ v.co)[axis_idx] for v in obj.data.vertices]


def _translate_slice(args):
    """Move a normalized bbox slice of one mesh's verts along an axis. Returns
    the edit-result dict for the report."""
    mesh_name = args.get("mesh")
    if not mesh_name:
        raise SystemExit("translate_slice: --mesh is required")
    axis = str(args.get("axis", "z")).lower()
    if axis not in _AXIS:
        raise SystemExit(f"translate_slice: --axis must be x|y|z, got {axis!r}")
    axis_idx = _AXIS[axis]
    from_pct = float(args.get("from-pct", 0.0))
    to_pct = float(args.get("to-pct", 0.15))
    delta_frac = float(args.get("delta-frac", -0.3))

    obj = _find_mesh(mesh_name)
    if obj is None:
        raise SystemExit(f"translate_slice: mesh {mesh_name!r} not found")

    coords = _world_axis_coords(obj, axis_idx)
    axis_min, axis_max = min(coords), max(coords)
    extent = axis_max - axis_min
    selected = select_slice_indices(coords, from_pct, to_pct)
    delta_world = delta_frac * extent

    # Convert the world-space axis translation into the mesh's LOCAL frame once
    # (rigid transform -> constant offset), so we move verts "down in world Z"
    # regardless of the mesh's baked -90 orientation.
    world_vec = mathutils.Vector((0.0, 0.0, 0.0))
    world_vec[axis_idx] = delta_world
    local_delta = obj.matrix_world.inverted().to_3x3() @ world_vec

    me = obj.data
    key_blocks = (me.shape_keys.key_blocks if me.shape_keys else None)
    for i in selected:
        me.vertices[i].co += local_delta
        if key_blocks:
            # Shift every shape-key target equally so morph deltas are preserved.
            for kb in key_blocks:
                kb.data[i].co += local_delta
    me.update()

    return {
        "mesh": obj.name,
        "axis": axis,
        "from_pct": from_pct,
        "to_pct": to_pct,
        "delta_frac": delta_frac,
        "axis_min": axis_min,
        "axis_max": axis_max,
        "extent": extent,
        "delta_world": delta_world,
        "moved_count": len(selected),
        "total_verts": len(coords),
    }


def _scale_slice(args):
    """Squash a normalized bbox slice along an axis toward the TOP of the slice
    (the pivot). Each selected vert's distance below the pivot is multiplied by
    `factor` (0<factor<1 shortens). Because the pivot plane (top of the slice,
    where it meets the rest of the mesh) does NOT move, there's no tear at that
    boundary — the band compresses smoothly. This is the "cut sole height"
    op. Vertex-count-preserving.

    Args: --mesh --axis x|y|z --from-pct F --to-pct T --factor S
    """
    mesh_name = args.get("mesh")
    if not mesh_name:
        raise SystemExit("scale_slice: --mesh is required")
    axis = str(args.get("axis", "z")).lower()
    if axis not in _AXIS:
        raise SystemExit(f"scale_slice: --axis must be x|y|z, got {axis!r}")
    axis_idx = _AXIS[axis]
    from_pct = float(args.get("from-pct", 0.0))
    to_pct = float(args.get("to-pct", 0.3))
    factor = float(args.get("factor", 0.5))

    obj = _find_mesh(mesh_name)
    if obj is None:
        raise SystemExit(f"scale_slice: mesh {mesh_name!r} not found")

    coords = _world_axis_coords(obj, axis_idx)
    axis_min, axis_max = min(coords), max(coords)
    extent = axis_max - axis_min
    selected = select_slice_indices(coords, from_pct, to_pct)
    # Pivot = top of the selected band, so the seam with the static mesh above
    # stays put (no tear); everything below it rises proportionally.
    pivot = max((coords[i] for i in selected), default=axis_max)

    me = obj.data
    inv3 = obj.matrix_world.inverted().to_3x3()
    key_blocks = (me.shape_keys.key_blocks if me.shape_keys else None)
    for i in selected:
        # Signed world-axis move: pull the vertex toward the pivot by (1-factor).
        d_world = (factor - 1.0) * (coords[i] - pivot)
        world_vec = mathutils.Vector((0.0, 0.0, 0.0))
        world_vec[axis_idx] = d_world
        local_delta = inv3 @ world_vec
        me.vertices[i].co += local_delta
        if key_blocks:
            for kb in key_blocks:
                kb.data[i].co += local_delta
    me.update()

    return {
        "mesh": obj.name,
        "axis": axis,
        "from_pct": from_pct,
        "to_pct": to_pct,
        "factor": factor,
        "axis_min": axis_min,
        "axis_max": axis_max,
        "extent": extent,
        "pivot": pivot,
        "height_removed": (1.0 - factor) * (pivot - axis_min),
        "moved_count": len(selected),
        "total_verts": len(coords),
    }


def _bisect_cut(args):
    """A real "cut": bisect the mesh by a world-axis plane at `cut_pct` of the
    bbox extent, delete the slab BELOW the plane, and cap the opening. Removes
    height with NO texture compression (geometry is removed, not squashed).

    Unlike translate/scale this CHANGES the vertex count, so it does not get the
    "rig untouched" guarantee — the integrity question is whether the new
    cut-edge verts carry skin weights. bisect_plane interpolates the deform
    layer onto verts it creates, and holes_fill caps with n-gons (no new verts),
    so in principle every surviving vert stays weighted. The report's
    `unweighted_verts_after` is the empirical check.

    Args: --mesh --axis x|y|z --cut-pct P   (remove everything below P*extent)
    """
    import bmesh

    mesh_name = args.get("mesh")
    if not mesh_name:
        raise SystemExit("bisect_cut: --mesh is required")
    axis = str(args.get("axis", "z")).lower()
    if axis not in _AXIS:
        raise SystemExit(f"bisect_cut: --axis must be x|y|z, got {axis!r}")
    axis_idx = _AXIS[axis]
    cut_pct = float(args.get("cut-pct", 0.15))

    obj = _find_mesh(mesh_name)
    if obj is None:
        raise SystemExit(f"bisect_cut: mesh {mesh_name!r} not found")

    me = obj.data
    coords = _world_axis_coords(obj, axis_idx)
    axis_min, axis_max = min(coords), max(coords)
    extent = axis_max - axis_min
    cut_world = axis_min + cut_pct * extent

    verts_before = len(me.vertices)
    weight_entries_before = sum(len(v.groups) for v in me.vertices)
    unweighted_before = sum(1 for v in me.vertices if len(v.groups) == 0)

    # The cut plane is world-space (axis up); express it in the mesh's local
    # frame (which carries the baked -90 orientation).
    inv = obj.matrix_world.inverted()
    plane_co_world = mathutils.Vector((0.0, 0.0, 0.0))
    plane_co_world[axis_idx] = cut_world
    plane_no_world = mathutils.Vector((0.0, 0.0, 0.0))
    plane_no_world[axis_idx] = 1.0
    plane_co = inv @ plane_co_world
    plane_no = (inv.to_3x3() @ plane_no_world).normalized()

    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.layers.deform.verify()  # ensure the weight layer rides along
    geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
    res = bmesh.ops.bisect_plane(
        bm, geom=geom, dist=1e-7,
        plane_co=plane_co, plane_no=plane_no,
        clear_inner=True,   # delete the slab below the plane (normal points up)
        clear_outer=False,
    )
    cut_edges = [e for e in res.get("geom_cut", [])
                 if isinstance(e, bmesh.types.BMEdge)]
    if cut_edges:
        # holes_fill caps each boundary loop with an n-gon (adds no new verts,
        # so no unweighted interior verts), and won't bridge separate loops the
        # way triangle_fill can.
        bmesh.ops.holes_fill(bm, edges=cut_edges, sides=0)
    bm.to_mesh(me)
    bm.free()
    me.update()

    verts_after = len(me.vertices)
    weight_entries_after = sum(len(v.groups) for v in me.vertices)
    unweighted_after = sum(1 for v in me.vertices if len(v.groups) == 0)

    return {
        "mesh": obj.name,
        "axis": axis,
        "cut_pct": cut_pct,
        "axis_min": axis_min,
        "axis_max": axis_max,
        "extent": extent,
        "cut_world": cut_world,
        "verts_before": verts_before,
        "verts_after": verts_after,
        "verts_removed": verts_before - verts_after,
        "cut_edges": len(cut_edges),
        "weight_entries_before": weight_entries_before,
        "weight_entries_after": weight_entries_after,
        "unweighted_verts_before": unweighted_before,
        "unweighted_verts_after": unweighted_after,
    }


_EDIT_OPS = {
    "translate_slice": _translate_slice,
    "scale_slice": _scale_slice,
    "bisect_cut": _bisect_cut,
}


# View direction (world-space, from object toward camera) for each named view.
# `front` looks along +Y (we see the X/Z face — toes); `side` looks along +X
# (we see the Y/Z face — the sole PROFILE, best for judging height).
_VIEWS = {
    "front": mathutils.Vector((0.0, -1.0, 0.0)),
    "side": mathutils.Vector((-1.0, 0.0, 0.0)),
}


def _apply_textures(tex_map):
    """Re-point each material's base color at a real image on disk (the FBX's
    own texture paths are usually dead). `tex_map` is {material_name: png_path}.
    Returns {material_name: applied_path}. Best-effort per material."""
    applied = {}
    for mat in bpy.data.materials:
        key = mat.name if mat.name in tex_map else mat.name.split(".")[0]
        path = tex_map.get(key)
        if not path or not os.path.exists(path):
            continue
        mat.use_nodes = True
        nt = mat.node_tree
        bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None:
            bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        img = bpy.data.images.load(path, check_existing=True)
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = img
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        nt.nodes.active = tex  # Workbench TEXTURE mode reads the active image
        applied[mat.name] = path
    return applied


def _render_preview(target_obj, out_png, view="front", textured=False):
    """Workbench ortho render framed on the target mesh from a named view.
    Camera up is world +Z (via to_track_quat) so the image is never sideways.
    `textured` shows the materials' base-color image instead of flat clay.
    Best-effort."""
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = 600
    scene.render.resolution_y = 800
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    shading = scene.display.shading
    shading.light = "STUDIO"
    shading.color_type = "TEXTURE" if textured else "SINGLE"

    corners = [target_obj.matrix_world @ mathutils.Vector(c)
               for c in target_obj.bound_box]
    center = sum(corners, mathutils.Vector()) / 8.0
    dx = max(c.x for c in corners) - min(c.x for c in corners)
    dy = max(c.y for c in corners) - min(c.y for c in corners)
    dz = max(c.z for c in corners) - min(c.z for c in corners)
    span = max(dx, dy, dz)

    offset = _VIEWS.get(view, _VIEWS["front"])
    cam_loc = center + offset * (span * 2.0 + 1.0)

    cam = bpy.data.objects.get("rf_preview_cam")
    if cam is None:
        cam_data = bpy.data.cameras.new("rf_preview_cam")
        cam_data.type = "ORTHO"
        cam = bpy.data.objects.new("rf_preview_cam", cam_data)
        scene.collection.objects.link(cam)
    cam.data.ortho_scale = span * 1.5 + 1e-4
    cam.location = cam_loc
    # Aim -Z at the object, keep +Y toward world up -> upright, world-Z-up image.
    cam.rotation_euler = (center - cam_loc).to_track_quat("-Z", "Y").to_euler()
    scene.camera = cam

    scene.render.filepath = out_png
    bpy.ops.render.render(write_still=True)


def main():
    args = _parse(_argv())
    op = args.get("op", "noop")
    src = args.get("in")
    report_path = args.get("report")
    preview_dir = args.get("preview-dir")
    if not src or not report_path:
        raise SystemExit("geometry_edit: --in and --report are required")

    _ensure_fbx_addon()
    _reset_scene()
    _import_fbx(src)

    # Optional: re-point materials at real textures on disk (FBX paths are dead)
    # and render textured instead of flat clay.
    textured = str(args.get("shading", "solid")).lower() == "texture"
    applied_textures = {}
    if args.get("textures"):
        with open(args["textures"], "r", encoding="utf-8") as f:
            tex_map = json.load(f)
        applied_textures = _apply_textures(tex_map)

    before_meshes, before_arms = _measure()
    report = {
        "op": op,
        "source": src,
        "blender_version": bpy.app.version_string,
        "meshes": before_meshes,
        "armatures": before_arms,
        "applied_textures": applied_textures,
    }

    if op in ("noop", "inspect"):
        if op == "noop":
            if not args.get("out"):
                raise SystemExit("geometry_edit: --out is required for op=noop")
            _export_fbx(args["out"])

    elif op in _EDIT_OPS:
        target = _find_mesh(args.get("mesh", ""))
        previews = {}

        def _shoot(stage):
            # Render both a front and a side profile so soles/pinches that hide
            # from one angle are still visible.
            if not (preview_dir and target is not None):
                return
            for view in ("front", "side"):
                key = f"{stage}_{view}"
                try:
                    png = os.path.join(preview_dir, f"{key}.png")
                    _render_preview(target, png, view=view, textured=textured)
                    previews[key] = png
                except Exception as exc:
                    previews[f"{key}_error"] = repr(exc)

        _shoot("before")
        edit = _EDIT_OPS[op](args)
        after_meshes, after_arms = _measure()
        report["before"] = {"meshes": before_meshes, "armatures": before_arms}
        report["after"] = {"meshes": after_meshes, "armatures": after_arms}
        report["edit"] = edit
        # Keep top-level meshes/armatures = the AFTER state for consistency.
        report["meshes"] = after_meshes
        report["armatures"] = after_arms

        _shoot("after")
        report["previews"] = previews

        if not args.get("out"):
            raise SystemExit(f"geometry_edit: --out is required for op={op}")
        _export_fbx(args["out"])

    else:
        raise SystemExit(f"geometry_edit: unknown op {op!r}")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    total_v = sum(m["verts"] for m in report["meshes"].values())
    moved = report.get("edit", {}).get("moved_count", 0)
    print(f"[geometry_edit] op={op} meshes={len(report['meshes'])} "
          f"verts={total_v} moved={moved} -> report {report_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
