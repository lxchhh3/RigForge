"""Pure validation of the Blender round-trip JSON report.

`blender_env/geometry_edit.py` emits a report describing what it measured in
the imported scene:

    {
      "op": "noop" | "inspect",
      "source": "<path>",
      "blender_version": "4.0.2",
      "meshes": {
        "<mesh_object_name>": {
          "verts": int,            # len(mesh.vertices)
          "shape_keys": int,       # key_blocks minus the Basis
          "vertex_groups": int,    # len(object.vertex_groups)
          "weight_entries": int,   # sum(len(v.groups) for v in mesh.vertices)
          "skinned": bool          # has an Armature modifier
        }, ...
      },
      "armatures": ["<armature_object_name>", ...]
    }

The preservation gate (Phase 0) compares the report of the original clothing
against the report of the Blender-round-tripped clothing. This module holds
that comparison as pure logic so it is unit-tested without launching Blender.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RoundtripCheck:
    """Outcome of comparing a before/after round-trip report.

    `ok` is True only when there are no `problems`. `warnings` are non-fatal
    observations (e.g. Blender renamed a mesh but kept its geometry intact).
    `before`/`after` carry the computed totals for logging/manifests.
    """
    ok: bool
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    before: dict = field(default_factory=dict)
    after: dict = field(default_factory=dict)


def report_totals(report: dict) -> dict:
    """Name-independent aggregate counts — the robust preservation signal,
    immune to Blender's mesh-name suffixing on re-import."""
    meshes = report.get("meshes", {}) or {}
    return {
        "mesh_count": len(meshes),
        "armature_count": len(report.get("armatures", []) or []),
        "total_verts": sum(int(m.get("verts", 0)) for m in meshes.values()),
        "total_shape_keys": sum(int(m.get("shape_keys", 0)) for m in meshes.values()),
        "total_weight_entries": sum(
            int(m.get("weight_entries", 0)) for m in meshes.values()
        ),
    }


def validate_roundtrip_report(before: dict, after: dict) -> RoundtripCheck:
    """Compare two reports and decide whether the round-trip preserved the
    clothing's geometry, skin weights, and shape keys.

    Problems (fatal): any aggregate count changed, or a mesh that kept its
    name changed vertex count. Warnings (non-fatal): the mesh-name *set*
    drifted (Blender re-import commonly appends `.001`), which is fine as long
    as the aggregates hold.
    """
    tb = report_totals(before)
    ta = report_totals(after)
    problems: list[str] = []
    warnings: list[str] = []

    def _cmp(key: str, label: str) -> None:
        if tb[key] != ta[key]:
            problems.append(f"{label} changed: {tb[key]} -> {ta[key]}")

    _cmp("mesh_count", "mesh count")
    _cmp("armature_count", "armature count")
    _cmp("total_verts", "total vertex count")
    _cmp("total_shape_keys", "total shape-key count")
    _cmp("total_weight_entries", "total skin-weight entries")

    before_meshes = before.get("meshes", {}) or {}
    after_meshes = after.get("meshes", {}) or {}
    before_names = set(before_meshes)
    after_names = set(after_meshes)

    only_before = sorted(before_names - after_names)
    only_after = sorted(after_names - before_names)
    if only_before or only_after:
        warnings.append(
            f"mesh name set drifted (Blender rename?): "
            f"removed={only_before} added={only_after}"
        )

    for name in sorted(before_names & after_names):
        vb = int(before_meshes[name].get("verts", 0))
        va = int(after_meshes[name].get("verts", 0))
        if vb != va:
            problems.append(f"mesh '{name}' vertex count changed: {vb} -> {va}")

    return RoundtripCheck(
        ok=not problems,
        problems=problems,
        warnings=warnings,
        before=tb,
        after=ta,
    )
