"""Tests for the pure round-trip report-validation logic.

This module is Blender-free: it validates the JSON report shape that the
headless `blender_env/geometry_edit.py` emits, so we can unit-test the
preservation gate without launching Blender (which is smoked manually, the
same way the fbx_env subprocess is).
"""
from __future__ import annotations

from rigforge.geometry.report import (
    RoundtripCheck,
    report_totals,
    validate_roundtrip_report,
)


def _mesh(verts: int, shape_keys: int = 0, vertex_groups: int = 0,
          weight_entries: int = 0, skinned: bool = False) -> dict:
    return {
        "verts": verts,
        "shape_keys": shape_keys,
        "vertex_groups": vertex_groups,
        "weight_entries": weight_entries,
        "skinned": skinned,
    }


def _report(meshes: dict, armatures: list[str] | None = None, op: str = "noop") -> dict:
    return {
        "op": op,
        "source": "x.fbx",
        "blender_version": "4.0.2",
        "meshes": meshes,
        "armatures": list(armatures or []),
    }


# --- report_totals -------------------------------------------------------

def test_report_totals_sums_across_meshes():
    rep = _report(
        {
            "Shoes": _mesh(100, shape_keys=2, vertex_groups=10, weight_entries=300, skinned=True),
            "Body": _mesh(50, shape_keys=1, vertex_groups=5, weight_entries=120, skinned=True),
        },
        armatures=["Armature"],
    )
    t = report_totals(rep)
    assert t["mesh_count"] == 2
    assert t["armature_count"] == 1
    assert t["total_verts"] == 150
    assert t["total_shape_keys"] == 3
    assert t["total_weight_entries"] == 420


def test_report_totals_empty():
    t = report_totals(_report({}))
    assert t == {
        "mesh_count": 0,
        "armature_count": 0,
        "total_verts": 0,
        "total_shape_keys": 0,
        "total_weight_entries": 0,
    }


# --- validate_roundtrip_report: clean ------------------------------------

def test_identical_reports_are_clean():
    before = _report(
        {"Shoes": _mesh(100, 2, 10, 300, True)}, armatures=["Armature"]
    )
    after = _report(
        {"Shoes": _mesh(100, 2, 10, 300, True)}, armatures=["Armature"]
    )
    check = validate_roundtrip_report(before, after)
    assert isinstance(check, RoundtripCheck)
    assert check.ok
    assert check.problems == []


def test_blender_renamed_mesh_but_counts_preserved_is_clean_with_warning():
    """Blender may suffix names (Shoes -> Shoes.001). Aggregate preservation
    is what matters; the name drift is a warning, not a failure."""
    before = _report({"Shoes": _mesh(100, 2, 10, 300, True)}, armatures=["Armature"])
    after = _report({"Shoes.001": _mesh(100, 2, 10, 300, True)}, armatures=["Armature"])
    check = validate_roundtrip_report(before, after)
    assert check.ok
    assert check.problems == []
    assert any("name" in w.lower() for w in check.warnings)


# --- validate_roundtrip_report: failures ---------------------------------

def test_vertex_count_loss_is_a_problem():
    before = _report({"Shoes": _mesh(100)}, armatures=["Armature"])
    after = _report({"Shoes": _mesh(98)}, armatures=["Armature"])
    check = validate_roundtrip_report(before, after)
    assert not check.ok
    assert any("vert" in p.lower() for p in check.problems)


def test_weight_loss_is_a_problem():
    before = _report({"Shoes": _mesh(100, weight_entries=300, skinned=True)}, armatures=["Armature"])
    after = _report({"Shoes": _mesh(100, weight_entries=0, skinned=False)}, armatures=["Armature"])
    check = validate_roundtrip_report(before, after)
    assert not check.ok
    assert any("weight" in p.lower() for p in check.problems)


def test_shape_key_loss_is_a_problem():
    before = _report({"Body": _mesh(100, shape_keys=5)}, armatures=["Armature"])
    after = _report({"Body": _mesh(100, shape_keys=0)}, armatures=["Armature"])
    check = validate_roundtrip_report(before, after)
    assert not check.ok
    assert any("shape" in p.lower() for p in check.problems)


def test_dropped_mesh_is_a_problem():
    before = _report({"Shoes": _mesh(100), "Body": _mesh(50)}, armatures=["Armature"])
    after = _report({"Shoes": _mesh(100)}, armatures=["Armature"])
    check = validate_roundtrip_report(before, after)
    assert not check.ok
    assert any("mesh" in p.lower() for p in check.problems)


def test_lost_armature_is_a_problem():
    before = _report({"Shoes": _mesh(100, skinned=True)}, armatures=["Armature"])
    after = _report({"Shoes": _mesh(100, skinned=True)}, armatures=[])
    check = validate_roundtrip_report(before, after)
    assert not check.ok
    assert any("armature" in p.lower() for p in check.problems)


def test_per_mesh_vert_mismatch_on_matching_name_is_a_problem():
    """When a name survives the round-trip but its vert count changed, that's a
    real problem even if the grand total happens to match elsewhere."""
    before = _report({"A": _mesh(100), "B": _mesh(100)}, armatures=["Armature"])
    after = _report({"A": _mesh(120), "B": _mesh(80)}, armatures=["Armature"])
    check = validate_roundtrip_report(before, after)
    assert not check.ok
    assert any("vert" in p.lower() for p in check.problems)
