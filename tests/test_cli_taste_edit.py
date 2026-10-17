"""Tests for the `rigforge taste-edit` CLI subcommand.

Blender-free: the geometry backend (`rigforge.geometry.blender_edit`) is
monkeypatched to return canned reports, so these exercise the CLI's op
routing, validation, and exit codes without launching Blender.
"""
from __future__ import annotations

import rigforge.geometry.blender_edit as be
from rigforge.cli import main


def _bisect_report(unweighted_before=0, unweighted_after=0):
    return {
        "op": "bisect_cut",
        "meshes": {"Heel": {"verts": 10488}},
        "edit": {
            "mesh": "Heel",
            "verts_before": 11008,
            "verts_after": 10488,
            "verts_removed": 520,
            "unweighted_verts_before": unweighted_before,
            "unweighted_verts_after": unweighted_after,
        },
        "previews": {},
    }


def _slice_report(before_verts, after_verts, moved=100):
    return {
        "before": {"meshes": {"Shoes": {"verts": before_verts}}, "armatures": ["Armature"]},
        "after": {"meshes": {"Shoes": {"verts": after_verts}}, "armatures": ["Armature"]},
        "edit": {"mesh": "Shoes", "moved_count": moved},
        "previews": {},
    }


def _inspect_report():
    return {
        "meshes": {
            "Main": {"verts": 5000, "wz_min": 0.5, "wz_max": 1.0, "skinned": True},
            "Heel": {"verts": 11008, "wz_min": -0.03, "wz_max": 0.17, "skinned": True},
        },
        "armatures": ["Armature"],
    }


# --- --list-meshes -------------------------------------------------------

def test_list_meshes(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(be, "inspect", lambda src, **kw: _inspect_report())
    src = tmp_path / "c.fbx"; src.write_bytes(b"x")
    rc = main(["taste-edit", "--clothing", str(src), "--list-meshes"])
    assert rc == 0
    out = capsys.readouterr().out
    # Bottom-most mesh (the sole) should be listed first.
    assert out.index("Heel") < out.index("Main")


# --- bisect_cut ----------------------------------------------------------

def test_bisect_cut_clean_exits_zero(monkeypatch, capsys, tmp_path):
    calls = {}

    def fake_cut(src, out, **kw):
        calls.update(kw)
        calls["out"] = out
        return _bisect_report()

    monkeypatch.setattr(be, "bisect_cut", fake_cut)
    src = tmp_path / "c.fbx"; src.write_bytes(b"x")
    rc = main([
        "taste-edit", "--clothing", str(src), "--mesh", "Heel",
        "--op", "bisect_cut", "--cut-pct", "0.15", "--out", str(tmp_path / "o.fbx"),
    ])
    assert rc == 0
    assert calls["mesh"] == "Heel"
    assert calls["cut_pct"] == 0.15


def test_bisect_cut_unweighted_verts_fails(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(be, "bisect_cut",
                        lambda src, out, **kw: _bisect_report(unweighted_after=7))
    src = tmp_path / "c.fbx"; src.write_bytes(b"x")
    rc = main([
        "taste-edit", "--clothing", str(src), "--mesh", "Heel",
        "--op", "bisect_cut", "--out", str(tmp_path / "o.fbx"),
    ])
    assert rc == 1
    err = capsys.readouterr().err
    assert "unweighted" in err.lower()


# --- scale_slice (vertex-count-preserving) -------------------------------

def test_scale_slice_preserved_exits_zero(monkeypatch, tmp_path):
    captured = {}

    def fake_scale(src, out, **kw):
        captured.update(kw)
        return _slice_report(1404, 1404)

    monkeypatch.setattr(be, "scale_slice", fake_scale)
    src = tmp_path / "c.fbx"; src.write_bytes(b"x")
    rc = main([
        "taste-edit", "--clothing", str(src), "--mesh", "Shoes",
        "--op", "scale_slice", "--from-pct", "0.0", "--to-pct", "0.4",
        "--factor", "0.5", "--out", str(tmp_path / "o.fbx"),
    ])
    assert rc == 0
    assert captured["factor"] == 0.5
    assert captured["to_pct"] == 0.4


def test_scale_slice_vertex_loss_fails(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(be, "scale_slice",
                        lambda src, out, **kw: _slice_report(1404, 1390))
    src = tmp_path / "c.fbx"; src.write_bytes(b"x")
    rc = main([
        "taste-edit", "--clothing", str(src), "--mesh", "Shoes",
        "--op", "scale_slice", "--out", str(tmp_path / "o.fbx"),
    ])
    assert rc == 1
    assert "vert" in capsys.readouterr().err.lower()


# --- routing -------------------------------------------------------------

def test_translate_slice_routes(monkeypatch, tmp_path):
    captured = {}

    def fake_translate(src, out, **kw):
        captured.update(kw)
        return _slice_report(1404, 1404)

    monkeypatch.setattr(be, "translate_slice", fake_translate)
    src = tmp_path / "c.fbx"; src.write_bytes(b"x")
    rc = main([
        "taste-edit", "--clothing", str(src), "--mesh", "Shoes",
        "--op", "translate_slice", "--delta-frac", "-0.3", "--out", str(tmp_path / "o.fbx"),
    ])
    assert rc == 0
    assert captured["delta_frac"] == -0.3


def test_edit_requires_mesh_and_out(tmp_path, capsys):
    src = tmp_path / "c.fbx"; src.write_bytes(b"x")
    # No --mesh / --out and not --list-meshes -> usage error (SystemExit).
    try:
        rc = main(["taste-edit", "--clothing", str(src), "--op", "bisect_cut"])
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else 1
    assert rc != 0
