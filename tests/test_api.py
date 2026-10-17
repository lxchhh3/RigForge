"""Tests for the FastAPI bridge that exposes the pipeline to the frontend.

Slice 1: read-only manifest browser (list + detail).
Slice 2 (compose flow): avatars list + clothings/inspect (returns the bone
tree from a clothing ASCII so the FE can render the part-toggle UI).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rigforge.api.server import create_app
from rigforge.avatars.registry import AvatarRegistry


def _write_manifest(dir_: Path, stem: str, payload: dict) -> Path:
    p = dir_ / f"{stem}.manifest.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _sample_manifest(donor: str = "maya", target: str = "maya",
                     kept: int = 10, key: str = "abc123") -> dict:
    return {
        "schema_version": 1,
        "generated_at": "2026-05-21T06:20:59+00:00",
        "input": {
            "donor_id": donor,
            "target_id": target,
            "donor_score": 1.0,
            "candidates": [{"id": donor, "score": 1.0}],
        },
        "output_fbx": f"/tmp/out_{donor}_{target}.fbx",
        "cache": {"key": key, "hit": False},
        "edit_plan": {"drops": [], "renames": {}, "n_drops": 0, "n_renames": 0},
        "warnings": [],
        "decisions_summary": {"kept": kept, "dropped": 0, "llm_model_id": None},
        "notes": [],
    }


@pytest.fixture
def stress_dir(tmp_path: Path) -> Path:
    """Fixture data directory mimicking data/training/_stress."""
    d = tmp_path / "_stress"
    d.mkdir()
    return d


@pytest.fixture
def client(stress_dir: Path) -> TestClient:
    app = create_app(data_dir=stress_dir, registry=AvatarRegistry.load_default())
    return TestClient(app)


def _fake_cut_report(mesh="Heel", unweighted_after=0):
    return {
        "op": "bisect_cut",
        "edit": {
            "mesh": mesh,
            "verts_before": 11008,
            "verts_after": 10488,
            "verts_removed": 520,
            "unweighted_verts_before": 0,
            "unweighted_verts_after": unweighted_after,
        },
        "previews": {},
    }


def test_taste_edit_clean_cut(client: TestClient, tmp_path: Path, monkeypatch):
    """Endpoint runs bisect_cut (mocked — no Blender) and returns the edited
    path + integrity flag + preview slots."""
    import rigforge.geometry.blender_edit as be
    clothing = tmp_path / "c.fbx"; clothing.write_bytes(b"x")
    monkeypatch.setattr(be, "bisect_cut", lambda src, out, **kw: _fake_cut_report(kw["mesh"]))

    r = client.post("/api/clothings/taste-edit",
                    json={"clothing_path": str(clothing), "mesh": "Heel", "cut_pct": 0.15})
    assert r.status_code == 200
    body = r.json()
    assert body["mesh"] == "Heel"
    assert body["verts_removed"] == 520
    assert body["weights_ok"] is True
    assert "preview_before" in body and "preview_after" in body  # None w/o render
    assert body["edited_path"].endswith("__Heel_cut.fbx")


def test_taste_edit_unweighted_fails_integrity(client: TestClient, tmp_path: Path, monkeypatch):
    import rigforge.geometry.blender_edit as be
    clothing = tmp_path / "c.fbx"; clothing.write_bytes(b"x")
    monkeypatch.setattr(be, "bisect_cut",
                        lambda src, out, **kw: _fake_cut_report(unweighted_after=4))
    r = client.post("/api/clothings/taste-edit",
                    json={"clothing_path": str(clothing), "mesh": "Heel"})
    assert r.status_code == 200
    assert r.json()["weights_ok"] is False


def test_taste_edit_missing_file_404(client: TestClient):
    r = client.post("/api/clothings/taste-edit",
                    json={"clothing_path": "/no/such/file.fbx", "mesh": "Heel"})
    assert r.status_code == 404


def test_taste_edit_unsupported_op_400(client: TestClient, tmp_path: Path):
    clothing = tmp_path / "c.fbx"; clothing.write_bytes(b"x")
    r = client.post("/api/clothings/taste-edit",
                    json={"clothing_path": str(clothing), "mesh": "Heel", "op": "scale_slice"})
    assert r.status_code == 400


def test_list_assemblies_empty(client: TestClient):
    r = client.get("/api/assemblies")
    assert r.status_code == 200
    assert r.json() == []


def test_list_assemblies_returns_manifest_payloads(
    client: TestClient, stress_dir: Path,
):
    _write_manifest(stress_dir, "classic_chic", _sample_manifest(key="k1"))
    _write_manifest(stress_dir, "azure_virtue", _sample_manifest(key="k2"))

    r = client.get("/api/assemblies")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 2
    keys = {item["cache"]["key"] for item in body}
    assert keys == {"k1", "k2"}
    # Every list item carries an `id` derived from the manifest filename stem.
    ids = {item["id"] for item in body}
    assert ids == {"classic_chic", "azure_virtue"}


def test_list_skips_non_manifest_json(client: TestClient, stress_dir: Path):
    _write_manifest(stress_dir, "good", _sample_manifest(key="ok"))
    (stress_dir / "_summary.json").write_text('{"unrelated": true}', encoding="utf-8")

    r = client.get("/api/assemblies")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    assert body[0]["id"] == "good"


def test_get_assembly_by_id(client: TestClient, stress_dir: Path):
    _write_manifest(stress_dir, "classic_chic", _sample_manifest(kept=42, key="kk"))

    r = client.get("/api/assemblies/classic_chic")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "classic_chic"
    assert body["decisions_summary"]["kept"] == 42
    assert body["cache"]["key"] == "kk"


def test_get_assembly_missing_returns_404(client: TestClient):
    r = client.get("/api/assemblies/does-not-exist")
    assert r.status_code == 404


def test_get_assembly_rejects_path_traversal(client: TestClient):
    """The id is just a stem — `..` segments must not escape the data dir."""
    r = client.get("/api/assemblies/..%2Fsecret")
    assert r.status_code in (400, 404)


def test_cors_allows_frontend_origin(client: TestClient):
    """Dev server runs the FE on a different port; preflight must pass."""
    r = client.options(
        "/api/assemblies",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert r.status_code in (200, 204)
    assert "access-control-allow-origin" in {k.lower() for k in r.headers}


# --- /api/avatars ----------------------------------------------------------


def test_list_avatars_returns_curated_registry(client: TestClient):
    r = client.get("/api/avatars")
    assert r.status_code == 200
    body = r.json()
    ids = {a["id"] for a in body}
    assert "maya" in ids
    assert "moe" in ids
    maya = next(a for a in body if a["id"] == "maya")
    assert maya["display_name"]  # non-empty
    # canonical_roles is the list of canonical roles the avatar can receive
    # (i.e. the keys of canonical_to_name), so the FE knows which roles the
    # target supports before assemble runs.
    assert "Hips" in maya["canonical_roles"]
    assert "Hand.L" in maya["canonical_roles"]


# --- GET /api/avatars/{id}/inspect -----------------------------------------


def test_avatar_inspect_returns_same_shape_as_clothing_inspect(client: TestClient):
    """A curated avatar must be inspectable too — the modder needs to see the
    target's bone tree so they can strip its bundled clothing bones before
    splicing new clothing in. Shape must match /api/clothings/inspect so the
    FE can reuse BoneTree.vue + the same TS interface."""
    r = client.get("/api/avatars/maya/inspect")
    assert r.status_code == 200, r.text
    body = r.json()
    # Same top-level shape as clothing inspect
    assert body["donor_id"] == "maya"
    assert body["donor_score"] == 1.0
    assert body["total_bones"] > 0
    sample = body["bones"][0]
    for key in ("model_id", "name", "type_class", "parent_id",
                "subtree_size", "cluster_weight_count", "deforms_meshes"):
        assert key in sample, f"bone record missing {key!r}"
    # Maya's canonical Hips must appear among bones
    assert any(b["name"] == "Hips" for b in body["bones"])


def test_avatar_inspect_unknown_id_returns_404(client: TestClient):
    r = client.get("/api/avatars/no-such-avatar/inspect")
    assert r.status_code == 404


# --- /api/clothings/inspect -------------------------------------------------


def test_inspect_returns_bone_tree(client: TestClient, maya_fbx_ascii: Path):
    """Inspecting an ASCII FBX returns the bone records the FE needs to render
    a part-toggle tree."""
    r = client.post("/api/clothings/inspect", json={"path": str(maya_fbx_ascii)})
    assert r.status_code == 200
    body = r.json()
    assert "donor_id" in body
    assert "donor_score" in body
    assert body["donor_id"] == "maya"
    assert body["total_bones"] > 0
    assert isinstance(body["bones"], list)
    # Each bone carries enough for tree-build + part-toggle semantics
    sample = body["bones"][0]
    for key in ("model_id", "name", "parent_id", "subtree_size",
                "cluster_weight_count"):
        assert key in sample, f"bone record missing {key!r}"
    # Hips appears among the bones — it's the canonical root.
    assert any(b["name"] == "Hips" for b in body["bones"])
    assert body["rigid"] is False


def test_inspect_flags_a_garment_with_no_bones_as_rigid(client: TestClient, rigid_garment_fbx: Path):
    """The FE offers 'attach to bone' only for a rigid garment (no bones,
    e.g. GG_headset); the BE decides which those are."""
    r = client.post("/api/clothings/inspect", json={"path": str(rigid_garment_fbx)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rigid"] is True
    assert [b["name"] for b in body["bones"] if b["type_class"] == "Mesh"] == ["Visor"]


def test_inspect_gives_a_rigid_mesh_its_placement_in_the_target_armature_frame(
    client: TestClient, rigid_garment_fbx: Path,
):
    """A rigid garment's move reads in the target's Armature frame (kevin's
    GG.unity: the headset under Maya's Armature), so the Move panel must open
    on the file's placement in THAT frame. The visor's file rotation (-90 X)
    is the Armature's own, so there it reads ~0."""
    r = client.post(
        "/api/clothings/inspect",
        json={"path": str(rigid_garment_fbx), "target_id": "maya"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rigid_frame"] == "Armature"
    (visor,) = [b for b in body["bones"] if b["name"] == "Visor"]
    lt = visor["local_transform_unity"]
    assert lt["position"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
    assert lt["rotation"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-4)
    assert lt["scale"] == pytest.approx([1.0, 1.0, 1.0], abs=1e-9)


def test_inspect_without_a_target_keeps_a_rigid_mesh_file_values(
    client: TestClient, rigid_garment_fbx: Path,
):
    r = client.post("/api/clothings/inspect", json={"path": str(rigid_garment_fbx)})
    body = r.json()
    assert body["rigid_frame"] is None
    (visor,) = [b for b in body["bones"] if b["name"] == "Visor"]
    assert visor["local_transform_unity"]["rotation"] == pytest.approx([-90.0, 0.0, 0.0], abs=1e-6)


def test_inspect_with_an_unknown_target_is_400(client: TestClient, rigid_garment_fbx: Path):
    r = client.post(
        "/api/clothings/inspect",
        json={"path": str(rigid_garment_fbx), "target_id": "no-such-avatar"},
    )
    assert r.status_code == 400


def test_inspect_rejects_missing_path(client: TestClient, tmp_path: Path):
    bogus = tmp_path / "does_not_exist.fbx"
    r = client.post("/api/clothings/inspect", json={"path": str(bogus)})
    assert r.status_code == 404


def test_inspect_rejects_empty_body(client: TestClient):
    r = client.post("/api/clothings/inspect", json={})
    assert r.status_code == 422  # pydantic validation


# --- POST /api/assemble ----------------------------------------------------


@pytest.fixture
def assemble_client(stress_dir: Path, tmp_path: Path):
    """A TestClient with a fake assemble_fn that captures its inputs and
    returns a stub PipelineRun-shaped object. Lets us test the endpoint
    plumbing without running the full pipeline / fbx_env subprocess."""
    captured: dict = {}

    def fake_assemble(**kwargs):
        captured.update(kwargs)
        # Touch the output path so the endpoint sees a "real" file.
        out_fbx = kwargs["out_fbx"]
        Path(out_fbx).parent.mkdir(parents=True, exist_ok=True)
        Path(out_fbx).write_bytes(b"FAKE_FBX")

        class _PhaseBStub:
            class _Decisions:
                bones = []
                llm_model_id = "stub"
            decisions = _Decisions()

        class _Run:
            output_fbx = out_fbx
            donor_id = "maya"
            target_id = kwargs["target_id"]
            score = 1.0
            candidates = [("maya", 1.0)]
            cache_hit = False
            cache_key = "stubkey"
            class _EditPlan:
                drops: list[int] = []
                renames: dict[int, str] = {}
                reparents: dict[int, int] = {}
            edit_plan = _EditPlan()
            warnings: list = []
            phase_b = _PhaseBStub()
            notes: list[str] = ["stub run"]
            bone_transforms: list = []
            target_bone_transforms: list = []
        return _Run()

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    app = create_app(
        data_dir=stress_dir,
        registry=AvatarRegistry.load_default(),
        assemble_fn=fake_assemble,
        output_dir=output_dir,
    )
    client = TestClient(app)
    return client, captured, output_dir


def test_assemble_endpoint_runs_pipeline_and_writes_manifest(
    assemble_client, maya_fbx_ascii: Path, stress_dir: Path,
):
    client, captured, output_dir = assemble_client
    r = client.post("/api/assemble", json={
        "target_id": "maya",
        "clothing_path": str(maya_fbx_ascii),
        "drop_bone_ids": [42, 100],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"]  # auto-derived from clothing stem
    assert body["output_fbx"]
    # The endpoint forwards drop_bone_ids as a set
    assert captured["user_drop_bone_ids"] == {42, 100}
    assert captured["target_id"] == "maya"
    assert str(captured["clothing_fbx"]) == str(maya_fbx_ascii)
    # A manifest file is written into data_dir so the FE's recent-runs list
    # picks it up next time it's queried.
    manifests = list(stress_dir.glob("*.manifest.json"))
    assert manifests, "endpoint must write a manifest into data_dir"


def test_assemble_endpoint_omits_drop_bone_ids_when_empty(
    assemble_client, maya_fbx_ascii: Path,
):
    client, captured, _ = assemble_client
    r = client.post("/api/assemble", json={
        "target_id": "maya",
        "clothing_path": str(maya_fbx_ascii),
    })
    assert r.status_code == 200
    # No drops means we pass None (or empty), not an undefined behavior.
    assert captured.get("user_drop_bone_ids") in (None, set())


def test_assemble_endpoint_forwards_target_drop_bone_ids(
    assemble_client, maya_fbx_ascii: Path,
):
    """Target-side drops let the modder strip the target avatar's bundled
    clothing bones before clothing splice. Endpoint must forward them."""
    client, captured, _ = assemble_client
    r = client.post("/api/assemble", json={
        "target_id": "maya",
        "clothing_path": str(maya_fbx_ascii),
        "target_drop_bone_ids": [7, 88],
    })
    assert r.status_code == 200, r.text
    assert captured["target_drop_bone_ids"] == {7, 88}


def test_assemble_endpoint_omits_target_drop_bone_ids_when_empty(
    assemble_client, maya_fbx_ascii: Path,
):
    client, captured, _ = assemble_client
    r = client.post("/api/assemble", json={
        "target_id": "maya",
        "clothing_path": str(maya_fbx_ascii),
    })
    assert r.status_code == 200
    assert captured.get("target_drop_bone_ids") in (None, set())


def test_assemble_endpoint_forwards_mesh_drop_ids(
    assemble_client, maya_fbx_ascii: Path,
):
    """FE outliner sends mesh-Model ids (Body, Cloth, Hair, ...). Endpoint
    must forward them to the pipeline so drop_mesh_edits runs."""
    client, captured, _ = assemble_client
    r = client.post("/api/assemble", json={
        "target_id": "maya",
        "clothing_path": str(maya_fbx_ascii),
        "drop_mesh_ids": [500, 600],
        "target_drop_mesh_ids": [700, 800, 900],
    })
    assert r.status_code == 200, r.text
    assert captured["drop_mesh_ids"] == {500, 600}
    assert captured["target_drop_mesh_ids"] == {700, 800, 900}


def test_assemble_endpoint_forwards_blend_shape_channel_drop_ids(
    assemble_client, maya_fbx_ascii: Path,
):
    """FE blendshape list sends channel SubDeformer ids. Endpoint must
    forward them to the pipeline so drop_blend_shape_channel_edits runs."""
    client, captured, _ = assemble_client
    r = client.post("/api/assemble", json={
        "target_id": "maya",
        "clothing_path": str(maya_fbx_ascii),
        "drop_blend_shape_channel_ids": [800, 801],
        "target_drop_blend_shape_channel_ids": [900, 901, 902],
    })
    assert r.status_code == 200, r.text
    assert captured["drop_blend_shape_channel_ids"] == {800, 801}
    assert captured["target_drop_blend_shape_channel_ids"] == {900, 901, 902}


def test_assemble_endpoint_omits_blend_shape_channel_drops_when_empty(
    assemble_client, maya_fbx_ascii: Path,
):
    client, captured, _ = assemble_client
    r = client.post("/api/assemble", json={
        "target_id": "maya",
        "clothing_path": str(maya_fbx_ascii),
    })
    assert r.status_code == 200
    assert captured.get("drop_blend_shape_channel_ids") in (None, set())
    assert captured.get("target_drop_blend_shape_channel_ids") in (None, set())


def test_inspect_returns_blend_shape_channels(
    client: TestClient, maya_fbx_ascii: Path,
):
    """Inspect must surface the channel list so the FE can render checkboxes.
    Each item has channel_id + name + owner_id + owner_name + owner_mesh."""
    r = client.post("/api/clothings/inspect", json={"path": str(maya_fbx_ascii)})
    assert r.status_code == 200, r.text
    payload = r.json()
    channels = payload.get("blend_shape_channels")
    assert isinstance(channels, list)
    assert channels, "expected at least one channel in Maya.fbx"
    sample = channels[0]
    assert set(sample.keys()) >= {
        "channel_id", "name", "owner_id", "owner_name", "owner_mesh",
    }
    # Sorted by (owner_mesh, name, channel_id) — grouped by owning mesh so
    # duplicate channel names (several meshes share e.g. "High heeled") are
    # distinguishable in the FE.
    keys = [(c.get("owner_mesh") or "", c["name"], c["channel_id"]) for c in channels]
    assert keys == sorted(keys)


def test_avatar_inspect_returns_blend_shape_channels(client: TestClient):
    """Same shape on the avatar side so FE can render the target's morph list."""
    r = client.get("/api/avatars/maya/inspect")
    assert r.status_code == 200
    payload = r.json()
    assert isinstance(payload.get("blend_shape_channels"), list)


def test_inspect_blend_shape_channels_include_deform_percent(
    client: TestClient, maya_fbx_ascii: Path,
):
    """Each channel row must include the on-disk DeformPercent so the FE
    slider can start at the file's actual value (not always 0)."""
    r = client.post("/api/clothings/inspect", json={"path": str(maya_fbx_ascii)})
    assert r.status_code == 200, r.text
    payload = r.json()
    channels = payload["blend_shape_channels"]
    assert channels
    sample = channels[0]
    assert "deform_percent" in sample
    assert isinstance(sample["deform_percent"], (int, float))


def test_assemble_endpoint_forwards_blend_shape_channel_overrides(
    assemble_client, maya_fbx_ascii: Path,
):
    """FE slider sends {channel_id, deform_percent} per channel touched.
    Endpoint must convert into a dict[int, float] for the orchestrator."""
    client, captured, _ = assemble_client
    r = client.post("/api/assemble", json={
        "target_id": "maya",
        "clothing_path": str(maya_fbx_ascii),
        "blend_shape_channel_overrides": [
            {"channel_id": 800, "deform_percent": 50},
            {"channel_id": 801, "deform_percent": 33.5},
        ],
        "target_blend_shape_channel_overrides": [
            {"channel_id": 900, "deform_percent": 75},
        ],
    })
    assert r.status_code == 200, r.text
    assert captured["blend_shape_channel_overrides"] == {800: 50.0, 801: 33.5}
    assert captured["target_blend_shape_channel_overrides"] == {900: 75.0}


def test_assemble_endpoint_omits_channel_overrides_when_empty(
    assemble_client, maya_fbx_ascii: Path,
):
    client, captured, _ = assemble_client
    r = client.post("/api/assemble", json={
        "target_id": "maya",
        "clothing_path": str(maya_fbx_ascii),
    })
    assert r.status_code == 200
    assert captured.get("blend_shape_channel_overrides") in (None, {})
    assert captured.get("target_blend_shape_channel_overrides") in (None, {})


# --- bone moves ----------------------------------------------------------------


def test_assemble_endpoint_forwards_bone_transforms(assemble_client, maya_fbx_ascii: Path):
    """The FE sends one entry per moved bone with the Unity-Inspector numbers
    it showed the modder. The endpoint hands the orchestrator
    BoneTransformEdit objects (frame preserved; conversion happens in Phase C)."""
    from rigforge.ascii_fbx.transform import LocalTransform
    from rigforge.pipeline.bone_transforms import BoneTransformEdit

    client, captured, _ = assemble_client
    r = client.post("/api/assemble", json={
        "target_id": "maya",
        "clothing_path": str(maya_fbx_ascii),
        "bone_transforms": [
            {"bone_id": 42, "position": [0, 0.05, 0]},
            {"bone_id": 43, "rotation": [0, 0, 10], "scale": [1, 1, 1.5], "frame": "fbx"},
        ],
        "target_bone_transforms": [
            {"bone_id": 7, "position": [1, 2, 3], "frame": "unity"},
        ],
    })
    assert r.status_code == 200, r.text
    assert captured["bone_transforms"] == [
        BoneTransformEdit(bone=42, transform=LocalTransform(position=(0.0, 0.05, 0.0)), frame="unity"),
        BoneTransformEdit(
            bone=43,
            transform=LocalTransform(rotation=(0.0, 0.0, 10.0), scale=(1.0, 1.0, 1.5)),
            frame="fbx",
        ),
    ]
    assert captured["target_bone_transforms"] == [
        BoneTransformEdit(bone=7, transform=LocalTransform(position=(1.0, 2.0, 3.0)), frame="unity"),
    ]


def test_assemble_endpoint_omits_bone_transforms_when_empty(
    assemble_client, maya_fbx_ascii: Path,
):
    client, captured, _ = assemble_client
    r = client.post("/api/assemble", json={
        "target_id": "maya",
        "clothing_path": str(maya_fbx_ascii),
    })
    assert r.status_code == 200
    assert captured.get("bone_transforms") in (None, [])
    assert captured.get("target_bone_transforms") in (None, [])


@pytest.mark.parametrize("bad", [
    {"bone_id": 1, "position": [0, 1]},              # wrong arity
    {"bone_id": 1, "position": [0, 1, 2], "frame": "blender"},
    {"bone_id": 1},                                   # nothing to set
    {"position": [0, 1, 2]},                          # no bone
])
def test_assemble_endpoint_rejects_malformed_bone_transform(
    assemble_client, maya_fbx_ascii: Path, bad,
):
    client, captured, _ = assemble_client
    r = client.post("/api/assemble", json={
        "target_id": "maya",
        "clothing_path": str(maya_fbx_ascii),
        "bone_transforms": [bad],
    })
    assert r.status_code == 422, r.text
    assert "bone_transforms" not in captured


def test_manifest_records_bone_transforms(assemble_client, maya_fbx_ascii: Path):
    """The manifest must carry the moves (the FE's other selections live only
    in browser storage — this one is written down)."""
    client, captured, output_dir = assemble_client
    r = client.post("/api/assemble", json={
        "target_id": "maya",
        "clothing_path": str(maya_fbx_ascii),
        "bone_transforms": [{"bone_id": 42, "position": [0, 0.05, 0]}],
    })
    assert r.status_code == 200, r.text
    # The stub run doesn't echo inputs; the manifest builder reads them off
    # the run object — so patch the stub's list to what the endpoint parsed.
    # (Real PipelineRun objects carry them; see test_cli_manifest.)
    assert "bone_transforms" in r.json()["manifest"]["input"]


def test_inspect_returns_local_transform_in_unity_frame(
    client: TestClient, maya_fbx_ascii: Path,
):
    """Every bone record carries `local_transform_unity` — the parent-local
    position / rotation / scale as Unity's Inspector will show them — so the
    FE can prefill the move panel with the file's current values."""
    from rigforge.ascii_fbx.lexer import parse
    from rigforge.ascii_fbx.sections import extract, model_local_transform
    from rigforge.ascii_fbx.transform import fbx_to_unity

    r = client.post("/api/clothings/inspect", json={"path": str(maya_fbx_ascii)})
    assert r.status_code == 200
    bones = r.json()["bones"]
    for b in bones:
        lt = b["local_transform_unity"]
        assert set(lt) == {"position", "rotation", "scale"}
        assert all(len(lt[k]) == 3 for k in lt)

    view = extract(parse(maya_fbx_ascii.read_bytes()))
    wrist = next(b for b in bones if b["name"] == "Left wrist")
    expected = fbx_to_unity(model_local_transform(view, wrist["model_id"]))
    assert wrist["local_transform_unity"]["position"] == pytest.approx(expected.position)
    assert wrist["local_transform_unity"]["rotation"] == pytest.approx(expected.rotation)
    assert wrist["local_transform_unity"]["scale"] == pytest.approx(expected.scale)
    # Sanity on the mirroring: the FBX puts the left arm on +X, Unity on -X.
    assert wrist["local_transform_unity"]["position"][0] < 0


# --- streaming assemble -----------------------------------------------------


@pytest.fixture
def assemble_stream_client(stress_dir: Path, tmp_path: Path):
    """Like `assemble_client`, but the fake assemble_fn invokes progress_cb
    before returning so the streaming endpoint produces non-trivial events.
    Returns (client, captured, output_dir, scripted_progress) where
    scripted_progress is the list of (phase, note) the fake will emit."""
    captured: dict = {}
    scripted: list[tuple[str, str]] = [
        ("phase_a", "donor=maya score=1.00"),
        ("phase_b", "12 drops + 3 renames (cache hit)"),
        ("phase_c", "merge complete"),
        ("write", "wrote stub.fbx"),
    ]

    def fake_assemble(**kwargs):
        captured.update(kwargs)
        cb = kwargs.get("progress_cb")
        if cb is not None:
            for phase, note in scripted:
                cb(phase, note)
        out_fbx = kwargs["out_fbx"]
        Path(out_fbx).parent.mkdir(parents=True, exist_ok=True)
        Path(out_fbx).write_bytes(b"FAKE_FBX")

        class _PhaseBStub:
            class _Decisions:
                bones = []
                llm_model_id = "stub"
            decisions = _Decisions()

        class _Run:
            output_fbx = out_fbx
            donor_id = "maya"
            target_id = kwargs["target_id"]
            score = 1.0
            candidates = [("maya", 1.0)]
            cache_hit = False
            cache_key = "stubkey"
            class _EditPlan:
                drops: list[int] = []
                renames: dict[int, str] = {}
                reparents: dict[int, int] = {}
            edit_plan = _EditPlan()
            warnings: list = []
            phase_b = _PhaseBStub()
            notes: list[str] = ["stub run"]
            bone_transforms: list = []
            target_bone_transforms: list = []
        return _Run()

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    app = create_app(
        data_dir=stress_dir,
        registry=AvatarRegistry.load_default(),
        assemble_fn=fake_assemble,
        output_dir=output_dir,
    )
    client = TestClient(app)
    return client, captured, output_dir, scripted


def _read_ndjson_events(response) -> list[dict]:
    """Decode the NDJSON body of a streaming response into events."""
    events = []
    for line in response.iter_lines():
        if not line:
            continue
        # iter_lines yields str in TestClient; be tolerant of bytes too.
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        events.append(json.loads(line))
    return events


def test_assemble_stream_emits_progress_events_in_order(
    assemble_stream_client, maya_fbx_ascii: Path,
):
    """The endpoint must stream a 'started' event, one 'progress' per
    callback invocation, and a final 'done' event with the same payload
    shape as the sync endpoint."""
    client, _captured, _out, scripted = assemble_stream_client
    with client.stream("POST", "/api/assemble/stream", json={
        "target_id": "maya",
        "clothing_path": str(maya_fbx_ascii),
    }) as response:
        assert response.status_code == 200
        # content-type signals NDJSON for the FE to parse line-by-line
        assert "application/x-ndjson" in response.headers["content-type"]
        events = _read_ndjson_events(response)

    types = [e["event"] for e in events]
    assert types[0] == "started"
    # Final event is done; no heartbeat needed for a fast fake
    assert types[-1] == "done"
    # All scripted progress events arrived in order
    progress_pairs = [(e["phase"], e["note"]) for e in events if e["event"] == "progress"]
    assert progress_pairs == scripted

    # The done event carries the same body shape as /api/assemble
    done = events[-1]
    assert set(done.keys()) >= {"event", "id", "output_fbx", "manifest"}
    assert done["manifest"]["id"] == done["id"]


def test_assemble_stream_emits_error_event_on_failure(
    stress_dir: Path, tmp_path: Path, maya_fbx_ascii: Path,
):
    """When assemble_fn raises, the stream must surface a single 'error'
    event and close cleanly — not 500. The 2xx status is correct: we
    successfully started the run; the failure is in-band."""
    def boom(**kwargs):
        cb = kwargs.get("progress_cb")
        if cb is not None:
            cb("phase_a", "starting")
        raise RuntimeError("synthetic phase B failure")

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    app = create_app(
        data_dir=stress_dir,
        registry=AvatarRegistry.load_default(),
        assemble_fn=boom,
        output_dir=output_dir,
    )
    client = TestClient(app)
    with client.stream("POST", "/api/assemble/stream", json={
        "target_id": "maya",
        "clothing_path": str(maya_fbx_ascii),
    }) as response:
        assert response.status_code == 200
        events = _read_ndjson_events(response)
    types = [e["event"] for e in events]
    assert types[0] == "started"
    assert types[-1] == "error"
    assert "synthetic phase B failure" in events[-1]["error"]


def test_assemble_stream_validates_target_before_streaming(
    assemble_stream_client, maya_fbx_ascii: Path,
):
    """Validation errors (unknown target, missing file) must come back as
    plain HTTP 4xx — not as in-band error events — so client error handling
    stays the same as the sync endpoint."""
    client, _captured, _out, _scripted = assemble_stream_client
    r = client.post("/api/assemble/stream", json={
        "target_id": "no-such",
        "clothing_path": str(maya_fbx_ascii),
    })
    assert r.status_code == 400


def test_assemble_stream_validates_missing_file_before_streaming(
    assemble_stream_client, tmp_path: Path,
):
    client, _captured, _out, _scripted = assemble_stream_client
    bogus = tmp_path / "nope.fbx"
    r = client.post("/api/assemble/stream", json={
        "target_id": "maya",
        "clothing_path": str(bogus),
    })
    assert r.status_code == 404


def test_assemble_endpoint_rejects_unknown_target(
    assemble_client, maya_fbx_ascii: Path,
):
    client, _, _ = assemble_client
    r = client.post("/api/assemble", json={
        "target_id": "no-such-avatar",
        "clothing_path": str(maya_fbx_ascii),
    })
    assert r.status_code == 400


def test_assemble_endpoint_rejects_missing_clothing(
    assemble_client, tmp_path: Path,
):
    client, _, _ = assemble_client
    bogus = tmp_path / "nope.fbx"
    r = client.post("/api/assemble", json={
        "target_id": "maya",
        "clothing_path": str(bogus),
    })
    assert r.status_code == 404


def test_inspect_subtree_size_matches_descendant_count(
    client: TestClient, maya_fbx_ascii: Path,
):
    """subtree_size on each bone equals 1 + count of transitive descendants —
    the FE relies on this to show 'dropping this drops N bones'."""
    r = client.post("/api/clothings/inspect", json={"path": str(maya_fbx_ascii)})
    body = r.json()
    by_id = {b["model_id"]: b for b in body["bones"]}
    # Build children index
    children: dict[int, list[int]] = {}
    for b in body["bones"]:
        if b["parent_id"] is not None:
            children.setdefault(b["parent_id"], []).append(b["model_id"])

    def descendants(bid: int) -> int:
        n = 1
        for c in children.get(bid, []):
            n += descendants(c)
        return n

    # Spot-check 5 bones
    for b in body["bones"][:5]:
        assert b["subtree_size"] == descendants(b["model_id"])


# ---------------------------------------------------------------------------
# Multi-clothing compose: ONE request carrying every garment -> ONE run
# ---------------------------------------------------------------------------


def test_assemble_endpoint_chains_multiple_clothings_into_one_run(
    assemble_client, maya_fbx_ascii: Path, tmp_path: Path,
):
    client, captured, output_dir = assemble_client
    second = tmp_path / "second_garment.fbx"
    second.write_bytes(b"stub")   # existence only; the fake never reads it
    r = client.post("/api/assemble", json={
        "target_id": "maya",
        "clothings": [
            {"clothing_path": str(maya_fbx_ascii), "drop_mesh_ids": [10]},
            {"clothing_path": str(second), "drop_blend_shape_channel_ids": [901],
             "blend_shape_channel_overrides": [{"channel_id": 902, "deform_percent": 30}]},
        ],
        "target_drop_mesh_ids": [200],
    })
    assert r.status_code == 200, r.text
    specs = captured["clothings"]
    assert [str(s.clothing_fbx) for s in specs] == [str(maya_fbx_ascii), str(second)]
    assert specs[0].drop_mesh_ids == {10}
    assert specs[0].drop_blend_shape_channel_ids is None
    assert specs[1].drop_blend_shape_channel_ids == {901}
    assert specs[1].blend_shape_channel_overrides == {902: 30.0}
    assert "clothing_fbx" not in captured, "multi form must not also send the single path"
    assert captured["target_drop_mesh_ids"] == {200}
    body = r.json()
    stem = f"{maya_fbx_ascii.stem}+second_garment"
    assert body["id"] == stem
    assert Path(body["output_fbx"]).name == f"{stem}__maya.fbx"


def test_assemble_endpoint_forwards_attach_to_bone_id(
    assemble_client, maya_fbx_ascii: Path, rigid_garment_fbx: Path,
):
    client, captured, _out = assemble_client
    r = client.post("/api/assemble", json={
        "target_id": "maya",
        "clothings": [
            {"clothing_path": str(maya_fbx_ascii)},
            {"clothing_path": str(rigid_garment_fbx), "attach_to_bone_id": 2226590603152},
        ],
    })
    assert r.status_code == 200, r.text
    specs = captured["clothings"]
    assert specs[0].attach_to_bone_id is None
    assert specs[1].attach_to_bone_id == 2226590603152


def test_assemble_endpoint_multi_clothing_missing_file_is_404(
    assemble_client, maya_fbx_ascii: Path,
):
    client, _captured, _out = assemble_client
    r = client.post("/api/assemble", json={
        "target_id": "maya",
        "clothings": [
            {"clothing_path": str(maya_fbx_ascii)},
            {"clothing_path": "C:/nope/missing_garment.fbx"},
        ],
    })
    assert r.status_code == 404
    assert "missing_garment.fbx" in r.json()["detail"]


def test_assemble_endpoint_requires_at_least_one_clothing(assemble_client):
    client, _captured, _out = assemble_client
    r = client.post("/api/assemble", json={"target_id": "maya", "clothings": []})
    assert r.status_code == 422


def test_assemble_stream_tags_events_with_clothing_index(
    assemble_stream_client, maya_fbx_ascii: Path, tmp_path: Path,
):
    """The orchestrator announces each garment with a 'clothing' progress
    event; the endpoint turns that into {"event":"clothing","index":i,...}
    and stamps every later progress event with clothing_index so the FE can
    drive one progress row per garment from a single stream."""
    client, _captured, _out, scripted = assemble_stream_client
    second = tmp_path / "second_garment.fbx"
    second.write_bytes(b"stub")
    scripted[:] = [
        ("clothing", maya_fbx_ascii.stem),
        ("phase_a", "donor=maya score=1.00"),
        ("clothing", "second_garment"),
        ("phase_a", "donor=maya score=1.00"),
        ("phase_c", "merge complete"),
        ("write", "wrote stub.fbx"),
    ]
    with client.stream("POST", "/api/assemble/stream", json={
        "target_id": "maya",
        "clothings": [
            {"clothing_path": str(maya_fbx_ascii)},
            {"clothing_path": str(second)},
        ],
    }) as response:
        assert response.status_code == 200
        events = _read_ndjson_events(response)

    clothing_events = [e for e in events if e["event"] == "clothing"]
    assert [e["index"] for e in clothing_events] == [0, 1]
    assert [e["stem"] for e in clothing_events] == [maya_fbx_ascii.stem, "second_garment"]
    assert clothing_events[1]["path"] == str(second)
    progress = [e for e in events if e["event"] == "progress"]
    assert [e["clothing_index"] for e in progress] == [0, 1, 1, 1]
    assert events[-1]["event"] == "done"
    assert events[-1]["id"] == f"{maya_fbx_ascii.stem}+second_garment"
