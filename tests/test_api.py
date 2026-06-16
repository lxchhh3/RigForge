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
