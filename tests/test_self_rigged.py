"""Self-rigged props: a garment with its OWN small armature that shares no bone
with the target (kevin's EclatAndCrumb_Toast: Armature_Toast -> PB_Toast_01/02,
skinned toast meshes). kevin's pick 2026-10-28: merge it as-is — nothing to
match or classify — placed where it stands (move read in the target's
Armature frame, like a rigid prop) and optionally hung from a target bone."""
from __future__ import annotations

from pathlib import Path

import pytest

from rigforge.ascii_fbx.lexer import parse
from rigforge.ascii_fbx.sections import _local_matrix, _mat_mul, extract, model_world_matrix
from rigforge.ascii_fbx.transform import LocalTransform, unity_to_fbx
from rigforge.avatars.registry import DonorIdentificationError
from rigforge.pipeline.bone_transforms import BoneTransformEdit
from rigforge.pipeline.phase_a import is_self_rigged, run_phase_a
from tests.fbx_fixtures import skinned_fbx

PROP = skinned_fbx(
    [
        (30, "Armature_Toast", 0, (0, 0, 0), (0, 0, 0), "Null"),
        (31, "PB_Toast_01", 30, (0, 0.0035, 0), (0, 0, 0)),
        (32, "PB_Toast_02", 31, (0, 0.038, 0), (0, 0, 0)),
    ],
    (40, "EC_Toast", (0, 0, 0), (0, 0, 0)),
    [(0, 0, 0), (0.05, 0, 0), (0, 0.05, 0)], [(0, 1, 2)],
    {31: [(0, 1.0), (1, 1.0)], 32: [(2, 1.0)]},
    bone_worlds={31: (0, 0.0035, 0), 32: (0, 0.0415, 0)},
)
# Shares Hips with every avatar: a garment for some other avatar, not a prop.
FOREIGN = skinned_fbx(
    [(1, "Hips", 0, (0, 1, 0), (0, 0, 0)), (2, "Tail_A", 1, (0, 0.1, 0), (0, 0, 0)),
     (3, "Tail_B", 2, (0, 0.1, 0), (0, 0, 0))],
    (40, "Tail", (0, 0, 0), (0, 0, 0)), [(0, 1, 0), (0, 1.1, 0), (0, 1.2, 0)], [(0, 1, 2)],
    {2: [(0, 1.0), (1, 1.0)], 3: [(2, 1.0)]},
)
# What kevin's Toast stands at in Eclat_Coffee: under Maya's Armature, mouth height.
TOAST_UNITY = LocalTransform(position=(0.0003, -0.1192, 1.1737), rotation=(90.0, 0.0, 0.0))


@pytest.fixture(scope="module")
def registry():
    from rigforge.avatars.registry import AvatarRegistry
    return AvatarRegistry.load_default()


@pytest.fixture(scope="module")
def schema():
    from rigforge.canonical.schema import CanonicalSchema
    return CanonicalSchema.load_default()


@pytest.fixture
def prop_fbx(tmp_path: Path) -> Path:
    p = tmp_path / "EclatAndCrumb_Toast.fbx"
    p.write_bytes(PROP)
    return p


def test_a_prop_whose_bones_are_none_of_the_targets_is_self_rigged(registry):
    maya = registry.get("maya").load_ascii_view()
    assert is_self_rigged(extract(parse(PROP)), maya) is True
    assert is_self_rigged(extract(parse(FOREIGN)), maya) is False
    assert is_self_rigged(maya, maya) is False


def test_a_rigid_mesh_is_not_self_rigged(registry, rigid_garment_fbx: Path):
    maya = registry.get("maya").load_ascii_view()
    assert is_self_rigged(extract(parse(rigid_garment_fbx.read_bytes())), maya) is False


def test_phase_a_passes_a_self_rigged_prop_through_with_a_target(registry, prop_fbx, tmp_path):
    a = run_phase_a(prop_fbx, registry, work_dir=tmp_path / "w", target_id="maya")
    assert a.self_rigged is True and a.rigid is False
    assert a.donor_id == "maya"


def test_phase_a_without_a_target_still_refuses_it(registry, prop_fbx, tmp_path):
    with pytest.raises(DonorIdentificationError):
        run_phase_a(prop_fbx, registry, work_dir=tmp_path / "w")


def test_a_garment_sharing_an_avatar_bone_stays_a_garment(registry, tmp_path):
    """Sharing even one bone (Hips) makes it a partial garment for that
    avatar (Phase A scores only avatar-known bones), classified as usual."""
    p = tmp_path / "tail.fbx"
    p.write_bytes(FOREIGN)
    a = run_phase_a(p, registry, work_dir=tmp_path / "w", target_id="maya")
    assert a.self_rigged is False and a.rigid is False and a.donor_id == "maya"


def _assemble(tmp_path, monkeypatch, registry, schema, spec):
    from rigforge.pipeline import orchestrator as orch
    from rigforge.pipeline.orchestrator import assemble

    class NoLLM:
        model_id = "none"

        def classify(self, request):
            raise AssertionError("a self-rigged prop needs no classification")

    monkeypatch.setattr(orch, "ascii_to_bin",
                        lambda src, dst: Path(dst).write_bytes(Path(src).read_bytes()))
    out = tmp_path / "out.fbx"
    run = assemble(clothings=[spec], target_id="maya", out_fbx=out, registry=registry,
                   schema=schema, llm_client=NoLLM(), cache=None, work_dir=tmp_path / "work")
    return run, extract(parse(out.read_bytes()))


def _by_name(view, name):
    (b,) = [b for b in view.bones.values() if b.name == name]
    return b


def _armature(view):
    return _by_name(view, "Armature").model_id


def test_merged_as_is_under_the_armature_where_it_stood(registry, schema, prop_fbx, tmp_path, monkeypatch):
    """Its Null goes like every garment's (Blender turned a Null kept under
    Maya's Armature into a bone and hung the toast meshes rigidly from it,
    no Armature modifier — measured 2026-10-28); its top bone hangs from the
    target's Armature where it stood."""
    from rigforge.pipeline.orchestrator import ClothingSpec
    run, merged = _assemble(tmp_path, monkeypatch, registry, schema, ClothingSpec(clothing_fbx=prop_fbx))
    assert not [b for b in merged.bones.values() if b.name == "Armature_Toast"]
    top = _by_name(merged, "PB_Toast_01")
    assert top.parent_id == _armature(merged)
    w = model_world_matrix(merged, top.model_id)
    assert [w[i][3] for i in range(3)] == pytest.approx([0.0, 0.0035, 0.0], abs=1e-9)
    assert _by_name(merged, "PB_Toast_02").parent_id == top.model_id
    assert _by_name(merged, "EC_Toast").type_class == "Mesh"
    assert any("self-rigged" in n for n in run.notes), run.notes


def test_placed_and_hung_from_a_target_bone(registry, schema, prop_fbx, tmp_path, monkeypatch):
    from rigforge.pipeline.orchestrator import ClothingSpec
    head = registry.get("maya").target_bone_id("Head")
    run, merged = _assemble(tmp_path, monkeypatch, registry, schema, ClothingSpec(
        clothing_fbx=prop_fbx, attach_to_bone_id=head,
        bone_transforms=[BoneTransformEdit(bone="PB_Toast_01", transform=TOAST_UNITY)],
    ))
    top = _by_name(merged, "PB_Toast_01")
    assert top.parent_id == head
    f = unity_to_fbx(TOAST_UNITY)
    want = _mat_mul(model_world_matrix(merged, _armature(merged)),
                    _local_matrix(f.position, f.rotation, (1.0, 1.0, 1.0)))
    got = model_world_matrix(merged, top.model_id)
    assert all(abs(got[i][j] - want[i][j]) < 1e-6 for i in range(4) for j in range(4)), got


def test_its_own_lower_bones_take_ordinary_moves(registry, schema, prop_fbx, tmp_path, monkeypatch):
    from rigforge.pipeline.orchestrator import ClothingSpec
    run, merged = _assemble(tmp_path, monkeypatch, registry, schema, ClothingSpec(
        clothing_fbx=prop_fbx,
        bone_transforms=[BoneTransformEdit(bone="PB_Toast_02", frame="fbx",
                                           transform=LocalTransform(position=(0.0, 0.05, 0.0)))],
    ))
    assert _by_name(merged, "PB_Toast_02").translation_xyz == pytest.approx((0.0, 0.05, 0.0))
