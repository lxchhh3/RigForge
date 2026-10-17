"""MochiFitter `_MFBase` bone-name suffix.

MochiFitter's FBX exports name every body bone `<bone>_MFBase` (`Hips_MFBase`,
`Left leg_MFBase`, ...), where the native FBXs carry the bare name. Matched
verbatim, none of those bones is a target bone: Phase A scored maya 0.000 and
the save_08 compose run died on SimpleLeotard_save08.fbx. Both name matchers
(Phase A's donor fingerprint, Phase C's name-based repoint) must read such a
bone as its base name.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from rigforge.ascii_fbx.lexer import parse
from rigforge.ascii_fbx.sections import extract
from rigforge.avatars.fingerprint import strip_rig_suffix
from rigforge.avatars.registry import AvatarRegistry
from rigforge.pipeline.edit_plan import EditPlan
from rigforge.pipeline.phase_a import run_phase_a


@pytest.mark.parametrize("name,expected", [
    ("Hips_MFBase", "Hips"),
    ("Left leg_MFBase", "Left leg"),
    ("Hips_L_MFBase", "Hips_L"),
    # Untouched: no suffix, the suffix text mid-name, MochiFitter's other helpers.
    ("Hips", "Hips"),
    ("MFBase_root", "MFBase_root"),
    ("origORS_UpperLeg_L", "origORS_UpperLeg_L"),
    ("", ""),
])
def test_strip_rig_suffix(name, expected):
    assert strip_rig_suffix(name) == expected


def _ascii_fbx(bones: list[tuple[int, str, int]]) -> bytes:
    """Minimal ASCII FBX: (model_id, name, parent_id) LimbNodes, parent 0 = root."""
    models = "".join(
        f'\tModel: {mid}, "Model::{name}", "LimbNode" {{\n\t}}\n' for mid, name, _ in bones
    )
    conns = "".join(f'\tC: "OO",{mid},{parent}\n' for mid, _, parent in bones)
    return (
        "; FBX 7.4.0 project file\n"
        f"Objects:  {{\n{models}}}\n"
        f"Connections:  {{\n{conns}}}\n"
    ).encode("utf-8")


def test_phase_a_picks_target_for_mfbase_suffixed_clothing(tmp_path: Path):
    """A MochiFitter export rigged for maya (every body bone `_MFBase`,
    plus MochiFitter's own `origORS_*` helpers) identifies as maya."""
    registry = AvatarRegistry.load_default()
    maya_names = list(registry.get("maya").canonical_to_name.values())[:20]
    bones = [(1000 + i, f"{n}_MFBase", 1000 if i else 0) for i, n in enumerate(maya_names)]
    bones.append((2000, "origORS_Head", 1000))
    fbx = tmp_path / "SimpleLeotard_save08.fbx"
    fbx.write_bytes(_ascii_fbx(bones))

    a = run_phase_a(fbx, registry, work_dir=tmp_path / "work", target_id="maya")

    assert a.donor_id == "maya"
    assert a.score == pytest.approx(1.0)
    # The view keeps the file's real names; only the matching reads base names.
    assert {b.name for b in a.view.limb_bones()} == {n for _, n, _ in bones}


def test_name_based_repoint_matches_mfbase_suffixed_bones():
    """`Hips_MFBase` / `Spine_MFBase` repoint onto the target's Hips / Spine
    (stripped, not ridden along as duplicates); a helper with no target
    counterpart still rides along."""
    from rigforge.pipeline.phase_c import _build_name_based_repoint

    clothing = extract(parse(_ascii_fbx([
        (100, "Hips_MFBase", 0),
        (101, "Spine_MFBase", 100),
        (102, "origORS_Hips", 100),
    ])))
    target = extract(parse(_ascii_fbx([
        (200, "Hips", 0),
        (201, "Spine", 200),
    ])))
    notes: list[str] = []

    repoint, kept = _build_name_based_repoint(clothing, target, EditPlan(), notes=notes)

    assert repoint == {100: 200, 101: 201}
    assert kept == {100, 101}
    assert any("matched 2/3" in n for n in notes)


def test_name_based_repoint_exact_name_wins_over_suffix():
    """If the target itself carries a `_MFBase` bone, the exact name match
    takes it, not the stripped base name."""
    from rigforge.pipeline.phase_c import _build_name_based_repoint

    clothing = extract(parse(_ascii_fbx([(100, "Hips_MFBase", 0)])))
    target = extract(parse(_ascii_fbx([
        (200, "Hips", 0),
        (201, "Hips_MFBase", 200),
    ])))

    repoint, _ = _build_name_based_repoint(clothing, target, EditPlan(), notes=[])

    assert repoint == {100: 201}
