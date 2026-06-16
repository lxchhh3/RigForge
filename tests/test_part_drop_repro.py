"""Reproduction of the 6/16 clothing-side part-drop bug on the REAL input.

History: assembling `retargeted_20260615_233505_FBX_MANUKA` (binary) onto maya,
the user deselected clothing parts but every one survived. Root cause: inspect
and Phase A each ran their own bin->ASCII conversion, and the FBX SDK mints node
ids from memory pointers, so the FE's drop ids never matched Phase A's view ->
`drop_mesh_edits` returned [] (silent no-op).

This test replays that exact input through the fixed flow (one shared, on-disk
bin->ASCII cache) and asserts the property that was broken: a mesh id read the
way the inspect endpoint reads it resolves to the SAME mesh in Phase A's view,
the drop produces real edits, and applying them removes the mesh from the bytes.

Runs the real fbx_env converter once (then cached). Skips when the toolchain or
the clothing fixture isn't present on this machine.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from rigforge.ascii_fbx.convert import ConverterConfig, bin_to_ascii_cached
from rigforge.ascii_fbx.edits import apply_edits, drop_mesh_edits
from rigforge.ascii_fbx.lexer import parse
from rigforge.ascii_fbx.sections import extract
from rigforge.avatars.registry import AvatarRegistry
from rigforge.pipeline.phase_a import run_phase_a

# The actual clothing the 6/16 run used (binary FBX from the retargeter output).
CLOTHING = Path(
    "D:/2files/unity_proj/MochiFitter_VCC/Assets/OutfitRetargetingSystem/"
    "Outputs/retargeted_20260615_233505_FBX_MANUKA.fbx"
)


@pytest.fixture(scope="module")
def _require_real_inputs():
    cfg = ConverterConfig.from_env()
    if not cfg.fbx_env_python.exists() or not cfg.toolchain_dir.exists():
        pytest.skip("fbx_env toolchain not present")
    if not CLOTHING.is_file():
        pytest.skip(f"clothing fixture missing: {CLOTHING}")


def _mesh_map(view) -> dict[int, str]:
    return {b.model_id: b.name for b in view.bones.values() if b.type_class == "Mesh"}


def test_inspect_ids_align_with_phase_a_and_drops_fire(tmp_path, _require_real_inputs):
    registry = AvatarRegistry.load_default()
    cache_dir = tmp_path / "_fbx_ascii_cache"

    # --- inspect path: exactly how the API reads ids for the FE outliner.
    inspect_ascii = bin_to_ascii_cached(CLOTHING, cache_dir)
    inspect_view = extract(parse(inspect_ascii.read_bytes()))
    inspect_meshes = _mesh_map(inspect_view)
    assert inspect_meshes, "clothing should expose Mesh models"

    # --- pipeline path: Phase A, pointed at the SAME shared cache dir (this is
    #     what the orchestrator now does). Reuses the cached conversion, so the
    #     ids match by construction. If anyone reverts Phase A to convert into
    #     its own dir, these maps diverge and this test fails.
    a = run_phase_a(
        CLOTHING, registry, work_dir=tmp_path, ascii_cache_dir=cache_dir,
    )
    assert _mesh_map(a.view) == inspect_meshes

    # --- the operation that used to be a silent no-op now fires for every id
    #     the FE could send.
    for mid, name in inspect_meshes.items():
        assert drop_mesh_edits(a.view, mid), f"drop_mesh_edits no-op for {name} ({mid})"

    # --- end-to-end on the bytes: dropping one mesh removes its Model node.
    victim_id, victim_name = next(iter(inspect_meshes.items()))
    src = a.view.document.source
    needle = f'"Model::{victim_name}"'.encode()
    assert needle in src  # sanity: present before the drop
    out = apply_edits(src, drop_mesh_edits(a.view, victim_id))
    assert needle not in out, f"{victim_name} survived its own drop"
