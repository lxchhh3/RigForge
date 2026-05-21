"""Same-avatar pass-through e2e with mock decisions (no LLM).

Runs ClassicChic_Moe -> target=Moe through the pipeline. This is the
natural production pairing: clothing rigged for Moe, targeted at Moe.
Phase C takes the pass-through branch (no armature merge), but Phase B
still classifies + normalizes bones and EditPlan synthesis still drops
aux bones.

Purpose: verify the end-to-end pipeline produces a clean assembly for a
real clothing FBX against a real curated avatar, isolated from real-LLM
flakiness. The cross-avatar (donor != target) branch is exercised
separately by tests/test_pipeline.py — it's the structural code path
for when someone has clothing rigged for a non-curated booth base.

Usage:
    PYTHONPATH=. python training/smoke_cross_merge_mock.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from rigforge.ascii_fbx.convert import compare
from rigforge.ascii_fbx.lexer import parse as parse_fbx
from rigforge.ascii_fbx.sections import extract
from rigforge.avatars.registry import AvatarRegistry
from rigforge.canonical.decisions import Decision, DecisionSet
from rigforge.canonical.schema import CanonicalSchema
from rigforge.llm.mock import MockLLMClient
from rigforge.manifest import write_manifest
from rigforge.pipeline.orchestrator import assemble


REPO_ROOT = Path(__file__).resolve().parent.parent
CLOTHING_BIN = Path("D:/2files/models/vrc/ASCII_models/clothing/ClassicChic_Moe/FBX_Moe.fbx")
CLOTHING_ASCII = Path("D:/2files/models/vrc/ASCII_models/clothing/ClassicChic_Moe/FBX_Moe_ascii.fbx")
GROUND_TRUTH = Path("D:/2files/models/vrc/ASCII_models/clothing/ClassicChic_Maya/FBX_Maya.fbx")


def _build_mock_decisions(clothing_ascii_path: Path, target_canonical_to_name: dict) -> DecisionSet:
    """Generate a DecisionSet that maps every clothing limb bone to either a
    canonical role (when the display name matches target's canonical_to_name)
    or Secondary.<name> with verdict=keep."""
    raw = clothing_ascii_path.read_bytes()
    view = extract(parse_fbx(raw))
    display_to_role = {v: k for k, v in target_canonical_to_name.items()}

    bones = []
    for b in view.limb_bones():
        role = display_to_role.get(b.name)
        if role is not None:
            bones.append(Decision(model_id=b.model_id, role=role, verdict="keep",
                                   confidence=1.0))
        else:
            bones.append(Decision(model_id=b.model_id,
                                   role=f"Secondary.{b.name}", verdict="keep",
                                   confidence=1.0))
    return DecisionSet(bones=bones, llm_model_id="mock-cross-merge")


def main() -> int:
    if not CLOTHING_BIN.exists():
        print(f"error: {CLOTHING_BIN} not found", file=sys.stderr)
        return 1
    if not CLOTHING_ASCII.exists():
        print(f"error: {CLOTHING_ASCII} not found", file=sys.stderr)
        return 1

    work = REPO_ROOT / "data" / "training" / "_smoke_passthrough_mock"
    work.mkdir(parents=True, exist_ok=True)
    out_fbx = work / "classicchic_moe_assembly.fbx"

    schema = CanonicalSchema.load_default()
    registry = AvatarRegistry.load_default()

    # Natural pairing: clothing rigged for Moe, target = Moe. Phase C will
    # take the pass-through branch (no armature merge); Phase B still does
    # the canonical-role assignment and aux drops.
    moe = registry.get("moe")
    decisions = _build_mock_decisions(CLOTHING_ASCII, moe.canonical_to_name)
    print(f"[setup] built {len(decisions.bones)} mock decisions")
    canonical = sum(1 for d in decisions.bones if "." in d.role and d.role.split(".")[0] != "Secondary")
    print(f"        canonical roles assigned: {canonical}")
    # Write decisions to disk for MockLLMClient
    decisions_path = work / "decisions.json"
    decisions_path.write_text(decisions.model_dump_json(indent=2))

    client = MockLLMClient(decisions_path, model_id="mock-passthrough")

    print(f"[run] ClassicChic_Moe -> Moe (pass-through)")
    print(f"      input:  {CLOTHING_BIN}")
    print(f"      target: moe  (registry avatars: {registry.ids()})")

    t0 = time.time()
    # Use ASCII directly so the runtime bone-id space matches the IDs we
    # built mock decisions from. (bin_to_ascii via fbx_env may regenerate
    # internal IDs; passing the pre-converted ASCII keeps them stable.)
    run = assemble(
        clothing_fbx=CLOTHING_ASCII,
        target_id="moe",
        out_fbx=out_fbx,
        registry=registry,
        schema=schema,
        llm_client=client,
        cache=None,
        work_dir=work / "work",
    )
    elapsed = time.time() - t0
    print(f"[run] done in {elapsed:.1f}s")
    print(f"      donor={run.donor_id} score={run.score:.3f}")
    print(f"      edits: {len(run.edit_plan.renames)} renames, "
          f"{len(run.edit_plan.drops)} drops")
    print(f"      output: {run.output_fbx}")
    print(f"      notes:")
    for n in run.notes:
        print(f"        - {n}")
    if run.warnings:
        print(f"      warnings ({len(run.warnings)}):")
        for w in run.warnings:
            print(f"        - {w.rule}: {w.message}")

    manifest_path = out_fbx.with_suffix(".manifest.json")
    write_manifest(run, manifest_path)
    print(f"      manifest: {manifest_path}")

    # Sanity: with all-keep mock decisions (no aux drops, no renames since
    # clothing already uses Moe's canonical names), output should be
    # structurally identical to the input clothing modulo bin↔ASCII roundtrip.
    print(f"[compare] running fbx_compare(assembled, ClassicChic_Moe input)...")
    cr = compare(out_fbx, CLOTHING_BIN)
    print(f"          identical={cr.identical}  drift={cr.drift_count}  schema_match={cr.schema_match}")
    diff_lines = [l for l in cr.raw_output.splitlines() if "DIFF" in l]
    if diff_lines:
        print(f"          first {min(15, len(diff_lines))} diff lines:")
        for line in diff_lines[:15]:
            print(f"          {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
