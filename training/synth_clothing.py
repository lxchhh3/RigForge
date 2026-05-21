"""Synthetic clothing fixture generator.

Until real booth clothing assets land, we exercise the v1 pipeline on a
synthetic clothing FBX derived from Maya.fbx — perturbations are applied to
the bone names so Phase B has something non-trivial to do. The output is an
ASCII FBX + a sidecar decisions JSON that a MockLLMClient can replay.

Perturbations (PLAN.md training augmentation, light v1 subset):
    - prefix the 5 spine bones with `JBip_`
    - (deferred) separator swaps, aux-bone injection, translation noise

The set of perturbed bones is intentionally small so Phase A's Jaccard
fingerprint stays above the 0.85 default threshold (most bones unchanged).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rigforge.ascii_fbx.edits import apply_edits, rename_bone_edits
from rigforge.ascii_fbx.lexer import parse
from rigforge.ascii_fbx.sections import extract
from rigforge.avatars.registry import AvatarRegistry
from rigforge.canonical.decisions import Decision, DecisionSet


# Bones to perturb (each prefixed with PREFIX). Only spine-chain entries from
# Maya's canonical_to_name — picking these guarantees fingerprint match stays
# high while still giving Phase B real work to do.
DEFAULT_PERTURB_TARGETS = ("Hips", "Spine", "Chest", "Neck", "Head")
DEFAULT_PREFIX = "JBip_"


@dataclass
class SynthBuild:
    ascii_path: Path
    decisions_path: Path
    decisions: DecisionSet
    expected_renames: dict[str, str]    # synth_name -> canonical_target_name


def build_synth_clothing(
    *,
    source_ascii: Path,
    out_ascii: Path,
    out_decisions: Path,
    avatar: str = "maya",
    registry: Optional[AvatarRegistry] = None,
    perturb_targets: tuple[str, ...] = DEFAULT_PERTURB_TARGETS,
    prefix: str = DEFAULT_PREFIX,
) -> SynthBuild:
    """Generate the synthetic clothing FBX + matched mock-LLM decisions.

    The decisions assign each perturbed bone its canonical role (with the
    target avatar's display name) so Phase B's EditPlan undoes the perturbation.
    """
    registry = registry or AvatarRegistry.load_default()
    av = registry.get(avatar)

    raw = source_ascii.read_bytes()
    view = extract(parse(raw))

    # Map canonical role -> bone in the source. canonical_to_name gives
    # role -> display_name, and we look that display_name up in the view.
    canon_to_bone = {}
    for role, display_name in av.canonical_to_name.items():
        for b in view.limb_bones():
            if b.name == display_name:
                canon_to_bone[role] = b
                break

    edits = []
    expected_renames: dict[str, str] = {}    # synth_name -> canonical_target
    perturbed_bones: list[tuple[int, str, str]] = []  # (model_id, synth_name, canonical_role)

    for role in perturb_targets:
        bone = canon_to_bone.get(role)
        if bone is None:
            continue
        synth_name = f"{prefix}{role}"
        edits.extend(rename_bone_edits(view, bone, synth_name))
        expected_renames[synth_name] = av.canonical_to_name[role]
        perturbed_bones.append((bone.model_id, synth_name, role))

    out_ascii.parent.mkdir(parents=True, exist_ok=True)
    perturbed = apply_edits(raw, edits)
    out_ascii.write_bytes(perturbed)

    # Build mock LLM decisions. Three groups:
    #   - perturbed bones → assigned their canonical role (so Phase B
    #     renames them back)
    #   - non-perturbed bones whose display name maps to a canonical role in
    #     the avatar's canonical_to_name → assigned that canonical role (so
    #     `required_roles_present` is satisfied; Phase B renames are no-ops)
    #   - everything else (eyes, secondary chains, fingers, etc.) → unique
    #     Secondary.<name> role with verdict=keep
    perturbed_view = extract(parse(perturbed))
    perturbed_id_to_role = {pid: role for pid, _, role in perturbed_bones}
    display_to_role = {v: k for k, v in av.canonical_to_name.items()}

    bones_dec: list[Decision] = []
    for b in perturbed_view.limb_bones():
        if b.model_id in perturbed_id_to_role:
            role = perturbed_id_to_role[b.model_id]
        elif b.name in display_to_role:
            role = display_to_role[b.name]
        else:
            role = f"Secondary.{b.name}"
        bones_dec.append(Decision(model_id=b.model_id, role=role, verdict="keep"))
    decisions = DecisionSet(bones=bones_dec, llm_model_id="synth-fixture")
    out_decisions.parent.mkdir(parents=True, exist_ok=True)
    out_decisions.write_text(decisions.model_dump_json(indent=2), encoding="utf-8")

    return SynthBuild(
        ascii_path=out_ascii,
        decisions_path=out_decisions,
        decisions=decisions,
        expected_renames=expected_renames,
    )
