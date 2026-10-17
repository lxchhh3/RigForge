"""Orchestrator — wires Phases A/B/C and the bin↔ASCII converter into a
single `assemble()` call.

End-to-end:
    bin → ASCII (converter) → Phase A → Phase B → Phase C → ASCII → bin

Compose = merge. Several clothings (`clothings=[...]`) run Phase A/B each and
ONE Phase C chain: garment 1 merges onto the target, garment 2 onto that
result, ... and the final ASCII converts once. One output file, one shared
skeleton — never one file per garment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from rigforge.ascii_fbx.convert import ascii_to_bin
from rigforge.avatars.registry import AvatarRegistry
from rigforge.cache.store import DecisionCache
from rigforge.canonical.decisions import DecisionSet
from rigforge.canonical.schema import CanonicalSchema
from rigforge.canonical.validators import Violation
from rigforge.llm.client import LLMClient

from .bone_transforms import BoneTransformEdit
from .edit_plan import EditPlan
from .phase_a import PhaseAResult, run_phase_a
from .phase_b import PhaseBError, PhaseBResult, run_phase_b
from .phase_c import PhaseCError, PhaseCResult, run_phase_c


class PipelineError(RuntimeError):
    pass


@dataclass
class ClothingSpec:
    """One garment of a compose run together with its own (FE) selections.
    Target-side selections are not here — they belong to the run, and apply
    once (to the base avatar, in the first merge)."""
    clothing_fbx: Path
    user_drop_bone_ids: Optional[set[int]] = None
    drop_mesh_ids: Optional[set[int]] = None
    drop_blend_shape_channel_ids: Optional[set[int]] = None
    blend_shape_channel_overrides: Optional[dict[int, float]] = None
    bone_transforms: Optional[list[BoneTransformEdit]] = None
    # Rigid garments only (no bones): the TARGET bone its meshes hang from so
    # they follow it. None = the scene root, which follows no bone.
    attach_to_bone_id: Optional[int] = None


@dataclass
class ClothingRun:
    """Per-garment record of a compose run: its Phase A/B and its Phase C
    step (the merge of THIS garment onto whatever came before)."""
    clothing_fbx: Path
    donor_id: str
    score: float
    candidates: list[tuple[str, float]]
    cache_hit: bool
    cache_key: str
    edit_plan: EditPlan
    warnings: list[Violation]
    phase_a: PhaseAResult
    phase_b: PhaseBResult
    phase_c: PhaseCResult
    notes: list[str] = field(default_factory=list)
    bone_transforms: list[BoneTransformEdit] = field(default_factory=list)

    @property
    def stem(self) -> str:
        return Path(self.clothing_fbx).stem


@dataclass
class PipelineRun:
    """Everything a manifest writer / CLI / tests want to inspect.

    For a multi-garment compose the scalar fields (donor_id, score, cache,
    edit_plan, phase_a/phase_b) mirror the FIRST garment for backward
    compatibility; `warnings`/`notes` are the union; `phase_c` is the FINAL
    merge (its `merged_ascii` is what shipped); `clothing_runs` has it all,
    per garment, in merge order.
    """
    output_fbx: Path
    donor_id: str
    target_id: str
    score: float
    candidates: list[tuple[str, float]]
    cache_hit: bool
    cache_key: str
    edit_plan: EditPlan
    warnings: list[Violation]
    phase_a: PhaseAResult
    phase_b: PhaseBResult
    phase_c: PhaseCResult
    notes: list[str] = field(default_factory=list)
    # The bone moves this run was asked for (both sides), echoed into the
    # manifest so a past run's moves are on record — unlike the FE's other
    # selections, which live only in browser storage.
    bone_transforms: list[BoneTransformEdit] = field(default_factory=list)
    target_bone_transforms: list[BoneTransformEdit] = field(default_factory=list)
    # One entry per garment, in merge order (a single-clothing run has one).
    clothing_runs: list[ClothingRun] = field(default_factory=list)


ProgressCallback = Callable[[str, str], None]
"""(phase, note) — phase is one of 'clothing'|'phase_a'|'phase_b'|'phase_c'|
'write'; note is a one-line human-readable description of the just-completed
step. 'clothing' announces the garment (note = its file stem) whose phases
follow — a multi-garment compose emits it once per garment, in merge order,
so a consumer can drive one progress row per garment from the single stream.

The callback is invoked at phase boundaries from the orchestrator. It runs
inline on the worker thread; the API layer wraps it to push events onto a
streaming queue."""


def assemble(
    *,
    clothing_fbx: Optional[Path] = None,
    target_id: str,
    out_fbx: Path,
    registry: AvatarRegistry,
    schema: CanonicalSchema,
    llm_client: LLMClient,
    cache: Optional[DecisionCache] = None,
    work_dir: Optional[Path] = None,
    ascii_cache_dir: Optional[Path] = None,
    user_drop_bone_ids: Optional[set[int]] = None,
    target_drop_bone_ids: Optional[set[int]] = None,
    drop_mesh_ids: Optional[set[int]] = None,
    target_drop_mesh_ids: Optional[set[int]] = None,
    drop_blend_shape_channel_ids: Optional[set[int]] = None,
    target_drop_blend_shape_channel_ids: Optional[set[int]] = None,
    blend_shape_channel_overrides: Optional[dict[int, float]] = None,
    target_blend_shape_channel_overrides: Optional[dict[int, float]] = None,
    translate_target_morphs: bool = True,
    bone_transforms: Optional[list[BoneTransformEdit]] = None,
    target_bone_transforms: Optional[list[BoneTransformEdit]] = None,
    progress_cb: Optional[ProgressCallback] = None,
    clothings: Optional[list[ClothingSpec]] = None,
) -> PipelineRun:
    """Run the full assembly pipeline and write the deliverable binary FBX.

    Give ONE garment as `clothing_fbx` (+ the flat per-clothing kwargs), or
    SEVERAL as `clothings=[ClothingSpec, ...]` — then each garment carries its
    own selections and the run produces ONE merged file: garment 1 merges onto
    the target, garment 2 onto that result, and so on. Target-side kwargs
    (target_drop_*, target overrides/moves, translate_target_morphs) apply
    once, in the first merge — they edit the base avatar's content.

    Hard-fails on any of:
      - Phase A donor identification below threshold
      - Phase B validator errors after re-prompt
      - Phase C unsupported merge case (cross-avatar in v1)

    `progress_cb` is invoked at each phase boundary so the API layer can
    stream live status to the FE. The callback signature is
    `progress_cb(phase, note)`; see [[ProgressCallback]].
    """
    def _emit(phase: str, note: str) -> None:
        if progress_cb is not None:
            try:
                progress_cb(phase, note)
            except Exception:
                # A broken callback must never crash the pipeline. The FE-side
                # consumer is best-effort; pipeline correctness comes first.
                pass

    if clothings is None:
        if clothing_fbx is None:
            raise PipelineError("assemble: give clothing_fbx or clothings=[...]")
        clothings = [ClothingSpec(
            clothing_fbx=Path(clothing_fbx),
            user_drop_bone_ids=user_drop_bone_ids,
            drop_mesh_ids=drop_mesh_ids,
            drop_blend_shape_channel_ids=drop_blend_shape_channel_ids,
            blend_shape_channel_overrides=blend_shape_channel_overrides,
            bone_transforms=bone_transforms,
        )]
    elif not clothings:
        raise PipelineError("assemble: clothings=[] — nothing to merge")
    elif clothing_fbx is not None:
        raise PipelineError("assemble: give clothing_fbx OR clothings, not both")

    out_fbx = Path(out_fbx).resolve()
    work_dir = (Path(work_dir).resolve() if work_dir else out_fbx.parent / "_rigforge_work")
    work_dir.mkdir(parents=True, exist_ok=True)
    # Shared bin→ASCII cache. The API passes the same directory to the inspect
    # endpoint so the clothing converts once and the FE's part ids match the
    # pipeline's view. Default keeps it beside the output for the CLI path.
    ascii_cache_dir = (
        Path(ascii_cache_dir).resolve() if ascii_cache_dir
        else out_fbx.parent / "_fbx_ascii_cache"
    )

    n = len(clothings)
    runs: list[ClothingRun] = []
    # The ASCII being merged INTO: None → the bare curated avatar (first
    # garment); afterwards the previous garment's merged output.
    merged_ascii: Optional[bytes] = None
    for i, spec in enumerate(clothings):
        cfbx = Path(spec.clothing_fbx).resolve()
        first, last = (i == 0), (i == n - 1)
        # Announce the garment whose phases follow (drives per-garment
        # progress rows on a single stream). Notes carry a [i/n stem] prefix
        # only when there is more than one garment, so single-clothing runs
        # read exactly as before.
        _emit("clothing", cfbx.stem)
        label = f"[{i + 1}/{n} {cfbx.stem}] " if n > 1 else ""

        _emit("phase_a", f"{label}identifying donor for {cfbx.name}")
        # --- Phase A
        a = run_phase_a(
            cfbx, registry, work_dir=work_dir, ascii_cache_dir=ascii_cache_dir,
            target_id=target_id,
        )
        if spec.attach_to_bone_id is not None and not a.rigid:
            raise PipelineError(
                f"{cfbx.name}: attach to a bone applies only to a garment with "
                f"no bones (a rigid mesh); this one is rigged"
            )
        if a.rigid:
            # --- No bones (a rigid prop): nothing to match, nothing to classify.
            _emit("phase_a", f"{label}rigid garment (no bones): merged as-is onto {target_id}")
            b = PhaseBResult(
                decisions=DecisionSet(bones=[]), edit_plan=EditPlan(),
                warnings=[], cache_hit=False, cache_key="",
            )
            _emit("phase_b", f"{label}no bones to classify — LLM skipped")
        else:
            _emit("phase_a", f"{label}donor={a.donor_id} score={a.score:.2f}")

            _emit("phase_b", f"{label}classifying {len(a.view.limb_bones())} bones (LLM may take minutes)")
            # --- Phase B
            b = run_phase_b(
                view=a.view,
                donor_id=a.donor_id,
                target_id=target_id,
                registry=registry,
                schema=schema,
                llm_client=llm_client,
                cache=cache,
                user_drop_bone_ids=spec.user_drop_bone_ids,
            )
            _emit(
                "phase_b",
                f"{label}{len(b.edit_plan.drops)} drops + {len(b.edit_plan.renames)} renames "
                f"({'cache hit' if b.cache_hit else 'fresh LLM run'})",
            )

        _emit(
            "phase_c",
            f"{label}applying edits + merging armature"
            + ("" if first else " onto the previous garment's result"),
        )
        # --- Phase C: merge onto the bare target (first) or the running result.
        #     Target-side edits act on the base avatar's content → first merge
        #     only (re-applying a bone move would double it; the ids they name
        #     are the target's, which survive every merge unchanged). The
        #     zero-weight sweep runs once, on the final merge, so an earlier
        #     garment's weightless target bone is still there for a later one.
        clothing_ascii = a.ascii_path.read_bytes()
        c = run_phase_c(
            clothing_ascii=clothing_ascii,
            clothing_view=a.view,
            donor_id=a.donor_id,
            target_id=target_id,
            registry=registry,
            edit_plan=b.edit_plan,
            decisions=b.decisions,
            target_drop_bone_ids=target_drop_bone_ids if first else None,
            target_drop_mesh_ids=target_drop_mesh_ids if first else None,
            drop_mesh_ids=spec.drop_mesh_ids,
            drop_blend_shape_channel_ids=spec.drop_blend_shape_channel_ids,
            target_drop_blend_shape_channel_ids=(
                target_drop_blend_shape_channel_ids if first else None
            ),
            blend_shape_channel_overrides=spec.blend_shape_channel_overrides,
            target_blend_shape_channel_overrides=(
                target_blend_shape_channel_overrides if first else None
            ),
            translate_target_morphs=translate_target_morphs if first else False,
            bone_transforms=spec.bone_transforms,
            target_bone_transforms=target_bone_transforms if first else None,
            attach_to_bone_id=spec.attach_to_bone_id,
            target_ascii=merged_ascii,
            zero_weight_sweep=last,
        )
        merged_ascii = c.merged_ascii
        _emit("phase_c", f"{label}merge complete")
        runs.append(ClothingRun(
            clothing_fbx=cfbx,
            donor_id=a.donor_id,
            score=a.score,
            candidates=a.candidates,
            cache_hit=b.cache_hit,
            cache_key=b.cache_key,
            edit_plan=b.edit_plan,
            warnings=b.warnings,
            phase_a=a,
            phase_b=b,
            phase_c=c,
            notes=list(c.notes),
            bone_transforms=list(spec.bone_transforms or []),
        ))

    assert merged_ascii is not None  # n >= 1
    _emit("write", "converting ASCII → binary FBX")
    # --- Write merged ASCII + convert to binary (once, for the whole chain)
    merged_ascii_path = work_dir / (out_fbx.stem + "_merged_ascii.fbx")
    merged_ascii_path.write_bytes(merged_ascii)
    ascii_to_bin(merged_ascii_path, out_fbx)
    _emit("write", f"wrote {out_fbx.name}")

    head = runs[0]
    if n == 1:
        notes = list(head.notes)
    else:
        notes = [f"[{r.stem}] {note}" for r in runs for note in r.notes]
    return PipelineRun(
        output_fbx=out_fbx,
        donor_id=head.donor_id,
        target_id=target_id,
        score=head.score,
        candidates=head.candidates,
        cache_hit=head.cache_hit,
        cache_key=head.cache_key,
        edit_plan=head.edit_plan,
        warnings=[w for r in runs for w in r.warnings],
        phase_a=head.phase_a,
        phase_b=head.phase_b,
        phase_c=runs[-1].phase_c,
        notes=notes,
        bone_transforms=list(head.bone_transforms),
        target_bone_transforms=list(target_bone_transforms or []),
        clothing_runs=runs,
    )
