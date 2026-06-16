"""Orchestrator — wires Phases A/B/C and the bin↔ASCII converter into a
single `assemble()` call.

End-to-end:
    bin → ASCII (converter) → Phase A → Phase B → Phase C → ASCII → bin
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from rigforge.ascii_fbx.convert import ascii_to_bin
from rigforge.avatars.registry import AvatarRegistry
from rigforge.cache.store import DecisionCache
from rigforge.canonical.schema import CanonicalSchema
from rigforge.canonical.validators import Violation
from rigforge.llm.client import LLMClient

from .edit_plan import EditPlan
from .phase_a import PhaseAResult, run_phase_a
from .phase_b import PhaseBError, PhaseBResult, run_phase_b
from .phase_c import PhaseCError, PhaseCResult, run_phase_c


class PipelineError(RuntimeError):
    pass


@dataclass
class PipelineRun:
    """Everything a manifest writer / CLI / tests want to inspect."""
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


ProgressCallback = Callable[[str, str], None]
"""(phase, note) — phase is one of 'phase_a'|'phase_b'|'phase_c'|'write';
note is a one-line human-readable description of the just-completed step.

The callback is invoked at phase boundaries from the orchestrator. It runs
inline on the worker thread; the API layer wraps it to push events onto a
streaming queue."""


def assemble(
    *,
    clothing_fbx: Path,
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
    progress_cb: Optional[ProgressCallback] = None,
) -> PipelineRun:
    """Run the full assembly pipeline and write the deliverable binary FBX.

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

    clothing_fbx = Path(clothing_fbx).resolve()
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

    _emit("phase_a", f"identifying donor for {clothing_fbx.name}")
    # --- Phase A
    a = run_phase_a(
        clothing_fbx, registry, work_dir=work_dir, ascii_cache_dir=ascii_cache_dir,
        target_id=target_id,
    )
    _emit("phase_a", f"donor={a.donor_id} score={a.score:.2f}")

    _emit("phase_b", f"classifying {len(a.view.limb_bones())} bones (LLM may take minutes)")
    # --- Phase B
    b = run_phase_b(
        view=a.view,
        donor_id=a.donor_id,
        target_id=target_id,
        registry=registry,
        schema=schema,
        llm_client=llm_client,
        cache=cache,
        user_drop_bone_ids=user_drop_bone_ids,
    )
    _emit(
        "phase_b",
        f"{len(b.edit_plan.drops)} drops + {len(b.edit_plan.renames)} renames "
        f"({'cache hit' if b.cache_hit else 'fresh LLM run'})",
    )

    _emit("phase_c", "applying edits + merging armature")
    # --- Phase C
    clothing_ascii = a.ascii_path.read_bytes()
    c = run_phase_c(
        clothing_ascii=clothing_ascii,
        clothing_view=a.view,
        donor_id=a.donor_id,
        target_id=target_id,
        registry=registry,
        edit_plan=b.edit_plan,
        decisions=b.decisions,
        target_drop_bone_ids=target_drop_bone_ids,
        target_drop_mesh_ids=target_drop_mesh_ids,
        drop_mesh_ids=drop_mesh_ids,
        drop_blend_shape_channel_ids=drop_blend_shape_channel_ids,
        target_drop_blend_shape_channel_ids=target_drop_blend_shape_channel_ids,
        blend_shape_channel_overrides=blend_shape_channel_overrides,
        target_blend_shape_channel_overrides=target_blend_shape_channel_overrides,
    )
    _emit("phase_c", "merge complete")

    _emit("write", "converting ASCII → binary FBX")
    # --- Write merged ASCII + convert to binary
    merged_ascii_path = work_dir / (out_fbx.stem + "_merged_ascii.fbx")
    merged_ascii_path.write_bytes(c.merged_ascii)
    ascii_to_bin(merged_ascii_path, out_fbx)
    _emit("write", f"wrote {out_fbx.name}")

    return PipelineRun(
        output_fbx=out_fbx,
        donor_id=a.donor_id,
        target_id=target_id,
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
    )
