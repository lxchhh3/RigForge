"""Orientation monitor — detects the "clothing rotated 90 degrees" class of bug.

What it catches
---------------
A mesh-Model **rotation convention mismatch** between merged sources. The 6/16
case: the clothing FBX bakes `Lcl Rotation (-90, 0, 0)` onto every mesh Model
(the Unity/Blender export convention — the Y-up/Z-up reconciliation), while the
maya avatar's meshes are identity and orient purely through skin + armature.
After a cross-avatar merge the clothing meshes keep their -90 Model rotation but
are skinned onto maya's bones, so the clothing sits 90 off. Both sides parse and
splice cleanly, so nothing in the pipeline fails — only the eye catches it.

Two checks:
  - `find_mixed_orientation(view)`  — on ONE FBX (e.g. the merged output): flags
    when meshes don't all share one rotation convention.
  - `compare_conventions(clothing_view, target_view)` — pre-merge: flags when the
    clothing's dominant mesh-Model rotation differs from the target's.

CLI:
    python -m rigforge.diagnostics.orientation MERGED.fbx           # mixed?
    python -m rigforge.diagnostics.orientation CLOTHING.fbx MAYA.fbx  # compare
Exits 1 when a mismatch is found, 0 when clean (so it can gate a run).
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rigforge.ascii_fbx.lexer import parse as parse_fbx
from rigforge.ascii_fbx.sections import SectionView, _extract_props_vec3, extract

# The component snap-step (degrees) used to group near-equal rotations. FBX
# exporters emit values like -90.0000093; 1-degree buckets absorb that noise.
_DEFAULT_TOL = 1.0

Rotation = tuple[float, float, float]


def _round_sig(vec: Rotation, tol: float) -> Rotation:
    """Snap each component to a multiple of `tol` so float noise groups together."""
    return tuple(round(v / tol) * tol for v in vec)  # type: ignore[return-value]


def mesh_orientations(view: SectionView, tol: float = _DEFAULT_TOL) -> dict[int, tuple[str, Rotation]]:
    """model_id -> (mesh_name, snapped Lcl Rotation). Identity meshes -> (0,0,0)."""
    src = view.document.source
    out: dict[int, tuple[str, Rotation]] = {}
    for b in view.bones.values():
        if b.type_class != "Mesh":
            continue
        rot = _extract_props_vec3(b.node_ref, src, "Lcl Rotation", (0.0, 0.0, 0.0))
        out[b.model_id] = (b.name, _round_sig(rot, tol))
    return out


def rotation_groups(view: SectionView, tol: float = _DEFAULT_TOL) -> dict[Rotation, list[str]]:
    """Mesh names grouped by their snapped Model Lcl Rotation signature."""
    groups: dict[Rotation, list[str]] = defaultdict(list)
    for _, (name, sig) in mesh_orientations(view, tol).items():
        groups[sig].append(name)
    return dict(groups)


def dominant_rotation(view: SectionView, tol: float = _DEFAULT_TOL) -> Optional[Rotation]:
    """The most common mesh-Model Lcl Rotation, or None when there are no meshes."""
    sigs = [sig for _, (_, sig) in mesh_orientations(view, tol).items()]
    if not sigs:
        return None
    return Counter(sigs).most_common(1)[0][0]


@dataclass(frozen=True)
class MixedOrientation:
    groups: dict[Rotation, list[str]]

    def message(self) -> str:
        parts = [
            f"{sig} <- {len(names)} mesh(es) e.g. {names[:3]}"
            for sig, names in sorted(self.groups.items())
        ]
        return "mixed mesh-Model rotation conventions: " + " | ".join(parts)


def find_mixed_orientation(
    view: SectionView, tol: float = _DEFAULT_TOL
) -> Optional[MixedOrientation]:
    """Flag when meshes in ONE file don't share a single rotation convention —
    the signature of the 90-degree bug in a merged output. None when consistent
    (including the trivial 0- or 1-mesh cases)."""
    groups = rotation_groups(view, tol)
    if len(groups) <= 1:
        return None
    return MixedOrientation(groups)


def _is_scene_root(parent_id: Optional[int], bone_ids: set[int]) -> bool:
    """A node is at the scene root when it has no parent, points at RootNode 0,
    or points at an id that isn't itself a Model in the view."""
    return parent_id is None or parent_id == 0 or parent_id not in bone_ids


@dataclass(frozen=True)
class RootRotationSplit:
    rotated_root_nulls: list[tuple[str, Rotation]]
    bypass_roots: list[tuple[str, Rotation]]

    def message(self) -> str:
        nn = ", ".join(f"{n}{r}" for n, r in self.rotated_root_nulls)
        bb = ", ".join(f"{n}" for n, _ in self.bypass_roots[:5])
        return (
            f"root-transform split: armature Null(s) carry rotation [{nn}] but "
            f"{len(self.bypass_roots)} skeleton root(s) bypass it at the scene root "
            f"(identity) e.g. [{bb}] — that content is rotated relative to the "
            f"armature-parented content"
        )


def find_root_rotation_split(
    view: SectionView, tol: float = _DEFAULT_TOL
) -> Optional[RootRotationSplit]:
    """Flag when some skeleton roots sit under a *rotated* armature Null while
    others bypass it, parented straight to the scene root at identity.

    This is the actual 6/16 "clothing rotated 90 degrees" signature: the merge
    strips the clothing's `Armature` Null (Lcl Rotation -90) and reparents its
    root bone (`Hips`) to scene RootNode 0, while the target's skeleton stays
    under the target `Armature` (-90). The bypassing subtree loses the -90 and
    renders rotated. Returns None when every skeleton root shares the same root
    transform context."""
    src = view.document.source
    bone_ids = set(view.bones)
    rotated_root_nulls: list[tuple[str, Rotation]] = []
    bypass_roots: list[tuple[str, Rotation]] = []
    for b in view.bones.values():
        if not _is_scene_root(b.parent_id, bone_ids):
            continue
        rot = _round_sig(
            _extract_props_vec3(b.node_ref, src, "Lcl Rotation", (0.0, 0.0, 0.0)), tol
        )
        if b.type_class == "Null" and rot != (0.0, 0.0, 0.0):
            rotated_root_nulls.append((b.name, rot))
        elif b.type_class == "LimbNode":
            bypass_roots.append((b.name, rot))
    if rotated_root_nulls and bypass_roots:
        return RootRotationSplit(rotated_root_nulls, bypass_roots)
    return None


def compare_conventions(
    clothing_view: SectionView, target_view: SectionView, tol: float = _DEFAULT_TOL
) -> Optional[str]:
    """Flag when the clothing's dominant mesh-Model rotation differs from the
    target's (the pre-merge form of the 90-degree mismatch). None when they
    agree, or when either side has no meshes."""
    c = dominant_rotation(clothing_view, tol)
    t = dominant_rotation(target_view, tol)
    if c is None or t is None:
        return None
    if c != t:
        delta = tuple(round(a - b, 1) for a, b in zip(c, t))
        return (
            f"mesh-Model rotation convention mismatch: clothing={c} vs target={t} "
            f"(delta={delta}) — clothing will appear rotated"
        )
    return None


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _load_view(path: Path) -> SectionView:
    """Parse an FBX into a SectionView. Binary inputs are routed through the
    shared bin->ASCII cache (same one the pipeline uses)."""
    raw = path.read_bytes()
    if raw.startswith(b"Kaydara FBX Binary"):
        from rigforge.ascii_fbx.convert import bin_to_ascii_cached
        ascii_path = bin_to_ascii_cached(path, path.parent / "_fbx_ascii_cache")
        raw = ascii_path.read_bytes()
    return extract(parse_fbx(raw))


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not (1 <= len(argv) <= 2):
        print(__doc__)
        return 2
    if len(argv) == 1:
        view = _load_view(Path(argv[0]))
        groups = rotation_groups(view)
        print(f"meshes: {sum(len(v) for v in groups.values())}; "
              f"distinct Model rotations: {len(groups)}")
        for sig, names in sorted(groups.items()):
            print(f"  {sig}: {len(names)}  e.g. {names[:4]}")
        failed = False
        mixed = find_mixed_orientation(view)
        if mixed:
            print("FAIL [mesh-rotation]:", mixed.message())
            failed = True
        split = find_root_rotation_split(view)
        if split:
            print("FAIL [armature-split]:", split.message())
            failed = True
        if not failed:
            print("OK: meshes share one rotation convention and no skeleton root "
                  "bypasses a rotated armature")
        return 1 if failed else 0
    # two-file compare
    clothing = _load_view(Path(argv[0]))
    target = _load_view(Path(argv[1]))
    warn = compare_conventions(clothing, target)
    print(f"clothing dominant rot = {dominant_rotation(clothing)}")
    print(f"target   dominant rot = {dominant_rotation(target)}")
    if warn:
        print("FAIL:", warn)
        return 1
    print("OK: clothing and target share one mesh-Model rotation convention")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
