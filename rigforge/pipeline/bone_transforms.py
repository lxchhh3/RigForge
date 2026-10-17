"""Bone moves: the request-side model, CLI spec parser, and resolver.

A bone move sets a bone's REST-POSE local transform (position / rotation /
scale in its parent's frame) without touching the skin bind pose, so the mesh
weighted to it follows rigidly — the same thing as dragging the bone in
Unity's editor. It is a pure FBX property edit (no Blender involved); see
`ascii_fbx.edits.set_model_local_transform_edits` for the byte-level write and
`ascii_fbx.transform` for the Unity <-> FBX frame math.

Three surfaces feed this:
  - FE / API:  `{bone_id, position?, rotation?, scale?, frame}` per entry
  - CLI:       `--bone-transform "NAME;pos=x,y,z;rot=x,y,z;scale=x,y,z"`
               (`#<id>` instead of NAME targets a Model id)
  - Python:    `BoneTransformEdit(...)` directly

Phase C resolves each entry against the relevant SectionView (clothing or
target) into `{model_id: LocalTransform}` in the FBX frame and applies it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Union

from rigforge.ascii_fbx.sections import SectionView
from rigforge.ascii_fbx.transform import FRAMES, LocalTransform, convert_local


class BoneTransformError(ValueError):
    """Malformed spec, or a bone reference that cannot be resolved safely."""


@dataclass(frozen=True)
class BoneTransformEdit:
    """One requested bone move. `bone` is a Model id (int) or a display name
    (str, must be unique in the file it is resolved against). `frame` says
    which convention `transform` is expressed in: "unity" (what Unity's
    Inspector shows — the FE/CLI default) or "fbx" (raw Lcl values)."""
    bone: Union[int, str]
    transform: LocalTransform = field(default_factory=LocalTransform)
    frame: str = "unity"

    def __post_init__(self) -> None:
        if self.frame not in FRAMES:
            raise BoneTransformError(
                f"unknown transform frame {self.frame!r}; expected one of {FRAMES}"
            )

    def to_dict(self) -> dict:
        return {"bone": self.bone, "frame": self.frame, **self.transform.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "BoneTransformEdit":
        return cls(
            bone=d["bone"],
            transform=LocalTransform.from_dict(d),
            frame=d.get("frame", "unity"),
        )


# --- CLI spec --------------------------------------------------------------------

_SPEC_KEYS = {"pos": "position", "rot": "rotation", "scale": "scale"}


def parse_bone_transform_spec(spec: str, *, frame: str = "unity") -> BoneTransformEdit:
    """Parse `NAME;pos=x,y,z;rot=x,y,z;scale=x,y,z` (any subset of the three
    components, at least one). `#123` as the name targets Model id 123.
    Whitespace around separators is ignored."""
    if frame not in FRAMES:
        raise BoneTransformError(
            f"unknown transform frame {frame!r}; expected one of {FRAMES}"
        )
    parts = [p.strip() for p in spec.split(";")]
    if not parts or not parts[0]:
        raise BoneTransformError(f"bone-transform spec {spec!r}: missing bone name")
    bone_text = parts[0]
    bone: Union[int, str]
    if bone_text.startswith("#"):
        try:
            bone = int(bone_text[1:])
        except ValueError as exc:
            raise BoneTransformError(
                f"bone-transform spec {spec!r}: {bone_text!r} is not a Model id"
            ) from exc
    else:
        bone = bone_text

    fields: dict[str, tuple[float, float, float]] = {}
    for part in parts[1:]:
        if not part:
            continue
        if "=" not in part:
            raise BoneTransformError(
                f"bone-transform spec {spec!r}: expected key=x,y,z, got {part!r}"
            )
        key, _, value = part.partition("=")
        key = key.strip().lower()
        if key not in _SPEC_KEYS:
            raise BoneTransformError(
                f"bone-transform spec {spec!r}: unknown key {key!r} "
                f"(expected one of {sorted(_SPEC_KEYS)})"
            )
        if _SPEC_KEYS[key] in fields:
            raise BoneTransformError(f"bone-transform spec {spec!r}: duplicate key {key!r}")
        nums = [v.strip() for v in value.split(",")]
        if len(nums) != 3:
            raise BoneTransformError(
                f"bone-transform spec {spec!r}: {key} needs 3 comma-separated numbers"
            )
        try:
            fields[_SPEC_KEYS[key]] = (float(nums[0]), float(nums[1]), float(nums[2]))
        except ValueError as exc:
            raise BoneTransformError(
                f"bone-transform spec {spec!r}: {key} values are not numbers"
            ) from exc
    if not fields:
        raise BoneTransformError(
            f"bone-transform spec {spec!r}: give at least one of pos=, rot=, scale="
        )
    return BoneTransformEdit(bone=bone, transform=LocalTransform(**fields), frame=frame)


# --- resolution ---------------------------------------------------------------------


def _merge(prev: Optional[LocalTransform], new: LocalTransform) -> LocalTransform:
    if prev is None:
        return new
    return LocalTransform(
        position=new.position if new.position is not None else prev.position,
        rotation=new.rotation if new.rotation is not None else prev.rotation,
        scale=new.scale if new.scale is not None else prev.scale,
    )


def resolve_bone_transform_edits(
    view: SectionView,
    edits: Iterable[BoneTransformEdit],
    *,
    notes: list[str],
    side: str,
) -> dict[int, LocalTransform]:
    """Turn requested moves into `{model_id: LocalTransform}` in the FBX
    frame against `view`.

    - An id or name that is not in the view is skipped with a
      `skip: bone_transform ...` note (a stale FE id must not kill the run).
    - A name matching more than one Model raises: guessing which duplicate
      the modder meant would silently move the wrong part.
    - Several entries for one bone merge component-wise; a later entry wins.
    - Entries with nothing to set are dropped.
    """
    by_name: dict[str, list[int]] = {}
    for b in view.bones.values():
        by_name.setdefault(b.name, []).append(b.model_id)

    out: dict[int, LocalTransform] = {}
    for e in edits:
        if e.transform.is_empty():
            continue
        if isinstance(e.bone, int):
            if e.bone not in view.bones:
                notes.append(
                    f"skip: bone_transform ({side}) id={e.bone} not in the {side} file"
                )
                continue
            mid = e.bone
        else:
            ids = by_name.get(e.bone, [])
            if not ids:
                notes.append(
                    f"skip: bone_transform ({side}) no bone named {e.bone!r} in the {side} file"
                )
                continue
            if len(ids) > 1:
                raise BoneTransformError(
                    f"bone name {e.bone!r} is ambiguous in the {side} file "
                    f"({len(ids)} matches: ids {ids}); target it by id instead"
                )
            mid = ids[0]
        out[mid] = _merge(out.get(mid), convert_local(e.transform, e.frame))
    return out
