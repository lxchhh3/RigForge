"""FastAPI app factory + endpoints.

Read-only manifest browser (slice 1):
    GET  /api/assemblies            list of recent run manifests
    GET  /api/assemblies/{id}       a single manifest by filename stem

Compose flow (slice 2):
    GET  /api/avatars               list curated target avatars
    POST /api/clothings/inspect     parse an ASCII FBX → return its bone tree

The bone tree powers the FE's part-toggle UI: every bone shows up with its
parent_id and subtree_size so the FE can render a collapsible tree and let
the user uncheck named secondary chains.
"""
from __future__ import annotations

import base64
import json
import queue
import re
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rigforge.ascii_fbx.convert import bin_to_ascii_cached
from rigforge.ascii_fbx.lexer import parse as parse_fbx
from rigforge.ascii_fbx.sections import (
    SectionView,
    bone_to_mesh_names,
    extract as extract_sections,
    model_local_transform,
)
from rigforge.ascii_fbx.transform import LocalTransform, fbx_to_unity
from rigforge.pipeline.bone_transforms import BoneTransformEdit, BoneTransformError
from rigforge.avatars.registry import AvatarRegistry, RegistryError
from rigforge.canonical.schema import CanonicalSchema
from rigforge.llm.client import LLMClient
from rigforge.manifest import build_manifest
from rigforge.pipeline.orchestrator import ClothingSpec


_MANIFEST_SUFFIX = ".manifest.json"
# Filename stems are user-derived but constrained: alnum, dash, underscore.
# Anything else (slashes, dots, percent escapes) is rejected before disk I/O.
_ID_PATTERN = re.compile(r"^[A-Za-z0-9_\-]+$")


class InspectRequest(BaseModel):
    path: str
    # The compose's target avatar. A rigid garment's move reads in this
    # avatar's Armature frame, so its top-level meshes report their placement
    # there (the Move panel's prefill). None = file values (no target yet).
    target_id: Optional[str] = None


class TasteEditRequest(BaseModel):
    """A vertex-count-safe geometry pre-step on a clothing (e.g. cut sole
    height). FE currently exposes only `bisect_cut`."""
    clothing_path: str
    mesh: str
    op: str = "bisect_cut"
    cut_pct: float = 0.15


class ChannelOverride(BaseModel):
    """A user-set DeformPercent for one BlendShapeChannel. The slider in the
    FE emits a list of these per source (clothing + target)."""
    channel_id: int
    deform_percent: float  # nominally 0-100, FBX accepts any float


class BoneTransform(BaseModel):
    """One bone move: set a bone's rest-pose local transform (its parent's
    frame). The mesh weighted to the bone follows — same as dragging the bone
    in Unity. Numbers are in `frame`: "unity" = what Unity's Inspector shows
    (the FE default), "fbx" = raw Lcl values. A null component is untouched;
    at least one must be given."""
    bone_id: int
    position: Optional[list[float]] = None
    rotation: Optional[list[float]] = None   # Euler degrees
    scale: Optional[list[float]] = None
    frame: str = "unity"


def _bone_transform_edits(
    items: list[BoneTransform], *, field_name: str,
) -> Optional[list[BoneTransformEdit]]:
    """Validate + convert the request's bone moves. Arity / frame / empty
    problems are the caller's fault → 422 with the offending bone named."""
    if not items:
        return None
    out: list[BoneTransformEdit] = []
    for it in items:
        try:
            t = LocalTransform.from_dict(
                {"position": it.position, "rotation": it.rotation, "scale": it.scale}
            )
            if t.is_empty():
                raise ValueError("nothing to set: give position, rotation, or scale")
            out.append(BoneTransformEdit(bone=it.bone_id, transform=t, frame=it.frame))
        except (ValueError, BoneTransformError) as e:
            raise HTTPException(
                status_code=422, detail=f"{field_name}: bone {it.bone_id}: {e}",
            )
    return out


class ClothingSpecRequest(BaseModel):
    """One garment of a compose run with its own selections (the FE's
    per-clothing outliner state). Same field names as the flat single-
    clothing form of AssembleRequest, minus the target_* ones."""
    clothing_path: str
    drop_bone_ids: list[int] = []
    drop_mesh_ids: list[int] = []
    drop_blend_shape_channel_ids: list[int] = []
    blend_shape_channel_overrides: list[ChannelOverride] = []
    bone_transforms: list[BoneTransform] = []
    # Rigid garment (no bones) only: the TARGET bone id its meshes hang from
    # so they follow it. None = the scene root (follows no bone).
    attach_to_bone_id: Optional[int] = None


class AssembleRequest(BaseModel):
    target_id: str
    # ONE garment (legacy flat form: the per-clothing fields below apply to
    # it) — or SEVERAL via `clothings`, each carrying its own selections.
    # Several garments become ONE merged output (garment 1 onto the target,
    # garment 2 onto that result, ...), never one file per garment.
    clothing_path: Optional[str] = None
    clothings: list[ClothingSpecRequest] = []
    drop_bone_ids: list[int] = []
    target_drop_bone_ids: list[int] = []
    # Mesh-id drops — what the FE outliner actually sends. Mesh-Model id,
    # not bone id. Pipeline drops Mesh-Model + Geometry + Skin + Clusters
    # as a unit.
    drop_mesh_ids: list[int] = []
    target_drop_mesh_ids: list[int] = []
    # BlendShapeChannel-id drops. SubDeformer ids of channels (morphs) that
    # the modder unchecked. Owning BlendShape Deformer is preserved (it may
    # carry surviving siblings).
    drop_blend_shape_channel_ids: list[int] = []
    target_drop_blend_shape_channel_ids: list[int] = []
    # DeformPercent overrides — the slider sends one entry per channel the
    # user moved off its on-disk value. Channels in the corresponding drop
    # set are ignored (the node is gone before override would apply).
    blend_shape_channel_overrides: list[ChannelOverride] = []
    target_blend_shape_channel_overrides: list[ChannelOverride] = []
    # Translate the base avatar's own morph + mesh names to English. Clothing
    # names are always translated; this checkbox controls the TARGET only, for
    # the rare case a downstream Unity project binds the base avatar's morphs
    # by name. Defaults on.
    translate_target_morphs: bool = True
    # Bone moves — one entry per bone the modder repositioned in the FE's
    # "Move bones" panel, on the clothing and on the target respectively.
    bone_transforms: list[BoneTransform] = []
    target_bone_transforms: list[BoneTransform] = []


def _build_inspect_response(
    view: SectionView, donor_id: Optional[str], donor_score: float,
    *,
    local_overrides: Optional[dict[int, LocalTransform]] = None,
    rigid_frame: Optional[str] = None,
) -> dict:
    """Bone-tree response shared by /api/clothings/inspect and
    /api/avatars/{id}/inspect — same shape so the FE can reuse BoneTree.vue
    and the same TS interface for both clothing donors and target avatars.
    `local_overrides` {model id: FBX-frame transform} replaces a Model's
    file values (a rigid mesh's placement in the target Armature frame);
    `rigid_frame` names that frame for the FE."""
    local_overrides = local_overrides or {}
    children_of: dict[int, list[int]] = {}
    for b in view.bones.values():
        if b.parent_id is not None:
            children_of.setdefault(b.parent_id, []).append(b.model_id)

    subtree_size_cache: dict[int, int] = {}

    def _subtree_size(bid: int, seen: set[int]) -> int:
        if bid in subtree_size_cache:
            return subtree_size_cache[bid]
        if bid in seen:
            return 0
        seen.add(bid)
        n = 1
        for c in children_of.get(bid, []):
            n += _subtree_size(c, seen)
        subtree_size_cache[bid] = n
        return n

    bone_ids = set(view.bones)
    bones_out = [
        {
            "model_id": b.model_id,
            "name": b.name,
            "type_class": b.type_class,
            "parent_id": b.parent_id if b.parent_id in bone_ids else None,
            "subtree_size": _subtree_size(b.model_id, set()),
            "cluster_weight_count": b.cluster_weight_count,
            "deforms_meshes": bone_to_mesh_names(view, b.model_id),
            # Rest-pose local transform as Unity's Inspector will show it, so
            # the FE's "Move bones" panel opens on the file's current values.
            "local_transform_unity": fbx_to_unity(
                local_overrides.get(b.model_id) or model_local_transform(view, b.model_id)
            ).to_dict(),
        }
        for b in view.bones.values()
    ]
    # BlendShape channels — flat list sorted by name. The FE renders each as
    # its own checkbox + DeformPercent slider; owner_name surfaces which
    # BlendShape Deformer the channel belongs to (informational, not used as
    # a group key).
    bs_owners = view.blend_shapes
    source = view.document.source

    # Channel -> owning MESH name. Blendshape channel names are routinely
    # duplicated across meshes (e.g. several "High heeled" / "Big breasts"),
    # so the mesh is the only thing that disambiguates them in the FE. Walk
    # BlendShape -> Geometry -> Mesh-Model once.
    from rigforge.ascii_fbx.edits import iter_oo_connections
    _bs_to_geo = {
        s: d for s, d in iter_oo_connections(view)
        if s in view.blend_shapes and d in view.geometries
    }
    _mesh_name_by_model = {
        b.model_id: b.name for b in view.bones.values() if b.type_class == "Mesh"
    }

    def _owner_mesh(ch):
        geo = _bs_to_geo.get(ch.blend_shape_id)
        return _mesh_name_by_model.get(view.geometry_owner_model.get(geo))

    def _read_deform_percent(ch) -> float:
        # Read the on-disk DeformPercent so the FE slider starts at the
        # file's actual value (typically 0). Channels without the property
        # fall through to 0.
        for child in ch.node_ref.children:
            if child.name != "DeformPercent":
                continue
            args = child.args_bytes(source).strip()
            if not args:
                return 0.0
            try:
                return float(args.split()[0])
            except (ValueError, IndexError):
                return 0.0
        return 0.0

    channels_out = [
        {
            "channel_id": ch.channel_id,
            "name": ch.name,
            "owner_id": ch.blend_shape_id,
            "owner_name": (
                bs_owners[ch.blend_shape_id].name
                if ch.blend_shape_id in bs_owners else None
            ),
            # The mesh this morph deforms — disambiguates duplicate channel names.
            "owner_mesh": _owner_mesh(ch),
            "deform_percent": _read_deform_percent(ch),
        }
        # Group by owning mesh, then name, so duplicate-named channels sit under
        # their mesh instead of being interleaved and indistinguishable.
        for ch in sorted(
            view.blend_shape_channels.values(),
            key=lambda c: (_owner_mesh(c) or "", c.name, c.channel_id),
        )
    ]
    return {
        "donor_id": donor_id,
        "donor_score": donor_score,
        "total_bones": len(bones_out),
        # No bones, only meshes (a rigid prop): merged as-is; the FE offers
        # "attach to bone" for these. Same predicate as Phase A.
        "rigid": view.is_rigid(),
        # The Model whose frame a rigid garment's top-level mesh values are
        # given in (the target's Armature); None = the file's own values.
        "rigid_frame": rigid_frame,
        "bones": bones_out,
        "blend_shape_channels": channels_out,
    }


def create_app(
    *,
    data_dir: Path,
    registry: Optional[AvatarRegistry] = None,
    schema: Optional[CanonicalSchema] = None,
    llm_client: Optional[LLMClient] = None,
    output_dir: Optional[Path] = None,
    assemble_fn=None,
) -> FastAPI:
    """Build a FastAPI app that serves manifests from `data_dir`, exposes the
    curated avatar registry, and (optionally) runs assemble jobs.

    `data_dir` is read on every request — manifests can be added/removed
    without a server restart. `registry` and `schema` are loaded once at app
    build time. `llm_client` is required for the assemble endpoint; tests
    inject `assemble_fn` instead to skip the real pipeline.
    """
    if registry is None:
        registry = AvatarRegistry.load_default()
    if schema is None:
        schema = CanonicalSchema.load_default()
    if output_dir is None:
        output_dir = data_dir
    if assemble_fn is None:
        from rigforge.pipeline.orchestrator import assemble as default_assemble
        assemble_fn = default_assemble

    app = FastAPI(title="RigForge API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # --- manifest browser ---------------------------------------------------

    @app.get("/api/assemblies")
    def list_assemblies() -> list[dict]:
        if not data_dir.exists():
            return []
        out: list[dict] = []
        for path in sorted(data_dir.glob(f"*{_MANIFEST_SUFFIX}")):
            stem = path.name[: -len(_MANIFEST_SUFFIX)]
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            payload["id"] = stem
            out.append(payload)
        return out

    @app.get("/api/assemblies/{assembly_id}")
    def get_assembly(assembly_id: str) -> dict:
        if not _ID_PATTERN.match(assembly_id):
            raise HTTPException(status_code=400, detail="invalid id")
        path = data_dir / f"{assembly_id}{_MANIFEST_SUFFIX}"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="not found")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise HTTPException(status_code=500, detail=f"manifest unreadable: {e}")
        payload["id"] = assembly_id
        return payload

    # --- compose flow -------------------------------------------------------

    @app.get("/api/avatars")
    def list_avatars() -> list[dict]:
        return [
            {
                "id": a.id,
                "display_name": a.display_name,
                "canonical_roles": sorted(a.canonical_to_name.keys()),
            }
            for a in registry.avatars.values()
        ]

    @app.post("/api/clothings/inspect")
    def inspect_clothing(req: InspectRequest) -> dict:
        # Local desktop tool: no path-traversal guard. The modder is running
        # this against their own filesystem.
        p = Path(req.path)
        if not p.is_file():
            raise HTTPException(status_code=404, detail=f"file not found: {req.path}")
        try:
            # Convert through the SHARED bin→ASCII cache. The assemble pipeline
            # (Phase A) reads the same cache dir, so a binary clothing converts
            # exactly once and the ids the FE reads here are the same ids the
            # pipeline drops against. Without this, two independent conversions
            # mint different (pointer-derived) ids and every clothing-side drop
            # the FE sends back is a silent no-op. An ASCII input is returned
            # as-is by the helper.
            ascii_path = bin_to_ascii_cached(p, output_dir / "_fbx_ascii_cache")
            raw = ascii_path.read_bytes()
            doc = parse_fbx(raw)
            view = extract_sections(doc)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"parse failed: {e}")

        # Phase A donor identification (best-effort; surface even on miss
        # so the FE can still render the bone tree for an unknown donor).
        donor_id: Optional[str] = None
        donor_score: float = 0.0
        try:
            from rigforge.avatars.registry import identify_donor
            bone_names = [b.name for b in view.bones.values() if b.name]
            donor_id, donor_score, _ = identify_donor(bone_names, registry)
        except Exception:
            pass

        # A rigid garment's move reads in the target's Armature frame, so its
        # top-level meshes open the Move panel on their placement there.
        local_overrides: dict[int, LocalTransform] = {}
        rigid_frame: Optional[str] = None
        if req.target_id is not None:
            try:
                target_view = registry.get(req.target_id).load_ascii_view()
            except RegistryError as e:
                raise HTTPException(status_code=400, detail=str(e))
            if view.is_rigid():
                from rigforge.pipeline.phase_c import (
                    PhaseCError, rigid_local_in_frame, rigid_move_frame,
                )
                try:
                    frame_id, frame_world = rigid_move_frame(target_view)
                    if frame_id is not None:
                        rigid_frame = target_view.bones[frame_id].name
                        for m in view.bones_of_type("Mesh"):
                            if m.parent_id in (None, 0):
                                local_overrides[m.model_id] = rigid_local_in_frame(
                                    view, m.model_id, frame_world,
                                )
                except PhaseCError as e:
                    raise HTTPException(status_code=400, detail=str(e))

        return _build_inspect_response(
            view, donor_id, donor_score,
            local_overrides=local_overrides, rigid_frame=rigid_frame,
        )

    @app.get("/api/avatars/{avatar_id}/inspect")
    def inspect_avatar(avatar_id: str) -> dict:
        """Return the bone tree of a curated target avatar — same shape as
        /api/clothings/inspect so the FE renders it via BoneTree.vue.
        Modder uses this to strip the target's bundled clothing bones
        before splicing new clothing in."""
        try:
            av = registry.get(avatar_id)
        except RegistryError:
            raise HTTPException(status_code=404, detail=f"unknown avatar: {avatar_id}")
        try:
            view = av.load_ascii_view()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"avatar load failed: {e}")
        # donor_score=1.0 — by definition, a curated avatar perfectly matches
        # itself.
        return _build_inspect_response(view, donor_id=avatar_id, donor_score=1.0)

    @app.post("/api/clothings/taste-edit")
    def taste_edit_clothing(req: TasteEditRequest) -> dict:
        """Run a geometry pre-step (bisect_cut) on a clothing and return the
        edited FBX path + a before/after preview (base64 PNGs) + the weight-
        integrity check. The FE then assembles using `edited_path`."""
        p = Path(req.clothing_path)
        if not p.is_file():
            raise HTTPException(status_code=404, detail=f"file not found: {req.clothing_path}")
        if req.op != "bisect_cut":
            raise HTTPException(
                status_code=400,
                detail=f"unsupported op '{req.op}' (FE supports bisect_cut)",
            )
        from rigforge.geometry import blender_edit

        work = output_dir / "_taste_edit"
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", req.mesh) or "mesh"
        out = work / f"{p.stem}__{safe}_cut.fbx"
        preview = work / f"{p.stem}__{safe}_preview"
        try:
            rep = blender_edit.bisect_cut(
                p, out, mesh=req.mesh, axis="z", cut_pct=req.cut_pct, preview_dir=preview,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"taste-edit failed: {e}")

        ed = rep["edit"]
        new_unweighted = ed["unweighted_verts_after"] - ed["unweighted_verts_before"]

        def _b64(name: str) -> Optional[str]:
            f = preview / name
            if f.is_file():
                return "data:image/png;base64," + base64.b64encode(f.read_bytes()).decode("ascii")
            return None

        return {
            "edited_path": str(out),
            "mesh": ed["mesh"],
            "verts_before": ed["verts_before"],
            "verts_after": ed["verts_after"],
            "verts_removed": ed["verts_removed"],
            # The integrity gate: a cut that strips skin weights off the new
            # cut edge is NOT ok (the sole would stop following the foot).
            "weights_ok": new_unweighted <= 0,
            "preview_before": _b64("before_side.png"),
            "preview_after": _b64("after_side.png"),
        }

    def _clothing_spec(c: ClothingSpecRequest, idx: int) -> ClothingSpec:
        overrides = {o.channel_id: o.deform_percent for o in c.blend_shape_channel_overrides}
        return ClothingSpec(
            clothing_fbx=Path(c.clothing_path),
            user_drop_bone_ids=set(c.drop_bone_ids) or None,
            drop_mesh_ids=set(c.drop_mesh_ids) or None,
            drop_blend_shape_channel_ids=set(c.drop_blend_shape_channel_ids) or None,
            blend_shape_channel_overrides=overrides or None,
            bone_transforms=_bone_transform_edits(
                c.bone_transforms, field_name=f"clothings[{idx}].bone_transforms",
            ) or None,
            attach_to_bone_id=c.attach_to_bone_id,
        )

    def _validate_assemble_request(
        req: AssembleRequest,
    ) -> tuple[Optional[Path], Path, str, dict]:
        """Shared preflight for both /api/assemble and /api/assemble/stream.

        Returns (clothing_path, out_fbx, stem, kwargs). For the legacy single
        form `clothing_path` is the garment and kwargs carries the flat
        per-clothing selections; for the `clothings` form `clothing_path` is
        None and kwargs carries `clothings=[ClothingSpec, ...]`. The output
        stem joins every garment's stem with '+'.
        Raises HTTPException on validation failure — the streaming endpoint
        translates that into a single error event before the stream closes.
        """
        try:
            registry.get(req.target_id)
        except RegistryError as e:
            raise HTTPException(status_code=400, detail=str(e))

        if req.clothings:
            if req.clothing_path is not None:
                raise HTTPException(
                    status_code=422,
                    detail="give clothing_path OR clothings, not both",
                )
            raw_paths = [c.clothing_path for c in req.clothings]
        elif req.clothing_path is not None:
            raw_paths = [req.clothing_path]
        else:
            raise HTTPException(
                status_code=422,
                detail="no clothing given: send clothing_path or a non-empty clothings list",
            )
        paths = [Path(p) for p in raw_paths]
        for raw, p in zip(raw_paths, paths):
            if not p.is_file():
                raise HTTPException(status_code=404, detail=f"clothing not found: {raw}")

        stem = "+".join(p.stem for p in paths)
        out_fbx = output_dir / f"{stem}__{req.target_id}.fbx"
        clothing_path: Optional[Path] = None if req.clothings else paths[0]

        drop_ids: Optional[set[int]] = set(req.drop_bone_ids) if req.drop_bone_ids else None
        target_drop_ids: Optional[set[int]] = (
            set(req.target_drop_bone_ids) if req.target_drop_bone_ids else None
        )
        drop_mesh: Optional[set[int]] = (
            set(req.drop_mesh_ids) if req.drop_mesh_ids else None
        )
        target_drop_mesh: Optional[set[int]] = (
            set(req.target_drop_mesh_ids) if req.target_drop_mesh_ids else None
        )
        drop_channels: Optional[set[int]] = (
            set(req.drop_blend_shape_channel_ids)
            if req.drop_blend_shape_channel_ids else None
        )
        target_drop_channels: Optional[set[int]] = (
            set(req.target_drop_blend_shape_channel_ids)
            if req.target_drop_blend_shape_channel_ids else None
        )
        channel_overrides: Optional[dict[int, float]] = (
            {o.channel_id: o.deform_percent for o in req.blend_shape_channel_overrides}
            if req.blend_shape_channel_overrides else None
        )
        target_channel_overrides: Optional[dict[int, float]] = (
            {o.channel_id: o.deform_percent for o in req.target_blend_shape_channel_overrides}
            if req.target_blend_shape_channel_overrides else None
        )
        bone_transforms = _bone_transform_edits(
            req.bone_transforms, field_name="bone_transforms",
        )
        target_bone_transforms = _bone_transform_edits(
            req.target_bone_transforms, field_name="target_bone_transforms",
        )

        kwargs = {
            # Same cache dir the inspect endpoint uses — keeps the FE's part
            # ids aligned with Phase A's view (one shared bin→ASCII conversion).
            "ascii_cache_dir": output_dir / "_fbx_ascii_cache",
            "target_drop_bone_ids": target_drop_ids,
            "target_drop_mesh_ids": target_drop_mesh,
            "target_drop_blend_shape_channel_ids": target_drop_channels,
            "target_blend_shape_channel_overrides": target_channel_overrides,
            "translate_target_morphs": req.translate_target_morphs,
            "target_bone_transforms": target_bone_transforms,
        }
        if req.clothings:
            kwargs["clothings"] = [
                _clothing_spec(c, i) for i, c in enumerate(req.clothings)
            ]
        else:
            kwargs.update({
                "user_drop_bone_ids": drop_ids,
                "drop_mesh_ids": drop_mesh,
                "drop_blend_shape_channel_ids": drop_channels,
                "blend_shape_channel_overrides": channel_overrides,
                "bone_transforms": bone_transforms,
            })
        return clothing_path, out_fbx, stem, kwargs

    def _write_manifest_and_response(run, stem: str, out_fbx: Path) -> dict:
        manifest = build_manifest(run)
        manifest_path = data_dir / f"{stem}.manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifest["id"] = stem
        return {
            "id": stem,
            "output_fbx": str(out_fbx),
            "manifest": manifest,
        }

    @app.post("/api/assemble")
    def assemble_clothing(req: AssembleRequest) -> dict:
        clothing_path, out_fbx, stem, kwargs = _validate_assemble_request(req)
        if clothing_path is not None:
            kwargs["clothing_fbx"] = clothing_path
        try:
            run = assemble_fn(
                target_id=req.target_id,
                out_fbx=out_fbx,
                registry=registry,
                schema=schema,
                llm_client=llm_client,
                **kwargs,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"assemble failed: {e}")
        return _write_manifest_and_response(run, stem, out_fbx)

    @app.post("/api/assemble/stream")
    def assemble_clothing_stream(req: AssembleRequest):
        """Streaming variant of /api/assemble. Emits one JSON object per line:

          {"event": "progress", "phase": "phase_a|phase_b|phase_c|write", "note": "..."}
          {"event": "done", "id": "...", "output_fbx": "...", "manifest": {...}}
          {"event": "error", "error": "..."}

        Pre-flight validation errors still raise HTTPException so the client
        gets a normal 4xx with a Content-Type of application/json — only the
        2xx "we started running it" case streams.
        """
        clothing_path, out_fbx, stem, kwargs = _validate_assemble_request(req)
        if clothing_path is not None:
            kwargs["clothing_fbx"] = clothing_path
        clothing_paths = (
            [c.clothing_path for c in req.clothings] if req.clothings
            else [req.clothing_path]
        )

        events: queue.Queue = queue.Queue()
        # Sentinel to terminate the generator.
        _DONE = object()
        # Index of the garment whose phases are streaming. The orchestrator
        # announces each garment with a 'clothing' callback (in merge order);
        # every later progress event is stamped with clothing_index so the FE
        # drives one progress row per garment from this single stream.
        current = {"index": -1}

        def on_progress(phase: str, note: str) -> None:
            if phase == "clothing":
                current["index"] += 1
                i = current["index"]
                events.put({
                    "event": "clothing", "index": i, "stem": note,
                    "path": clothing_paths[i] if i < len(clothing_paths) else None,
                })
                return
            events.put({
                "event": "progress", "phase": phase, "note": note,
                "clothing_index": max(current["index"], 0),
            })

        def worker() -> None:
            try:
                run = assemble_fn(
                    target_id=req.target_id,
                    out_fbx=out_fbx,
                    registry=registry,
                    schema=schema,
                    llm_client=llm_client,
                    progress_cb=on_progress,
                    **kwargs,
                )
                body = _write_manifest_and_response(run, stem, out_fbx)
                events.put({"event": "done", **body})
            except Exception as e:
                events.put({"event": "error", "error": f"{type(e).__name__}: {e}"})
            finally:
                events.put(_DONE)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        def gen():
            # Initial "started" event so the FE can show "running…" immediately;
            # the worker thread might be slow to emit its first real progress
            # update (Phase A reads the FBX, converts binary→ASCII).
            yield json.dumps({"event": "started", "stem": stem}) + "\n"
            # Heartbeat interval: long Phase B (Ollama) runs can be silent for
            # minutes. Re-yielding the latest note every few seconds keeps the
            # connection alive through proxies + reassures the FE.
            last_progress: Optional[dict] = None
            last_heartbeat = time.monotonic()
            HEARTBEAT_S = 5.0
            while True:
                try:
                    item = events.get(timeout=1.0)
                except queue.Empty:
                    now = time.monotonic()
                    if last_progress and now - last_heartbeat >= HEARTBEAT_S:
                        yield json.dumps({"event": "heartbeat", **last_progress}) + "\n"
                        last_heartbeat = now
                    continue
                if item is _DONE:
                    break
                if item.get("event") == "progress":
                    last_progress = item
                    last_heartbeat = time.monotonic()
                yield json.dumps(item) + "\n"

        return StreamingResponse(
            gen(),
            media_type="application/x-ndjson",
            headers={
                # CORS: StreamingResponse doesn't auto-inherit the middleware's
                # ACAO when the response is built lazily; the middleware adds
                # headers post-hoc so this is fine — but X-Accel-Buffering off
                # so nginx (if any) doesn't hold the whole stream.
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
            },
        )

    return app
