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
)
from rigforge.avatars.registry import AvatarRegistry, RegistryError
from rigforge.canonical.schema import CanonicalSchema
from rigforge.llm.client import LLMClient
from rigforge.manifest import build_manifest


_MANIFEST_SUFFIX = ".manifest.json"
# Filename stems are user-derived but constrained: alnum, dash, underscore.
# Anything else (slashes, dots, percent escapes) is rejected before disk I/O.
_ID_PATTERN = re.compile(r"^[A-Za-z0-9_\-]+$")


class InspectRequest(BaseModel):
    path: str


class ChannelOverride(BaseModel):
    """A user-set DeformPercent for one BlendShapeChannel. The slider in the
    FE emits a list of these per source (clothing + target)."""
    channel_id: int
    deform_percent: float  # nominally 0-100, FBX accepts any float


class AssembleRequest(BaseModel):
    target_id: str
    clothing_path: str
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


def _build_inspect_response(
    view: SectionView, donor_id: Optional[str], donor_score: float,
) -> dict:
    """Bone-tree response shared by /api/clothings/inspect and
    /api/avatars/{id}/inspect — same shape so the FE can reuse BoneTree.vue
    and the same TS interface for both clothing donors and target avatars."""
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

        return _build_inspect_response(view, donor_id, donor_score)

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

    def _validate_assemble_request(req: AssembleRequest) -> tuple[Path, Path, str, dict]:
        """Shared preflight for both /api/assemble and /api/assemble/stream.

        Returns (clothing_path, out_fbx, stem, kwargs) where kwargs is the
        unpacked set of drop-id sets ready to pass to assemble_fn.
        Raises HTTPException on validation failure — the streaming endpoint
        translates that into a single error event before the stream closes.
        """
        try:
            registry.get(req.target_id)
        except RegistryError as e:
            raise HTTPException(status_code=400, detail=str(e))

        clothing_path = Path(req.clothing_path)
        if not clothing_path.is_file():
            raise HTTPException(
                status_code=404, detail=f"clothing not found: {req.clothing_path}",
            )

        stem = clothing_path.stem
        out_fbx = output_dir / f"{stem}__{req.target_id}.fbx"

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

        kwargs = {
            # Same cache dir the inspect endpoint uses — keeps the FE's part
            # ids aligned with Phase A's view (one shared bin→ASCII conversion).
            "ascii_cache_dir": output_dir / "_fbx_ascii_cache",
            "user_drop_bone_ids": drop_ids,
            "target_drop_bone_ids": target_drop_ids,
            "drop_mesh_ids": drop_mesh,
            "target_drop_mesh_ids": target_drop_mesh,
            "drop_blend_shape_channel_ids": drop_channels,
            "target_drop_blend_shape_channel_ids": target_drop_channels,
            "blend_shape_channel_overrides": channel_overrides,
            "target_blend_shape_channel_overrides": target_channel_overrides,
        }
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
        try:
            run = assemble_fn(
                clothing_fbx=clothing_path,
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

        events: queue.Queue = queue.Queue()
        # Sentinel to terminate the generator.
        _DONE = object()

        def on_progress(phase: str, note: str) -> None:
            events.put({"event": "progress", "phase": phase, "note": note})

        def worker() -> None:
            try:
                run = assemble_fn(
                    clothing_fbx=clothing_path,
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
