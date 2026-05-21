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
import re
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rigforge.ascii_fbx.convert import bin_to_ascii
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
# Binary FBX magic — first 21 bytes of every binary FBX (Autodesk format).
_FBX_BINARY_MAGIC = b"Kaydara FBX Binary  \x00"


class InspectRequest(BaseModel):
    path: str


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
    return {
        "donor_id": donor_id,
        "donor_score": donor_score,
        "total_bones": len(bones_out),
        "bones": bones_out,
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
            raw = p.read_bytes()
            if raw.startswith(_FBX_BINARY_MAGIC):
                # Binary FBX — round-trip through fbx_env converter to get ASCII.
                # Cached under output_dir so subsequent inspects of the same path
                # are cheap (the converter subprocess is the slow part).
                ascii_cache = output_dir / "_inspect_cache" / f"{p.stem}_ascii.fbx"
                ascii_cache.parent.mkdir(parents=True, exist_ok=True)
                if not ascii_cache.is_file() or ascii_cache.stat().st_mtime < p.stat().st_mtime:
                    bin_to_ascii(p, ascii_cache)
                raw = ascii_cache.read_bytes()
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

    @app.post("/api/assemble")
    def assemble_clothing(req: AssembleRequest) -> dict:
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

        try:
            run = assemble_fn(
                clothing_fbx=clothing_path,
                target_id=req.target_id,
                out_fbx=out_fbx,
                registry=registry,
                schema=schema,
                llm_client=llm_client,
                user_drop_bone_ids=drop_ids,
                target_drop_bone_ids=target_drop_ids,
                drop_mesh_ids=drop_mesh,
                target_drop_mesh_ids=target_drop_mesh,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"assemble failed: {e}")

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

    return app
