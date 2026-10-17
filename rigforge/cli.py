"""rigforge — v1 CLI entry point.

Subcommands:
    rigforge assemble    produce the merged FBX
    rigforge inspect     print Phase A + B diagnostics without writing FBX
    rigforge serve       run the FastAPI bridge for the Vite + Vue 3 frontend
    rigforge taste-edit  vertex-count-safe geometry tweaks on a clothing
                         (e.g. cut sole height) BEFORE assemble — a pre-step,
                         like MochiFitter. Its output feeds `assemble`.

LLM selection
-------------
v1 ships without the LoRA-fine-tuned tag (see PLAN.md Open Question #1), so
`assemble` defaults to MockLLMClient and requires a `--llm-mock FILE.json`.
When the real Ollama tag lands, `--llm-ollama TAG` will become the default.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from rigforge.avatars.registry import AvatarRegistry
from rigforge.cache.store import DecisionCache
from rigforge.canonical.schema import CanonicalSchema
from rigforge.llm.anthropic import AnthropicLLMClient
from rigforge.llm.client import LLMClient
from rigforge.llm.mock import MockLLMClient
from rigforge.pipeline.bone_transforms import (
    BoneTransformError,
    parse_bone_transform_spec,
)
from rigforge.llm.ollama import OllamaLLMClient
from rigforge.manifest import write_manifest
from rigforge.pipeline.orchestrator import ClothingSpec, PipelineError, assemble


def _build_llm_client(args) -> LLMClient:
    if args.llm_mock:
        return MockLLMClient(Path(args.llm_mock), model_id=f"mock@{args.llm_mock}")
    if args.llm_ollama is not None:
        config_path = None if args.llm_ollama == "<default>" else Path(args.llm_ollama)
        return OllamaLLMClient.from_config_file(config_path)
    if getattr(args, "llm_anthropic", None) is not None:
        key_path = None if args.llm_anthropic == "<default>" else Path(args.llm_anthropic)
        return AnthropicLLMClient.from_key_file(key_path)
    raise SystemExit(
        "error: provide one of --llm-mock <decisions.json>, --llm-ollama "
        "[config.json], or --llm-anthropic [keyfile.md]. Ollama config "
        "lives in data/ollama.json; Anthropic key in "
        "API_KEY_MAKE_VERY_SURE_NO_LEAKAGE.md by default."
    )


def cmd_assemble(args) -> int:
    registry = AvatarRegistry.load(Path(args.registry)) if args.registry \
        else AvatarRegistry.load_default()
    schema = CanonicalSchema.load(Path(args.schema)) if args.schema \
        else CanonicalSchema.load_default()
    cache = DecisionCache(root=Path(args.cache_dir)) if args.cache_dir else None

    llm = _build_llm_client(args)

    # Bone moves: parse every spec up front so a typo fails before any work.
    frame = args.bone_transform_frame
    try:
        bone_transforms = [
            parse_bone_transform_spec(s, frame=frame) for s in (args.bone_transform or [])
        ]
        target_bone_transforms = [
            parse_bone_transform_spec(s, frame=frame)
            for s in (args.target_bone_transform or [])
        ]
    except BoneTransformError as e:
        print(f"error: --bone-transform: {e}", file=sys.stderr)
        return 2

    clothing_paths = [Path(p) for p in args.clothing]
    if len(clothing_paths) > 1 and bone_transforms:
        print("error: --bone-transform names a clothing bone; with several "
              "--clothing it is ambiguous which garment it belongs to "
              "(use --target-bone-transform, or one --clothing)", file=sys.stderr)
        return 2
    common = dict(
        target_id=args.target,
        out_fbx=Path(args.out),
        registry=registry,
        schema=schema,
        llm_client=llm,
        cache=cache,
        translate_target_morphs=not args.no_translate_target_morphs,
        target_bone_transforms=target_bone_transforms or None,
    )
    try:
        if len(clothing_paths) == 1:
            run = assemble(
                clothing_fbx=clothing_paths[0],
                bone_transforms=bone_transforms or None,
                **common,
            )
        else:
            # Several garments → ONE merged output, in the order given.
            run = assemble(
                clothings=[ClothingSpec(clothing_fbx=p) for p in clothing_paths],
                **common,
            )
    except PipelineError as e:
        print(f"pipeline failed: {e}", file=sys.stderr)
        return 1

    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        manifest_path = Path(args.out).with_suffix(".manifest.json")
    write_manifest(run, manifest_path)

    print(f"ok  wrote {run.output_fbx}")
    if len(run.clothing_runs) > 1:
        print(f"    merged {len(run.clothing_runs)} garments onto target={run.target_id}:")
        for r in run.clothing_runs:
            print(f"    - {r.stem}: donor={r.donor_id} (score={r.score:.3f})  "
                  f"edits: {len(r.edit_plan.renames)} renames, {len(r.edit_plan.drops)} drops  "
                  f"cache: {'HIT' if r.cache_hit else 'MISS'} ({r.cache_key[:12]}...)")
            for w in r.warnings:
                print(f"        warning {w.rule}: {w.message}")
    else:
        print(f"    donor={run.donor_id} (score={run.score:.3f})  target={run.target_id}")
        print(f"    edits: {len(run.edit_plan.renames)} renames, "
              f"{len(run.edit_plan.drops)} drops")
        print(f"    cache: {'HIT' if run.cache_hit else 'MISS'} ({run.cache_key[:12]}...)")
        if run.warnings:
            print(f"    warnings ({len(run.warnings)}):")
            for w in run.warnings:
                print(f"      - {w.rule}: {w.message}")
    print(f"    manifest: {manifest_path}")
    return 0


def cmd_inspect(args) -> int:
    from rigforge.pipeline.phase_a import run_phase_a

    registry = AvatarRegistry.load(Path(args.registry)) if args.registry \
        else AvatarRegistry.load_default()
    work_dir = Path(args.work_dir) if args.work_dir else Path(args.clothing).parent / "_rigforge_work"
    try:
        a = run_phase_a(Path(args.clothing), registry,
                        work_dir=work_dir, threshold=args.threshold)
    except Exception as e:
        print(f"phase A failed: {e}", file=sys.stderr)
        return 1
    print(f"donor: {a.donor_id} (score={a.score:.3f})")
    print(f"top candidates:")
    for cid, score in a.candidates:
        print(f"  - {cid}: {score:.3f}")
    print(f"limb bones in clothing: {len(a.view.limb_bones())}")
    return 0


def cmd_serve(args) -> int:
    import uvicorn  # local import: optional dep, only required for `serve`
    from rigforge.api.server import create_app

    # LLM client + cache wiring — same options the assemble subcommand
    # accepts, so the FE-triggered assemble runs against the same config.
    llm_client = None
    if (
        args.llm_mock
        or args.llm_ollama is not None
        or getattr(args, "llm_anthropic", None) is not None
    ):
        llm_client = _build_llm_client(args)
    cache = DecisionCache(root=Path(args.cache_dir)) if args.cache_dir else None
    schema = CanonicalSchema.load(Path(args.schema)) if args.schema \
        else CanonicalSchema.load_default()
    registry = AvatarRegistry.load(Path(args.registry)) if args.registry \
        else AvatarRegistry.load_default()

    output_dir = Path(args.output_dir) if args.output_dir else None

    if llm_client is None:
        # Run anyway — inspect + manifest browsing still work without an LLM.
        # The assemble endpoint will 500 with a clear message if the user
        # tries to hit it without one.
        print("warn: no LLM client configured; assemble endpoint will fail. "
              "Pass --llm-ollama [CONFIG], --llm-anthropic [KEYFILE], or "
              "--llm-mock FIXTURE to enable it.")

    # Wrap assemble so the API can pass a single fn into create_app and we
    # close over cache from here (orchestrator.assemble takes cache as a kwarg).
    def assemble_with_cache(**kwargs):
        from rigforge.pipeline.orchestrator import assemble
        return assemble(cache=cache, **kwargs)

    app = create_app(
        data_dir=Path(args.data_dir),
        registry=registry,
        schema=schema,
        llm_client=llm_client,
        output_dir=output_dir,
        assemble_fn=assemble_with_cache,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def cmd_taste_edit(args) -> int:
    """Apply a vertex-count-safe geometry edit to a clothing FBX (a pre-step;
    the result feeds `assemble`). Edits run through the headless-Blender backend
    in rigforge.geometry."""
    from rigforge.geometry import blender_edit
    from rigforge.geometry.report import validate_roundtrip_report

    src = Path(args.clothing)

    # --list-meshes: help the user find the sole/footwear mesh name (it's the
    # bottom-most mesh), then exit — no edit, no output.
    if args.list_meshes:
        rep = blender_edit.inspect(src)
        meshes = rep.get("meshes", {})
        print(f"{len(meshes)} mesh(es), bottom-most first (world-{args.axis.upper()}):")
        for name, m in sorted(meshes.items(), key=lambda kv: kv[1].get("wz_min", 0.0)):
            print(f"  wz=[{m.get('wz_min')},{m.get('wz_max')}] "
                  f"verts={m.get('verts')} skinned={m.get('skinned')}  {name}")
        return 0

    if not args.mesh or not args.out:
        raise SystemExit("taste-edit: --mesh and --out are required (or use --list-meshes)")

    preview = Path(args.preview_dir) if args.preview_dir else None
    out = Path(args.out)

    if args.op == "bisect_cut":
        rep = blender_edit.bisect_cut(
            src, out, mesh=args.mesh, axis=args.axis,
            cut_pct=args.cut_pct, preview_dir=preview,
        )
        e = rep["edit"]
        new_unweighted = e["unweighted_verts_after"] - e["unweighted_verts_before"]
        print(f"ok  bisect_cut {e['mesh']}: verts {e['verts_before']} -> "
              f"{e['verts_after']} (removed {e['verts_removed']})")
        if new_unweighted > 0:
            print(f"FAIL: cut left {new_unweighted} new unweighted vertex(es) — the "
                  f"cut edge lost skin weights; the sole won't deform with the foot",
                  file=sys.stderr)
            return 1
        print("    weights intact (no new unweighted verts)")
    else:
        if args.op == "scale_slice":
            rep = blender_edit.scale_slice(
                src, out, mesh=args.mesh, axis=args.axis,
                from_pct=args.from_pct, to_pct=args.to_pct,
                factor=args.factor, preview_dir=preview,
            )
        else:  # translate_slice
            rep = blender_edit.translate_slice(
                src, out, mesh=args.mesh, axis=args.axis,
                from_pct=args.from_pct, to_pct=args.to_pct,
                delta_frac=args.delta_frac, preview_dir=preview,
            )
        e = rep["edit"]
        check = validate_roundtrip_report(rep["before"], rep["after"])
        print(f"ok  {args.op} {e['mesh']}: moved {e.get('moved_count')} verts")
        if not check.ok:
            for prob in check.problems:
                print(f"FAIL: {prob}", file=sys.stderr)
            return 1
        print("    geometry preserved (verts / skin weights / shape keys)")

    print(f"    wrote {out}")
    if preview:
        print(f"    previews in {preview}")
    print(f"    next: rigforge assemble --clothing {out} --target <id> --out <final>.fbx ...")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rigforge", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("assemble", help="produce the merged FBX")
    a.add_argument("--clothing", required=True, action="append",
                   help="input clothing FBX path; repeat to merge several "
                        "garments into ONE output, in the order given")
    a.add_argument("--target", required=True, help="curated target avatar id")
    a.add_argument("--out", required=True, help="output FBX path")
    a.add_argument("--registry", help="path to curated_avatars.json (default: data/curated_avatars.json)")
    a.add_argument("--schema", help="path to canonical_schema.json")
    a.add_argument("--cache-dir", help="decision cache root (default: ./cache)")
    a.add_argument("--manifest", help="output path for the run manifest JSON")
    a.add_argument("--no-translate-target-morphs", action="store_true",
                   help="keep the base avatar's original (untranslated) morph + "
                        "mesh names; clothing names are always translated")
    a.add_argument("--bone-transform", action="append", metavar="SPEC",
                   help='move a CLOTHING bone (rest pose; its mesh follows): '
                        '"NAME;pos=x,y,z;rot=x,y,z;scale=x,y,z" — any subset of '
                        'pos/rot/scale; "#<id>" instead of NAME targets a Model '
                        'id. Repeatable.')
    a.add_argument("--target-bone-transform", action="append", metavar="SPEC",
                   help="same as --bone-transform, for a bone of the TARGET avatar")
    a.add_argument("--bone-transform-frame", choices=["unity", "fbx"], default="unity",
                   help="convention of the numbers in --bone-transform specs: "
                        "'unity' = what Unity's Inspector shows for the bone "
                        "(default); 'fbx' = raw Lcl Translation/Rotation/Scaling")
    llm_group = a.add_mutually_exclusive_group()
    llm_group.add_argument("--llm-mock", help="path to a fixture decisions JSON (MockLLMClient)")
    llm_group.add_argument(
        "--llm-ollama", nargs="?", const="<default>", metavar="CONFIG",
        help="Use OllamaLLMClient. Optional path to ollama config JSON; "
             "defaults to data/ollama.json.",
    )
    llm_group.add_argument(
        "--llm-anthropic", nargs="?", const="<default>", metavar="KEYFILE",
        help="Use AnthropicLLMClient (Messages API). Optional path to a "
             "file containing the API key; defaults to "
             "API_KEY_MAKE_VERY_SURE_NO_LEAKAGE.md.",
    )
    a.set_defaults(func=cmd_assemble)

    i = sub.add_parser("inspect", help="Phase A donor diagnostics only (no FBX output)")
    i.add_argument("--clothing", required=True)
    i.add_argument("--registry")
    i.add_argument("--threshold", type=float, default=0.85)
    i.add_argument("--work-dir")
    i.set_defaults(func=cmd_inspect)

    s = sub.add_parser("serve", help="run the FastAPI bridge for the FE")
    s.add_argument("--data-dir", default="data/training/_stress",
                   help="directory of *.manifest.json files (default: data/training/_stress)")
    s.add_argument("--output-dir", help="where assembled FBXs are written "
                                         "(default: same as --data-dir)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--registry", help="path to curated_avatars.json")
    s.add_argument("--schema", help="path to canonical_schema.json")
    s.add_argument("--cache-dir", help="decision cache root (default: ./cache)")
    s_llm = s.add_mutually_exclusive_group()
    s_llm.add_argument("--llm-mock", help="path to a fixture decisions JSON (MockLLMClient)")
    s_llm.add_argument(
        "--llm-ollama", nargs="?", const="<default>", metavar="CONFIG",
        help="Use OllamaLLMClient. Optional path to ollama config JSON; "
             "defaults to data/ollama.json.",
    )
    s_llm.add_argument(
        "--llm-anthropic", nargs="?", const="<default>", metavar="KEYFILE",
        help="Use AnthropicLLMClient (Messages API). Optional path to the "
             "API key file; defaults to API_KEY_MAKE_VERY_SURE_NO_LEAKAGE.md.",
    )
    s.set_defaults(func=cmd_serve)

    te = sub.add_parser(
        "taste-edit",
        help="vertex-count-safe geometry tweak on a clothing (pre-assemble)",
    )
    te.add_argument("--clothing", required=True, help="input clothing FBX path")
    te.add_argument("--list-meshes", action="store_true",
                    help="list meshes (with world-Z bounds) to find the sole/"
                         "footwear mesh, then exit — no edit")
    te.add_argument("--mesh", help="target mesh name (see --list-meshes)")
    te.add_argument("--op", choices=["bisect_cut", "scale_slice", "translate_slice"],
                    default="bisect_cut",
                    help="bisect_cut: real cut, removes height with no texture "
                         "compression (default). scale_slice: squash a band "
                         "(compresses texture). translate_slice: rigid move.")
    te.add_argument("--axis", choices=["x", "y", "z"], default="z",
                    help="world axis (default z = up)")
    te.add_argument("--out", help="output edited clothing FBX path")
    te.add_argument("--preview-dir", help="render before/after front+side PNGs here")
    # bisect_cut
    te.add_argument("--cut-pct", type=float, default=0.15,
                    help="bisect_cut: remove everything below this fraction of "
                         "the mesh's axis extent (default 0.15)")
    # slice ops (scale_slice / translate_slice)
    te.add_argument("--from-pct", type=float, default=0.0,
                    help="slice ops: band start (fraction of axis extent)")
    te.add_argument("--to-pct", type=float, default=0.2,
                    help="slice ops: band end (fraction of axis extent)")
    te.add_argument("--factor", type=float, default=0.5,
                    help="scale_slice: squash factor toward band top (0<f<1 shortens)")
    te.add_argument("--delta-frac", type=float, default=-0.3,
                    help="translate_slice: signed move as a fraction of axis extent")
    te.set_defaults(func=cmd_taste_edit)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
