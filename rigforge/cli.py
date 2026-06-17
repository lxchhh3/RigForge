"""rigforge — v1 CLI entry point.

Subcommands:
    rigforge assemble   produce the merged FBX
    rigforge inspect    print Phase A + B diagnostics without writing FBX
    rigforge serve      run the FastAPI bridge for the Vite + Vue 3 frontend

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
from rigforge.llm.ollama import OllamaLLMClient
from rigforge.manifest import write_manifest
from rigforge.pipeline.orchestrator import PipelineError, assemble


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

    try:
        run = assemble(
            clothing_fbx=Path(args.clothing),
            target_id=args.target,
            out_fbx=Path(args.out),
            registry=registry,
            schema=schema,
            llm_client=llm,
            cache=cache,
            translate_target_morphs=not args.no_translate_target_morphs,
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rigforge", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("assemble", help="produce the merged FBX")
    a.add_argument("--clothing", required=True, help="input clothing FBX path")
    a.add_argument("--target", required=True, help="curated target avatar id")
    a.add_argument("--out", required=True, help="output FBX path")
    a.add_argument("--registry", help="path to curated_avatars.json (default: data/curated_avatars.json)")
    a.add_argument("--schema", help="path to canonical_schema.json")
    a.add_argument("--cache-dir", help="decision cache root (default: ./cache)")
    a.add_argument("--manifest", help="output path for the run manifest JSON")
    a.add_argument("--no-translate-target-morphs", action="store_true",
                   help="keep the base avatar's original (untranslated) morph + "
                        "mesh names; clothing names are always translated")
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

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
