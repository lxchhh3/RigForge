"""Manual smoke test against the real Ollama Cloud API.

NOT run by pytest. Use this to validate your data/ollama.json + key + model
tag work end-to-end on real bone data. Defaults exercise the synthetic
clothing fixture (perturbed Maya).

Usage:
    PYTHONPATH=. python training/smoke_ollama.py

    # Or with custom inputs:
    PYTHONPATH=. python training/smoke_ollama.py \\
        --clothing path/to/clothing_ascii.fbx \\
        --target maya \\
        --config data/ollama.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rigforge.ascii_fbx.lexer import parse as parse_fbx
from rigforge.ascii_fbx.sections import extract
from rigforge.avatars.registry import AvatarRegistry, identify_donor
from rigforge.canonical.schema import CanonicalSchema
from rigforge.canonical.validators import errors, validate_all, warnings
from rigforge.llm.config import load_config
from rigforge.llm.ollama import OllamaLLMClient


REPO_ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clothing", type=Path,
                   help="Clothing ASCII FBX (default: builds synth from Maya)")
    p.add_argument("--target", default="maya", help="Target avatar id (default: maya)")
    p.add_argument("--config", type=Path,
                   help="Ollama config JSON (default: data/ollama.json)")
    p.add_argument("--n-bones", type=int, default=0,
                   help="If >0, only send first N limb bones (cheaper smoke test)")
    args = p.parse_args(argv)

    config = load_config(args.config)
    schema = CanonicalSchema.load_default()
    registry = AvatarRegistry.load_default()

    if args.clothing:
        clothing_ascii_path = args.clothing
    else:
        clothing_ascii_path = _ensure_synth_clothing()

    raw = clothing_ascii_path.read_bytes()
    view = extract(parse_fbx(raw))
    all_limbs = view.limb_bones()

    # Phase A on the FULL bone set (fingerprint must see the whole rig)
    donor_id, score, ranked = identify_donor([b.name for b in all_limbs], registry)
    print(f"[phase A] donor={donor_id} score={score:.3f}")
    print(f"[phase A] top: {ranked[:3]}")

    # Now optionally slice for the LLM call to keep cost down
    limbs = all_limbs
    if args.n_bones > 0:
        limbs = all_limbs[: args.n_bones]
        print(f"[trim] sending only first {len(limbs)} of {len(all_limbs)} bones to LLM")

    bones_payload = [b.to_json_record() for b in limbs]
    request = {
        "donor_id": donor_id,
        "target_id": args.target,
        "canonical_schema_version": schema.version,
        "bones": bones_payload,
    }

    print(f"[ollama] model={config.model} host={config.host}")
    print(f"[ollama] sending {len(bones_payload)} bone records...")
    client = OllamaLLMClient(config, schema=schema)
    decisions = client.classify(request)
    print(f"[ollama] got {len(decisions.bones)} decisions back")

    # Summarise verdicts
    n_keep = sum(1 for d in decisions.bones if d.verdict == "keep")
    n_drop = sum(1 for d in decisions.bones if d.verdict == "drop")
    print(f"[summary] keep={n_keep} drop={n_drop}")
    # Print a few role assignments
    print("[sample assignments]")
    for d in decisions.bones[:10]:
        print(f"  {d.model_id:>20} -> {d.role!r:<30} {d.verdict}  (conf={d.confidence:.2f})")

    # Run validators so user sees how clean the zero-shot output is
    full_view_for_validators = extract(parse_fbx(raw))
    if args.n_bones > 0:
        # filter to only the bones we sent so validators don't complain about
        # missing decisions for bones we didn't classify
        wanted = {d.model_id for d in decisions.bones}
        full_view_for_validators.bones = {
            k: v for k, v in full_view_for_validators.bones.items() if k in wanted
        }
    violations = validate_all(decisions, full_view_for_validators, schema)
    errs = errors(violations)
    warns = warnings(violations)
    print(f"[validate] errors={len(errs)} warnings={len(warns)}")
    for v in errs[:10]:
        print(f"  ERR  {v.rule}: {v.message}")
    for v in warns[:5]:
        print(f"  WARN {v.rule}: {v.message}")

    print("[done] zero-shot DS V4 Flash baseline captured.")
    return 0 if not errs else 1


def _ensure_synth_clothing() -> Path:
    """Build the synth clothing fixture in /tmp if not already present."""
    from training.synth_clothing import build_synth_clothing

    out_dir = REPO_ROOT / "data" / "training" / "_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    ascii_path = out_dir / "synth_clothing_ascii.fbx"
    decisions_path = out_dir / "synth_decisions.json"
    if not ascii_path.exists():
        source = Path("D:/2files/models/vrc/Maya/Maya_Ver1.02.2/Maya_ascii.fbx")
        if not source.exists():
            raise SystemExit(
                f"Maya_ascii.fbx not found at {source}. Pass --clothing PATH instead."
            )
        build_synth_clothing(
            source_ascii=source,
            out_ascii=ascii_path,
            out_decisions=decisions_path,
        )
        print(f"[setup] built synth clothing at {ascii_path}")
    return ascii_path


if __name__ == "__main__":
    sys.exit(main())
