# RigForge — Booth-Asset Assembly Tool

Sibling repo to REE-ModPilot. Standalone Python tool. Output is a clean assembly FBX that ModPilot's existing Phase D pipeline consumes.

## Context

REE-ModPilot already handles the *adapt-to-Hunter* leg of MHWilds modding well. The pain that remains is *upstream*: producing the clean assembly FBX that ModPilot consumes — combining a booth-purchased base avatar + clothing rigged to some other booth base into one coherent FBX with consolidated armature and clean bone names. Blender is hostile here because:

- Booth assets have wildly inconsistent bone naming. Same logical bone shows up under dozens of name variants.
- Aux/null bones (IK targets, twist correctives, mixamo locators) pollute the rig and Blender's drop-bone op is stateful and unreliable.
- The agent currently burns most of its context translating modding intent into Blender's general-purpose API.

Goal: a standalone tool whose only deliverable is a *clean assembly FBX* — single armature inherited from the chosen target avatar, clothing meshes re-skinned to that armature, junk bones dropped, names normalized to canonical roles. ModPilot picks up from there.

## Architectural Decisions (locked)

- **ASCII FBX is the internal representation.** Binary → ASCII at import, ASCII → binary at export. Conversion verified bit-exact (see Converter Toolchain).
- **bpy is fully excluded.** All bone-level ops (drop, rename, reparent, armature merge) are deterministic text transforms on ASCII FBX. The converter resolves the only path that previously required bpy. This is the core motivation — escape Blender's stateful unreliability.
- **The bone-analysis intelligence is a small fine-tuned LLM**, not a structured-data parser + general LLM. Reasoning: pure parser-to-schema is too rigid for booth's diversity, can't do role interpretation, and forces maintenance pain whenever a new convention appears. A trained model handles all three.
- **Curated avatar set only.** Non-curated donor rigs → hard fail with a "closest matches" error.
- **All curated avatars are T-pose.** No pose normalization in v1.
- **Mesh fitting, materials, blendshapes deferred to v2.**

## Decisions Locked

| # | Decision |
|---|----------|
| Tool name | **RigForge** |
| Repo location | Sibling of REE-ModPilot (this directory) |
| Training compute | **Lambda Labs cloud LoRA** (user provisioning) |
| Training data | Modder collaborator has historical hand-fixed pairs — primary source |
| Base model | **DeepSeek V4 Flash** (matches ModPilot production); fallback Qwen2.5-7B if not LoRA-able |
| Inference | Ollama tag |
| Curated avatars | 4 (specific identities TBD) |
| Test corpus | 10+ clothing items |

## Converter Toolchain (resolved)

The binary↔ASCII conversion is solved by a standalone toolchain built in a separate session:

- **Env:** dedicated `fbx_env` Python 3.10 conda env with Autodesk FBX SDK 2020.3.9 Python bindings (kept isolated from the main project env — the SDK install is fragile).
- **Scripts:** `fbx_bin2ascii.py`, `fbx_ascii2bin.py`, `fbx_compare.py`.
- **Behavior:** both converters auto-detect the source FBX schema version and pin the output to match (e.g., 7.4.0 in → 7.4.0 out), select writer by description with a header magic-byte sanity check, and hard-fail on any SDK status error — no silent drops.
- **Verified:** round-trip on Maya.fbx (6.1MB bin → 40MB ASCII → 10.7MB bin) is bit-exact for data; skin-weight checksum across 1,368 clusters / 32K weights agreed to 2.29e-16 relative error (1 ULP at double precision). Output binary size differs from input only because Blender uses more aggressive zlib than the SDK default.
- **Integration shape:** `ascii_fbx/convert.py` is a thin subprocess wrapper that invokes these scripts against the `fbx_env` conda env. No FBX SDK in the main project env.
- **Regression-test harness:** `fbx_compare.py` is the gate. Any pipeline change that produces an FBX must round-trip through compare against a baseline before merging.

## Two-Rail Architecture

```
  Rail D (deterministic text)      Rail L (one trained LLM call)
  ─────────────────────────        ────────────────────────────
  binary → ASCII convert
  Lexer + Section extractor   ───►  per-bone {role, verdict}
  Fingerprint (Phase A)
  Cache lookup
                              ◄─── (validated JSON output)
  Validators (7 rules)
  EditPlan synthesis
  Atomic ASCII edits
  Armature merge (Phase C)
  ASCII → binary convert
```

Rail D is the spine. Rail L is a single classification node inside Phase B — no agent loop, no ReAct, no tool-calling. One batched call per assembly run, cached aggressively.

## Pipeline Phases

### Phase A — Donor Identification (deterministic only)

**Goal:** "Which curated base was this clothing rigged for?"
**Method:** Convert clothing to ASCII → extract bone-name set → normalize (lowercase, separator-stripped) → Jaccard similarity against curated registry fingerprints → highest match ≥ 0.85 wins, else hard-fail with top-3 candidates.
**No LLM** — closed-set classification is cheaper as fingerprinting.

### Phase B — Cross-Booth Rig Mapping (Rail L lives here)

**Inputs:** ASCII text + donor_id (from A) + target_id (user-selected).
**Outputs:** validated `EditPlan` (drops + renames).

**Steps:**
1. **Section extraction** (Rail D) — lexer produces normalized JSON record per bone: `{model_id, name, parent_name, translation_xyz, child_names, has_skin_cluster, cluster_weight_count}`. Raw text offsets retained on the side, keyed by `model_id`, for the editor stage.
2. **Cache lookup** (Rail D) — see Cache Strategy. Hit → skip step 3.
3. **LLM call** (Rail L) — batched, one call. Input: JSON list of all bones + target/donor IDs as context. Output: `{bones: [{model_id, role, verdict, drop_category?, confidence}, ...]}`. JSON-schema-validated.
4. **Validation** (Rail D) — 7 rules below. Any error-level violation → re-prompt once with the violation text appended; second failure → hard fail.
5. **EditPlan synthesis** (Rail D) — drops = `verdict=drop`; renames = `{donor_name → target_avatar.canonical_to_name[role]}` for each kept bone; cluster cascade drops attached to dropped bones.

**LLM/Rail D boundary stays clean:** LLM never sees text it modifies. It sees a normalized JSON view and emits a JSON judgment. All edits are mechanical.

### Phase C — Armature Merge (deterministic only)

1. Load cached target-avatar ASCII (binary→ASCII done once at registry-build).
2. Drop clothing's armature root. Strip clothing's **canonical** bone Models — the target avatar owns those, and its bones supply the canonical names via the name/role repoint (step 3), so renaming the clothing copy is moot. Non-canonical bones (skirt/hair/accessory chains, plus anything the LLM couldn't place) ride along and are **renamed to their English translation** (`Decision.name_en` — e.g. a Japanese front-skirt bone → `Skirt_Front.001`), so the multilingual team can read JP/KR/CN Booth bone names. Established rigging romaji (Kemomimi, Ahoge) is kept as-is; bones the LLM left without a `name_en` keep their original name.
3. Re-point cluster connections so `Cluster`-to-bone connections target the *target avatar's* model ids by name lookup.
4. Append clothing meshes + (re-pointed) clusters into the target avatar's Objects + Connections sections.
5. Renumber colliding model ids (collision check happens at registry load).

Output is the merged ASCII. Final `ASCII → binary` produces the deliverable.

## Compose UI (v2 user-facing flow)

The CLI is the v1 entry point and still works (`rigforge assemble`). The v2 face is the compose UI, served by `rigforge serve` + the Vite dev server.

What the modder does:
1. Pick a target avatar from a dropdown (curated registry).
2. Inspect lists the target's meshes in a Blender-outliner layout — armature header + parallel mesh names (Body, Hair, Cloth, Shoes, Hat, ...). Uncheck the ones bundled with the avatar that they don't want kept (Maya's default outfit, etc.).
3. Paste a clothing FBX path → inspect. Repeat for multiple clothings if needed; each gets its own panel.
4. In each clothing's panel, uncheck meshes they want stripped from that clothing (e.g., a cape they don't want).
5. Click Assemble. Backend runs the full pipeline (Phase A → B → C) for each clothing against the target, applying the user's mesh drops as pre-filters. Output is a binary FBX + manifest per clothing.

The modder never sees bones. Bone-level decisions still happen — they live in the pipeline and the LLM call — but bones are an internal abstraction. The unit of user agency is the mesh, because Maya's rigging convention puts every bone in every mesh's skin with zero weights, which would make a bone-tree UI lie about what each checkbox actually does.

## Canonical Schema (v1, lean)

Unity-Humanoid-shaped intermediate, but trimmed for v1. Frozen and versioned (`canonical_schema_version` in cache key).

Locked v1 roles (~22):
- Core: Hips, Spine, Chest, (UpperChest optional), Neck, Head
- Arms: Shoulder.{L,R}, UpperArm.{L,R}, LowerArm.{L,R}, Hand.{L,R}
- Legs: UpperLeg.{L,R}, LowerLeg.{L,R}, Foot.{L,R}, Toes.{L,R}
- Secondary chains: Breast.{L,R}.NN, HairSecondary.{C,L,R}.NN, SkirtFront.C.NN, etc.

**Excluded from v1:** finger bones, twist bones. Both can be added later without breaking cache (schema version bump invalidates).

LLM output of `"unknown"` or `"aux"`:
- `"unknown"` → validator hard-fails.
- `"aux"` → auto-routed to drop verdict.

## Validation Rules (deterministic, run between LLM and edit)

1. **unique_role** — every non-secondary canonical role assigned exactly once
2. **required_roles_present** — Hips/Spine/Chest/Head/UpperArm.L/.R/UpperLeg.L/.R all present
3. **l_r_pairing** — every `.L` has a `.R` peer mirroring in X (±5% tolerance)
4. **monotonic_spine_y** — Hips < Spine < Chest < (UpperChest?) < Neck < Head in Y
5. **hierarchy_consistency** — extracted parent chain agrees with canonical parent (e.g., assigned `Hand.L` must trace up to `UpperChest` through arm chain)
6. **drop_safety** — bones with `verdict=drop` have summed cluster weight ≤ 0.05; high-weight drops are likely misclassifications
7. **finger_count_sanity** — warn-only in v1 (no fingers in lean schema)

Severity: errors block, warnings log.

## Cache Strategy

```
cache_key = sha256(
    canonical_schema_version
    | llm_model_id_and_version
    | donor_avatar_id
    | target_avatar_id
    | sorted(normalized_bone_name_set)
    | sorted(bone_hierarchy_edges)         # (parent, child) tuples
    | rounded_translation_signature        # mm-precision to absorb DCC re-save noise
)
```

Stored as `cache/decisions/<key>.json` containing **raw LLM output**, not EditPlan. EditPlan is regenerated each run from raw output + current avatar registry (registry is mutable).

## Training Pipeline

**Dataset bootstrap:**
- 4 curated avatars auto-label themselves (their `canonical_to_name` is the gold).
- Modder collaborator's historical hand-fixed clothing pairs auto-label — primary high-quality source.
- Augmentation: for each labeled rig, generate K≈20 perturbations (name-pattern substitutions `J_Bip_`↔`bone_`↔`b_`, separator swaps, injected aux bones with `_dummy`/`_null`/`IK_target` patterns, small noise translations).
- Target dataset size for v1: 500–2000 examples.

**Base model:** DeepSeek V4 Flash — matches ModPilot's production LLM for consistency. Open-weights LoRA-fineability needs verification before committing compute. Fallback if blocked: Qwen2.5-7B-Instruct, same training pipeline.

**Compute:** Lambda Labs cloud LoRA (user provisioning).

**Inference:** Ollama tag — reuses ModPilot's `LLMClient` Ollama provider pattern.

**Eval metrics:** per-bone role accuracy, per-rig EditPlan equivalence on holdout, hard-failure rate (<5% target on holdout).

**Active learning (v2):** low-confidence runs surface a hand-review UI; corrections feed back into the dataset.

## Module Layout

```
D:/2files/models/vrc/CC/Blender-MHWilds/
└── RigForge/
    ├── rigforge/                        # python package
    │   ├── ascii_fbx/                   # lexer, sections, edits, writer, convert
    │   ├── canonical/                   # schema, validators, target rename map
    │   ├── avatars/                     # registry, fingerprint
    │   ├── llm/                         # client (Ollama tag), prompt builder, JSON output parser
    │   ├── pipeline/                    # phase_a / phase_b / phase_c / orchestrator
    │   ├── cache/                       # key, store
    │   ├── sections/                    # materials/blendshapes merge primitives (v2)
    │   ├── api/                         # FastAPI bridge for the compose UI (v2)
    │   ├── manifest.py                  # run report emitter
    │   └── cli.py                       # `assemble` / `inspect` / `serve`
    ├── frontend/                        # Vite + Vue 3 compose UI (v2)
    ├── data/
    │   ├── curated_avatars.json
    │   ├── canonical_schema.json
    │   ├── ollama.example.json          # template; ollama.json gitignored
    │   └── training/                    # dataset, gitignored beyond pointer
    ├── training/                        # dataset_build, smoke, eval scripts
    ├── tests/
    └── pyproject.toml
```

## v1 vs v2 Scope Split

| Capability | v1 | v2 |
|---|---|---|
| Phase A (fingerprint donor ID) | yes | — |
| Phase B (LLM canonical mapping + drop) | yes | — |
| Phase C (text-level armature merge) | yes | — |
| Bone drop + rename | yes | — |
| Bone reparent (chain restructure) | no — validator hard-fails the case | yes (landed) |
| Materials, blendshapes (parsers + dedup primitives + Phase C integration) | no | yes (landed) |
| Fingers, twist bones in canonical | no | yes (schema v2.1) |
| JP/KR→EN name translation — bones (LLM `name_en`, schema v2.2) | no | yes (landed) |
| JP/KR→EN name translation — morphs + meshes (dictionary, no LLM; FE checkbox for target) | no | yes (landed) |
| CLI front door (`assemble`, `inspect`) | yes | + `serve` |
| FastAPI bridge | no | yes (landed) |
| Compose UI: avatar pick → clothing list → mesh-level drops → assemble | no | yes (landed — v2's user-facing deliverable) |
| User pre-filter (drop unwanted meshes from clothing AND target) | no | yes (landed) |
| User pre-filter (drop unwanted blendshape channels from clothing AND target) | no | yes (landed) |
| Per-channel DeformPercent override (bake baseline morph intensity) | no | yes (landed) |
| Streaming assemble progress (NDJSON) + FE progress bar | no | yes (landed) |
| Caching | yes | — |
| Manifest emission | yes | — |
| Mesh shape fitting | no | **out of scope** — clothing is assumed pre-fit |
| Non-T-pose normalization | no | **out of scope** — asset pipeline guarantees T-pose |
| Non-curated donor rigs / unknown-donor recovery | no — hard fail | **out of scope** — Phase A hard-fails below 0.85 |
| Active-learning correction UI | no | parked — revisit after v2 ships |

## Critical Files (will be created)

- `rigforge/ascii_fbx/sections.py` — extractor producing the LLM input shape; gates everything downstream
- `rigforge/ascii_fbx/edits.py` — atomic drop/rename/reparent ops on ASCII text
- `rigforge/ascii_fbx/convert.py` — subprocess wrapper around `fbx_env` conda scripts
- `rigforge/canonical/validators.py` — the 7 rules; the safety net between LLM and FBX edits
- `rigforge/pipeline/orchestrator.py` — glue: bin→ASCII→A→B→C→ASCII→bin
- `data/canonical_schema.json` — frozen v1 IR
- `data/curated_avatars.json` — registry seed

## Reference Files in REE-ModPilot (patterns to mirror, NOT to import)

- `ModPilot/app/llm/client.py` — provider abstraction (vendor a copy; RigForge stays independent)
- `ModPilot/app/phases/base.py` — pure-executor pattern, `PhaseResult` shape
- `ModPilot/app/agent/history.py` — for v2 active-learning conversation state (not needed in v1)

## Verification Plan

End-to-end smoke test for v1:
1. Pick one curated target avatar.
2. Pick one clothing FBX known to be rigged for a *different* curated base.
3. Run: `rigforge assemble --clothing path.fbx --target <id> --out out.fbx`
4. Confirm: output FBX loads in Blender; armature is a single rig matching target's bone names; clothing mesh visible and skinned; no junk bones in outliner.
5. Run `fbx_compare.py` against a held-out golden output for the same inputs — schema, node/mesh/bone/material/cluster/blend-shape counts, axis/unit/time, and skin-weight checksum must all match.
6. Pass to existing ModPilot Phase D pipeline and confirm Phase D consumes it without preset gymnastics.

Per-component tests:
- `ascii_fbx/edits.py` — drop bone, drop cluster, rename, write-back; assert byte-for-byte preservation of unrelated nodes
- `ascii_fbx/convert.py` — round-trip (bin → ASCII → bin) sanity, verified with `fbx_compare.py` on a fixture set
- `canonical/validators.py` — each rule has positive + negative fixtures
- `avatars/fingerprint.py` — Jaccard scoring on a small synthetic registry
- LLM JSON adherence — invariant test: 100 examples in, all outputs schema-valid

## Open Questions

1. **DeepSeek V4 Flash LoRA-ability** — RESOLVED: DS V4 Flash is not LoRA-able. Production runs zero-shot via Ollama Cloud `/api/chat` (Bearer auth, `format=json`, `temperature=0`, `stream=true`). The 5-clothing stress test (see Validation Status below) showed zero-shot Flash passes drop sanity with 0 high-weight drops across 877 bones — the over-pruning failure mode is not triggered on real clothing. LoRA pipeline + Qwen fallback are deferred indefinitely. Revisit only if production surfaces a high-weight-drop red flag.
2. **Converter script paths on disk** — RESOLVED: `D:/2files/models/vrc/Maya/Maya_Ver1.02.2/{fbx_bin2ascii,fbx_ascii2bin,fbx_compare}.py`, invoked through `fbx_env` conda env at `D:/2files/env_build/conda/conda/envs/fbx_env/python.exe`.
3. **Curated set seed** — Maya + Moe are in (`data/curated_avatars.json`). Production focus has shifted to Maya-only since the 5-clothing stress validated that all common booth clothing patterns can be classified against Maya's bone set; additional avatars are nice-to-have, not blocking.
4. **Donor scope variety** — Partially closed. Real-clothing evaluation across 5 booth-distinct assets (ClassicChic, AzureVirtue, RabbitGear, PicodraTech, SchoolUniform) all rigged for Maya passed. No modder hand-fixed pairs yet — only relevant if we ever decide to LoRA.

## Validation Status (2026-05-21)

**Phase B / DS V4 Flash on 5 real clothings → target=maya:**

| Clothing | Bones | Keep | Drop | High-weight drops | Phase A score |
|---|---:|---:|---:|---:|---:|
| classic_chic    | 211 | 155 |  56 | 0 | 1.000 |
| azure_virtue    | 298 | 125 | 173 | 0 | 1.000 |
| rabbit_gear     |  56 |  44 |  12 | 0 | 1.000 |
| picodra_tech    | 220 | 161 |  59 | 0 | 0.964 |
| school_uniform  |  92 |  57 |  35 | 0 | 0.971 |
| **Total**       | **877** | **542** | **335** | **0** | — |

All 5 closed cleanly: required canonical roles assigned, validators passed first-try (no re-prompt), 0 bones dropped where `cluster_weight_count > 100` — the over-pruning red flag we'd watch for. LLM-bound wall time totaled 35 min (3–10 min per clothing).

**Pass-through e2e (mock LLM, ClassicChic_Moe → Moe):** structurally identical to input modulo bin↔ASCII roundtrip (node_count, total_clusters, total_skin_weights all match to 1 ULP).

**Cross-avatar merge (Phase C donor≠target):** implemented + unit-tested as a structural fallback. Not a real production pairing — every clothing in `D:/2files/models/vrc/ASCII_models/clothing/` is already paired with its intended curated avatar.

Full report at `data/training/_stress/REPORT.md`; per-clothing manifests at `data/training/_stress/<slug>.manifest.json`.
