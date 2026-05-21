# RigForge — Architecture (as built)

Snapshot of the implemented pipeline. PLAN.md is the design doc; this file describes what actually exists in the tree and how the pieces fit together. v1 is the released pipeline; v2 work is landed incrementally and called out inline. Updated 2026-05-21.

## Runtime envs

Two Python envs, both required:

| Env | Purpose | Python | Lives at |
|---|---|---|---|
| RigForge project env | Everything except FBX SDK calls | 3.10+ | active env when running `rigforge` |
| `fbx_env` (conda) | Binary↔ASCII FBX conversion via Autodesk FBX SDK 2020.3.9 | 3.10 | `D:/2files/env_build/conda/conda/envs/fbx_env/` |

`rigforge.ascii_fbx.convert` shells out to `fbx_env` over subprocess; the SDK is never imported into the main env.

Toolchain scripts (third-party, not in this repo): `D:/2files/models/vrc/Maya/Maya_Ver1.02.2/{fbx_bin2ascii,fbx_ascii2bin,fbx_compare}.py`.

## Data flow (one assembly run)

```
clothing.fbx (binary)
    │
    │  fbx_env subprocess
    ▼
clothing_ascii.fbx (text)
    │
    ▼  ── lexer + sections.extract ──►  SectionView { bones, connections, byte offsets }
    │
    ├──► Phase A: fingerprint(bone names) → donor_id, score          [deterministic]
    │
    ├──► Phase B:                                                     [Rail D + Rail L]
    │       cache.get(key)? ── hit ─────────────────────┐
    │           │ miss                                  │
    │           ▼                                       │
    │       LLM.classify(bones)  ◄── one batched call ──┘
    │           │
    │           ▼
    │       validators (7 rules)
    │           │       │
    │           │ errs  │ clean
    │           ▼       ▼
    │       re-prompt   EditPlan.from_decisions(...)
    │       (once)
    │
    ├──► Phase C: apply drops + renames as text edits on ASCII
    │             (cross-avatar armature merge deferred to v1.1)
    │
    ▼
assembled_ascii.fbx
    │
    │  fbx_env subprocess
    ▼
assembled.fbx (binary) + assembled.manifest.json
```

## Module map

```
rigforge/
├── ascii_fbx/
│   ├── convert.py    bin↔ASCII + compare via fbx_env subprocess
│   ├── lexer.py      byte-level FBX tokenizer (preserves all offsets)
│   ├── sections.py   3-pass extract → Models + Deformers + Connections → BoneRecord
│   └── edits.py      TextEdit primitives; drop_bone_edits, rename_bone_edits
├── canonical/
│   ├── schema.py     CanonicalSchema (22 roles, parents, categories)
│   ├── decisions.py  Decision / DecisionSet pydantic models
│   └── validators.py 7 rules + driver
├── avatars/
│   ├── fingerprint.py  normalize + Jaccard
│   └── registry.py     load + identify_donor (threshold 0.85)
├── llm/
│   ├── client.py     LLMClient Protocol + LLMError
│   ├── mock.py       MockLLMClient + DispatchMockClient (fixture-driven)
│   ├── config.py     OllamaConfig (host/model/api_key/timeout)
│   ├── prompts.py    build_messages (system embeds full schema)
│   ├── parser.py     parse_decision_set (strict JSON → fenced → brace-slice)
│   └── ollama.py     OllamaLLMClient — POSTs /api/chat via urllib
├── cache/
│   ├── key.py        sha256 over schema|llm|donor|target|bones|edges|mm-trans
│   └── store.py      DecisionCache (filesystem, stores raw LLM output)
├── pipeline/
│   ├── phase_a.py    donor identification
│   ├── phase_b.py    cache → LLM → validate → (re-prompt) → EditPlan
│   ├── phase_c.py    apply edits (pass-through for donor==target)
│   ├── edit_plan.py  EditPlan.from_decisions
│   └── orchestrator.py  assemble() — full bin→ASCII→A→B→C→ASCII→bin
├── manifest.py       run report emitter
└── cli.py            `rigforge assemble` + `rigforge inspect`
```

## Key invariants

- **The LLM never sees the bytes it modifies.** Rail L takes a normalized JSON view (`BoneRecord.to_json_record`) and returns a JSON judgment. Rail D applies all edits mechanically.
- **All ASCII edits are byte-range operations.** No regex rewriting; edits are `(start, end, replacement)` triples applied right-to-left so offsets stay valid.
- **`FBXDocument.serialize()` returns source verbatim** when no edits are applied — the lexer round-trips exactly.
- **FBX SDK cluster-connection convention:** `C: "OO", bone_id, cluster_id` (bone = src, cluster = dst). Both directions handled in `sections.extract` for safety, but the SDK always writes the bone-first form.
- **Cache stores raw LLM output, not EditPlan.** EditPlan is regenerated from cache + current avatar registry on every run (registry is mutable, raw output is not).
- **Phase A sees the full bone set** — fingerprinting before slicing for cost reasons would tank Jaccard scores below threshold. Slicing is for LLM-cost smokes only.

## LLM wiring

`OllamaLLMClient` posts to `{host}/api/chat`:
- `Authorization: Bearer <key>` (Ollama Cloud uses the same header as on-prem Ollama)
- Body: `{ model, messages: [system, user, optional-violations], stream: true, format: "json", options: { temperature: 0.0 } }`
- Stdlib `urllib.request` — no httpx dep, no proxy surprises
- Streaming NDJSON: accumulates `message.content` chunks into a single string for `parse_decision_set`. Optional `progress_callback` fires per chunk with cumulative `chunks` / `content_bytes` / `elapsed_seconds` — used by the stress harness for live heartbeat logging (a 200-bone classification can spend 3–10 min in the model; `stream=false` gave dead-silence runs).
- Timeout configured at 600s in `data/ollama.json` because (a) some clothings hit 580s on Flash, and (b) the re-prompt path is a full second call.

Config resolution order: explicit `path` arg → `RIGFORGE_OLLAMA_CONFIG` env var → `data/ollama.json` (gitignored; `data/ollama.example.json` is the template).

## Validators (between LLM and edit)

| # | Rule | Severity |
|---|---|---|
| 1 | unique_role | error |
| 2 | required_roles_present | error |
| 3 | l_r_pairing | error |
| 4 | monotonic_spine_y | error |
| 5 | hierarchy_consistency | error (subsequence match + required ancestors) |
| 6 | drop_safety | error (>10 cluster weights = block) |
| 7 | finger_count_sanity | warn (no fingers in v1 schema) |

Error → one re-prompt with violation text appended → second failure raises.

## Tests + smokes

| Path | Kind |
|---|---|
| `tests/test_*.py` | pytest — 182 tests, ~3m. HTTP mocked, FBX via fixtures. Includes streaming-NDJSON, cross-merge primitives, dedup. |
| `tests/test_e2e.py` | Full pipeline against `MockLLMClient` → binary compare vs Maya.fbx → must be identical. |
| `training/smoke_ollama.py` | Manual. Real Ollama, LLM-only (no Phase C). Spits keep/drop summary + validator output. |
| `training/smoke_full_pipeline.py` | Manual. Real Ollama, full assemble(), fbx_compare vs Maya.fbx (synth-clothing fixture). |
| `training/smoke_cross_merge_mock.py` | Manual. Mock LLM, exercises Phase C pass-through on a real clothing (ClassicChic_Moe → Moe). Should round-trip structurally identical. |
| `training/stress_llm_clothing.py` | Manual. Real Ollama, 5 real Maya-rigged clothings → maya. Live streaming heartbeats; per-clothing manifests + aggregate `_summary.json` + `REPORT.md`. |

`fbx_compare` is the regression gate. Any change that emits an FBX must round-trip through it.

## Current results (2026-05-21)

- **Mock LLM end-to-end (synth clothing → maya):** structurally identical to Maya.fbx.
- **Mock LLM pass-through (ClassicChic_Moe → moe):** structurally identical to input clothing (node_count, total_clusters, total_skin_weights all match to 1 ULP through bin↔ASCII roundtrip).
- **Real DS V4 Flash, 5-clothing stress against Maya:** all 5 pass. 877 total limb bones → 542 keep / 335 drop, with **0 drops at `cluster_weight_count > 100`** across the set (no over-pruning of real skinned bones). Validators pass first-try on every run (no re-prompt needed). All 8 required canonical roles assigned in every run. Phase A donor=maya 0.96–1.00. Latency 3–10 min per clothing; 35 min total wall time.
  - Per-clothing detail + manifests: `data/training/_stress/`.
  - Headline verdict: zero-shot Flash is production-ready for v1 clothing classification. LoRA/Pro A/B deferred unless we ever see a high-weight-drop red flag.
- **Cross-avatar merge (Phase C donor≠target):** implemented + unit-tested as a structural fallback. Not a real production pairing — every clothing in the asset set is already paired with its intended curated avatar.

## Scope beyond v1

### v2 (landed)

- **Bone reparenting** — `reparent_bone_edits` in `ascii_fbx/edits.py`; `Decision.new_parent_role` carries the LLM's request; `hierarchy_consistency` validator follows the post-reparent topology; `EditPlan.reparents` resolves `new_parent_role` → bone id; pass-through `_build_passthrough_edits` applies the edits. LLM prompt documents the optional `new_parent_role` field.
- **Fingers + twist bones in canonical schema** — schema bumped to **v2.0** (cache auto-invalidates because key includes `canonical_schema_version`). 30 finger roles (Thumb/Index/Middle/Ring/Little × 1/2/3 × .L/.R) and 8 twist roles (Upper/LowerArm/Leg.Twist.L/.R), all optional. `finger_count_sanity` validator now checks L/R parity instead of warning on any finger name. `curated_avatars.json` carries finger renames for Maya + Moe.
- **Materials + blendshapes parsers + merge primitives** — `MaterialRecord` / `BlendShapeRecord` / `BlendShapeChannelRecord` extraction in `sections.py`; `merge_materials` / `merge_blendshapes` with name-keyed dedup in `rigforge/sections/merge.py`. Unit-tested. **Not yet wired into Phase C cross-merge** — see below.
- **Frontend scaffold** — `frontend/` Vite + Vue 3 + TypeScript + Pinia + Vue Router + Playwright. Landing page + assemblies store skeleton. Not yet wired to the Python backend.

### v2 (remaining)

- **Materials/blendshapes integration into Phase C cross-merge.** The dedup primitives exist and are unit-tested, but the current wholesale Objects splice doesn't call them. Integration needs id-collision handling (donor material/blendshape ids landing in target's id space after the wholesale splice).
- **Frontend ↔ backend wiring.** FastAPI bridge — endpoints for listing assembly runs + fetching individual manifests. Browser-side fetch from the Pinia store.

### Out of scope — explicitly not problems v2 solves

- **Mesh shape fitting.** Clothing is assumed pre-fit to the target avatar's proportions before it hits RigForge. The tool never touches vertex positions.
- **Non-T-pose input.** Asset pipeline guarantees T-pose.
- **Non-curated donor rigs / wrong-target mismatch.** Phase A hard-fails below threshold; no LLM recovery path for unknown donor families.

### Parked

- Active-learning correction UI + ingest path for the modder collaborator's hand-fix corpus. Useful but slow to build; revisit after v2 ships.
- LoRA training: not viable on DS V4 Flash, and the 5-clothing stress showed zero-shot is good enough. Reopen only if production surfaces a high-weight-drop red flag.
- DS V4 Pro A/B: same reasoning. Zero-shot Flash passed drop sanity, no reason to swap unless real-use data says otherwise.
