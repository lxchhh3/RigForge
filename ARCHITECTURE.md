# RigForge — Architecture (as built)

Snapshot of the implemented pipeline. PLAN.md is the design doc; this file describes what actually exists in the tree and how the pieces fit together. v1 was the CLI-only release; v2 added a FastAPI bridge + Vite/Vue compose UI driven entirely off mesh-level user decisions. Updated 2026-05-21.

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
    │           │   emits {role, verdict, drop_category?, new_parent_role?}
    │           │   per bone — NEVER emits names. Naming is a deterministic
    │           │   lookup against target_avatar.canonical_to_name[role].
    │           ▼
    │       validators (7 rules)
    │           │       │
    │           │ errs  │ clean
    │           ▼       ▼
    │       re-prompt   EditPlan.from_decisions(...)
    │       (once)        drops   = {bone_id : verdict == "drop"}
    │                     renames = {bone_id : new_name} (if it differs; canonical→target
    │                               name, non-canonical ride-along→English name_en translation)
    │                     reparents = {bone_id : target_bone_id_for(new_parent_role)}
    │
    ├──► Phase C: merge onto target. Apply drops; apply renames to the
    │             ride-along (non-canonical) bones (canonical bones are stripped —
    │             the target owns their names — so those renames + reparents
    │             are moot); in cross-avatar, also dedup materials/blendshapes
    │             by name, apply user mesh/channel drops + DeformPercent
    │             overrides; translate JP/KR morph (blendshape channel) + mesh +
    │             mesh-data names to English via the dictionary (clothing always;
    │             target behind translate_target_morphs); splice into target's
    │             Objects + Connections.
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
│   ├── sections.py   3-pass extract → Models + Geometries + Skins + Clusters
│   │                 + materials/blendshapes + cluster_skin/geometry_owner_model
│   │                 indexes + bone_to_mesh_names() affinity helper
│   └── edits.py      TextEdit primitives:
│                       drop_bone_edits, rename_bone_edits, reparent_bone_edits,
│                       drop_cluster_edits, drop_mesh_edits,
│                       drop_blend_shape_channel_edits,
│                       set_blend_shape_channel_deform_percent_edits,
│                       rename_blend_shape_channel_edits, rename_mesh_edits,
│                       rename_geometry_edits (all v2)
├── naming.py        dictionary JP/KR→EN name translation (no LLM):
│                     load_translation_table() + translate_name() — exact match,
│                     then peel laterality (_L/_R/右/左) + index tails (incl.
│                     full-width digits) and recompose. Backs the morph/mesh
│                     translation step in Phase C. Table: data/name_translation.json
├── canonical/
│   ├── schema.py     CanonicalSchema (v2.2: core/arms/legs + fingers + twist
│   │                 + per-bone name_en English translation)
│   ├── decisions.py  Decision / DecisionSet (incl. new_parent_role)
│   └── validators.py 7 rules + driver (user_dropped exempt from drop_safety)
├── avatars/
│   ├── fingerprint.py  normalize + Jaccard
│   └── registry.py     load + identify_donor (threshold 0.85);
│                       CuratedAvatar.load_ascii_view() caches a SectionView
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
│   ├── phase_b.py    cache → LLM → validate → (re-prompt) → EditPlan;
│   │                 applies user_drop_bone_ids pre-filter with subtree cascade
│   ├── phase_c.py    apply edits (pass-through and cross-merge);
│   │                 applies drop_mesh_ids + drop_blend_shape_channel_ids
│   │                 + blend_shape_channel_overrides (DeformPercent) in
│   │                 both branches; target_* counterparts in cross-merge only;
│   │                 materials/blendshapes dedup-and-repoint in cross-merge;
│   │                 _translate_morph_and_mesh_edits (morph/mesh/mesh-data
│   │                 names → EN); name-based repoint has a rename-target
│   │                 fallback so a misnamed canonical bone strips, not doubles
│   ├── edit_plan.py  EditPlan.from_decisions (drops + renames + reparents)
│   └── orchestrator.py  assemble() — full bin→ASCII→A→B→C→ASCII→bin;
│                     emits progress_cb(phase, note) at each phase boundary
├── sections/
│   └── merge.py      merge_materials / merge_blendshapes (donor→target name
│                     dedup primitives; Phase C cross-merge takes the inverse
│                     strip-and-repoint path — see "Materials/BlendShapes dedup")
├── api/
│   └── server.py     FastAPI app factory; the v2 bridge between Python
│                     pipeline and the Vite/Vue compose UI. See "API surface".
├── manifest.py       run report emitter
└── cli.py            `rigforge assemble` + `rigforge inspect` + `rigforge serve`
```

## API surface (v2)

`rigforge serve` builds the FastAPI app via `create_app(...)` in `rigforge/api/server.py` and binds it to the Vite dev server's expected origin (CORS allows `localhost:5173` + `127.0.0.1:5173`).

| Method | Path                              | Purpose |
|--------|-----------------------------------|---------|
| GET    | `/api/avatars`                    | List curated avatars + their canonical roles |
| GET    | `/api/avatars/{id}/inspect`       | Parse a curated avatar's ASCII FBX; return bone records (typed: LimbNode / Mesh / Null) for the FE outliner |
| POST   | `/api/clothings/inspect`          | Same shape as avatar inspect, but for a clothing path. Auto-converts binary FBX through `fbx_env` and caches the ASCII under `data/_inspect_cache/` |
| POST   | `/api/assemble`                   | Run the full pipeline with optional user pre-filters: `drop_bone_ids` / `target_drop_bone_ids` (advanced) and `drop_mesh_ids` / `target_drop_mesh_ids` (what the FE sends) |
| GET    | `/api/assemblies`                 | List recent manifests in `data_dir` |
| GET    | `/api/assemblies/{id}`            | Fetch one manifest by stem |

Inspect returns the same shape for clothings and avatars so the FE renders both through the same `MeshList.vue` component. Bone records include `type_class` (so the FE can filter for `"Mesh"`) and `deforms_meshes` (the bone-affinity helper from sections.py) so power-user tooling can show "this bone drives meshes X, Y, Z."

## Compose flow (v2, what the modder sees)

The FE (`frontend/`) is the user-facing slice. Modder workflow:

```
Compose.vue
├── Avatar picker  ─────► GET /api/avatars
├── TargetPanel    ─────► GET /api/avatars/{id}/inspect
│   └── MeshList: armature row (no checkbox) + parallel mesh names with checkboxes.
│       Unchecking strips that mesh from the TARGET before splice
│       (e.g. Maya ships with Cloth/Shoes/Hat — strip them so new clothing sits clean).
├── Add clothing path ─► POST /api/clothings/inspect
│   └── ClothingItem per added clothing
│       └── MeshList: same outliner shape; unchecking strips that mesh from the clothing.
└── Assemble button ───► POST /api/assemble
    └── Sends drop_mesh_ids (per clothing) + target_drop_mesh_ids (shared across batch).
        Pipeline returns the merged binary FBX path + manifest.
```

The modder never sees bones. Bone-level drops still exist in the API for power use, but the UI surfaces only mesh model_ids — Maya's rigging convention puts every bone in every mesh's skin, so a bone-level UI lies about what a checkbox actually does. Mesh ids are the honest unit.

`drop_mesh_edits` (in `ascii_fbx/edits.py`) removes a mesh as one consistent chain: Mesh-Model + Geometry + Skin + every Cluster of that Skin + all referencing connections. Bones are left in place (they're shared across meshes; cascading bone cleanup would corrupt siblings).

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
| `tests/test_*.py` | pytest — 394 tests (~13m for the full sweep; the cost is fbx_env bin↔ASCII conversion of the 40MB Maya, not the LLM, which is mocked — most files run in seconds). HTTP mocked, FBX via fixtures. Covers lexer, sections + affinity, edits (incl. drop_mesh + drop/rename blend_shape_channel + rename_mesh/geometry + set_blend_shape_channel_deform_percent), dictionary name translation (`test_naming`, `test_morph_mesh_translation`), validators, cache, merge primitives, Phase A/B/C (incl. cross-merge materials/blendshapes dedup, channel drops, DeformPercent overrides, morph/mesh translation), the API endpoints, the LLM client + streaming NDJSON, and end-to-end on real Maya.fbx. |
| `tests/test_e2e.py` | Full pipeline against `MockLLMClient`: a PARTIAL synth clothing (one garment mesh, renamed distinct) merged onto maya, then a binary→ASCII round-trip asserts the merge (garment added distinctly, target meshes survive, skeleton repointed not doubled). Reworked from the old byte-identity-to-Maya.fbx check, which the always-merge pivot made obsolete. |
| `tests/test_api.py` | TestClient against the FastAPI app: avatar/clothing inspect, assemble endpoint, mesh + bone drop forwarding, translate_target_morphs flag. |
| `frontend/tests/e2e/*.spec.ts` | Playwright — 18 specs covering compose flow (avatar pick, clothing inspect, mesh-drop UX, target strip, blendshape-channel drop UX, blendshape DeformPercent slider, target morph-translation checkbox, live progress + error events, history view). Auto-starts the dev server and mocks the API. |
| `training/smoke_ollama.py` | Manual. Real Ollama, LLM-only (no Phase C). Spits keep/drop summary + validator output. |
| `training/smoke_full_pipeline.py` | Manual. Real Ollama, full assemble(), fbx_compare vs Maya.fbx (synth-clothing fixture). |
| `training/smoke_cross_merge_mock.py` | Manual. Mock LLM, exercises Phase C pass-through on a real clothing (ClassicChic_Moe → Moe). Should round-trip structurally identical. |
| `training/stress_llm_clothing.py` | Manual. Real Ollama, 5 real Maya-rigged clothings → maya. Live streaming heartbeats; per-clothing manifests + aggregate `_summary.json` + `REPORT.md`. |

`fbx_compare` is the regression gate. Any change that emits an FBX must round-trip through it.

## Current results (2026-05-21)

- **Mock LLM end-to-end (partial synth clothing → maya):** garment mesh added distinctly, target meshes survive, clothing skeleton repointed onto the target (no duplicated bones), binary round-trips cleanly. (Was "structurally identical to Maya.fbx" pre-always-merge; see `test_e2e.py`.)
- **Mock LLM pass-through (ClassicChic_Moe → moe):** structurally identical to input clothing (node_count, total_clusters, total_skin_weights all match to 1 ULP through bin↔ASCII roundtrip).
- **Real DS V4 Flash, 5-clothing stress against Maya:** all 5 pass. 877 total limb bones → 542 keep / 335 drop, with **0 drops at `cluster_weight_count > 100`** across the set (no over-pruning of real skinned bones). Validators pass first-try on every run (no re-prompt needed). All 8 required canonical roles assigned in every run. Phase A donor=maya 0.96–1.00. Latency 3–10 min per clothing; 35 min total wall time.
  - Per-clothing detail + manifests: `data/training/_stress/`.
  - Headline verdict: zero-shot Flash is production-ready for v1 clothing classification. LoRA/Pro A/B deferred unless we ever see a high-weight-drop red flag.
- **Cross-avatar merge (Phase C donor≠target):** implemented + unit-tested as a structural fallback. Not a real production pairing — every clothing in the asset set is already paired with its intended curated avatar.

## Scope beyond v1

### v2 (landed)

- **Bone reparenting** — `reparent_bone_edits` in `ascii_fbx/edits.py`; `Decision.new_parent_role` carries the LLM's request; `hierarchy_consistency` validator follows the post-reparent topology; `EditPlan.reparents` resolves `new_parent_role` → bone id. Since the always-merge pivot (`324cd02`) the old pass-through `_build_passthrough_edits` path is gone: reparents are moot for name-matched/canonical bones (they're stripped, the target's topology wins) — the same way canonical renames are moot. LLM prompt documents the optional `new_parent_role` field.
- **Fingers + twist bones in canonical schema** — schema bumped to **v2.1** (cache auto-invalidates because key includes `canonical_schema_version`). 30 finger roles (Thumb/Index/Middle/Ring/Little × 1/2/3 × .L/.R) and 8 twist roles (Upper/LowerArm/Leg.Twist.L/.R), all optional. `finger_count_sanity` validator now checks L/R parity instead of warning on any finger name. `curated_avatars.json` carries finger renames for Maya + Moe.
- **Materials + blendshapes parsers + merge primitives** — `MaterialRecord` / `BlendShapeRecord` / `BlendShapeChannelRecord` extraction in `sections.py`; `merge_materials` / `merge_blendshapes` with name-keyed dedup in `rigforge/sections/merge.py`. Unit-tested.
- **Materials/BlendShapes dedup in Phase C cross-merge** — `_compute_dedup_repoint` in `pipeline/phase_c.py` strips donor materials and BlendShape Deformers + channels whose short name collides with the target, and adds donor→target id mappings to `repoint_table` so `id_offset_edits` rewrites surviving clothing-side connections (e.g. Geometry→Material, BlendShape→Geometry) to the target's existing ids. Symmetric with how kept-bone repointing works.
- **FastAPI bridge** — `rigforge/api/server.py`, started via `rigforge serve`. Avatars + inspect + assemble + manifest list. CORS for the Vite dev origin. Auto-converts binary FBXs via `fbx_env` for the inspect endpoints, with on-disk caching under `data/_inspect_cache/`.
- **Compose UI** — `frontend/` Vue 3 / Pinia / Vue Router / Playwright. Modder picks an avatar, adds clothing FBX paths, unchecks meshes in a Blender-outliner-style list (armature header + parallel mesh checkboxes), assembles. Bones are never surfaced.
- **Mesh-level user pre-filter** — `drop_mesh_edits` primitive cleanly strips Mesh-Model + Geometry + Skin + Clusters as one unit. Endpoint accepts `drop_mesh_ids` + `target_drop_mesh_ids`; Phase C applies them in both pass-through and cross-merge.
- **Blendshape-channel user pre-filter** — `drop_blend_shape_channel_edits` primitive strips a BlendShapeChannel SubDeformer + connections referencing it (owning BlendShape Deformer preserved — may hold surviving siblings). Inspect surfaces `blend_shape_channels[]`; assemble accepts `drop_blend_shape_channel_ids` + `target_drop_blend_shape_channel_ids`; FE renders a flat per-channel checkbox list with a name filter (channels number in the hundreds on real avatars).
- **DeformPercent override (blendshape slider)** — `set_blend_shape_channel_deform_percent_edits` primitive rewrites just the numeric `DeformPercent` token on a BlendShapeChannel (matches Maya's int-vs-%g formatting so diffs stay clean). Inspect returns each channel's on-disk `deform_percent` so the FE slider opens at the file's actual baseline. Assemble accepts `blend_shape_channel_overrides` + `target_blend_shape_channel_overrides` (list of `{channel_id, deform_percent}`). Channels in the corresponding drop set are silently skipped — drop wins. FE: 0-100 range slider beside each channel checkbox; snapping back to the on-disk value clears the override (keeps payloads clean); current value highlights yellow when overridden.
- **JP/KR→EN name translation** — makes non-English Booth assets readable. Two halves: (1) **bones** — the LLM emits a per-bone `name_en` (schema **v2.2**); `EditPlan` renames ride-along non-canonical bones to it. (2) **morphs + meshes** — `rigforge/naming.py` does a deterministic, offline dictionary lookup (`data/name_translation.json`, no LLM) for blendshape-channel, mesh (Model), and mesh-DATA (Geometry) names; `phase_c._translate_morph_and_mesh_edits` applies the renames via `rename_blend_shape_channel_edits` / `rename_mesh_edits` / `rename_geometry_edits`. `translate_name` peels laterality (`_L`/`_R`/`右`/`左`) + index tails (incl. full-width digits) and a bare trailing `L`/`R`, with an NFKC fallback for half-width katakana. Collisions are allowed (a smile spanning mouth+eyes+cheeks can legitimately share a name — no forced uniqueness). Safe because FBX wires by object id, not name. Clothing always translates; the target (base avatar) is behind `translate_target_morphs` (FE checkbox in `TargetPanel.vue`, default on) for the rare downstream-Unity-binds-by-name case. Coverage on Maya's morph set: 142/147; the rest are symbol/Greek mouth visemes left for human labels.
- **Live assemble progress** — `assemble()` in `pipeline/orchestrator.py` takes an optional `progress_cb(phase, note)` invoked at phase boundaries. New `POST /api/assemble/stream` runs the pipeline on a worker thread and emits NDJSON events (`started` / `progress` / `heartbeat` / `done` / `error`) for the FE to read line-by-line. FE's `AssembleProgressList.vue` shows a 4-phase bar + most recent note per clothing in flight. Sync `POST /api/assemble` kept for CLI/scripted callers.
- **Bone affinity helper** — `bone_to_mesh_names(view, bone_id)` walks the bone subtree (chain-root case) and the cluster→skin→geometry→model chain, filtering out the zero-weight clusters that Maya's rigging convention inserts. The inspect endpoint returns this per bone so power-user tooling has the right `bone → meshes` map.
- **`user_drop_bone_ids` pre-filter in Phase B** — forces verdict=drop on user-checked bones and cascades to their subtree; the drop_safety validator skips `drop_category="user_dropped"` so high-weight user drops don't error.

### v2 (remaining)

(none — v2 is complete.)

### Out of scope — explicitly not problems v2 solves

- **Mesh shape fitting.** Clothing is assumed pre-fit to the target avatar's proportions before it hits RigForge. The tool never touches vertex positions.
- **Non-T-pose input.** Asset pipeline guarantees T-pose.
- **Non-curated donor rigs / wrong-target mismatch.** Phase A hard-fails below threshold; no LLM recovery path for unknown donor families.

### Parked

- Active-learning correction UI + ingest path for the modder collaborator's hand-fix corpus. Useful but slow to build; revisit after v2 ships.
- LoRA training: not viable on DS V4 Flash, and the 5-clothing stress showed zero-shot is good enough. Reopen only if production surfaces a high-weight-drop red flag.
- DS V4 Pro A/B: same reasoning. Zero-shot Flash passed drop sanity, no reason to swap unless real-use data says otherwise.
