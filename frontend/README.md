# RigForge Frontend

RigForge v2 compose UI — Vite + Vue 3 + TypeScript with Pinia, Vue Router, and Playwright. Talks to the Python pipeline via a FastAPI bridge.

The modder's workflow: pick a target avatar, paste clothing FBX paths, uncheck the meshes (Blender-outliner view) and the blendshape channels they don't want, then assemble. Bones never surface in the UI — the units of user agency are the mesh and the blendshape channel.

## Scripts

```sh
npm install         # install deps
npm run dev         # start dev server on http://localhost:5173
npm run build       # type-check + production build into dist/
npm run test:e2e    # run Playwright E2E suite (auto-starts the dev server; mocks the API)
```

## Backend

The FE expects the FastAPI bridge at `http://localhost:8000` by default. Override with `VITE_API_BASE`. Start it from the RigForge project root:

```sh
pip install -e ".[api]"
rigforge serve                              # default --port 8000
rigforge serve --data-dir path/to/runs      # serve manifests from a different directory
rigforge serve --llm-mock                   # use the mock LLM client (no Ollama)
rigforge serve --llm-ollama                 # use the real Ollama client (default)
```

### Endpoints

| Method | Path                            | Returns                                                                                  |
|--------|---------------------------------|------------------------------------------------------------------------------------------|
| GET    | `/api/avatars`                  | List of curated avatars + their canonical roles                                          |
| GET    | `/api/avatars/{id}/inspect`     | Bone records for the target avatar (FE renders Mesh-type ones in the outliner)           |
| POST   | `/api/clothings/inspect`        | Body `{path}` — bone records for a clothing FBX. Auto-converts binary FBX via `fbx_env`. |
| POST   | `/api/assemble`                 | Synchronous variant — runs the pipeline, blocks until done, returns the final `{id, output_fbx, manifest}` payload. Useful for CLI / scripted callers. |
| POST   | `/api/assemble/stream`          | Same request body, streams NDJSON progress events (`{event: "progress" \| "started" \| "heartbeat" \| "done" \| "error"}`) so the FE can render a live progress bar + per-phase status. **What the compose UI uses.** |
| GET    | `/api/assemblies`               | List of recent manifests in `--data-dir`                                                 |
| GET    | `/api/assemblies/{id}`          | Single manifest                                                                          |

`drop_mesh_ids` / `target_drop_mesh_ids` are arrays of mesh-Model ids (returned by the inspect endpoints as `model_id` on `type_class === "Mesh"` records). The pipeline strips each mesh as a unit (Mesh-Model + Geometry + Skin + Clusters).

`drop_blend_shape_channel_ids` / `target_drop_blend_shape_channel_ids` are arrays of channel SubDeformer ids (returned by inspect as `channel_id` on each entry of `blend_shape_channels`). The pipeline strips the channel node + connections referencing it; the owning BlendShape Deformer is preserved (it may carry other surviving channels).

## Compose flow

```
Compose.vue (/)
├── Avatar picker        ──► GET /api/avatars
├── TargetPanel          ──► GET /api/avatars/{id}/inspect
│   ├── MeshList: armature header + parallel mesh checkboxes (strip target's bundled outfit)
│   └── BlendShapeList: per-channel checkboxes + name filter (strip target's bundled morphs)
├── Add clothing path    ──► POST /api/clothings/inspect
│   └── ClothingItem (one per added clothing)
│       ├── MeshList: same outliner shape (strip unwanted clothing meshes)
│       └── BlendShapeList: per-channel checkboxes (strip unwanted clothing morphs)
└── Assemble             ──► POST /api/assemble/stream per clothing (NDJSON)
    ├── AssembleProgressList: per-clothing phase bar + most recent note
    └── Results panel: id + output_fbx path (final 'done' event)

History.vue (/history)
└── AssemblyList: GET /api/assemblies, shows recent runs from manifests on disk
```

## Layout

```
frontend/
├── playwright.config.ts
├── vite.config.ts
├── src/
│   ├── components/
│   │   ├── MeshList.vue       # the outliner — armature + mesh checkboxes
│   │   ├── BlendShapeList.vue # flat per-channel checkboxes + name filter
│   │   ├── TargetPanel.vue    # target avatar's mesh + blendshape list, top of compose page
│   │   ├── ClothingItem.vue   # one added clothing — header + mesh + blendshape list
│   │   ├── AssembleProgressList.vue # live phase bar + note per in-flight clothing
│   │   └── AssemblyList.vue   # recent runs (history page)
│   ├── stores/
│   │   ├── avatars.ts         # GET /api/avatars
│   │   ├── compose.ts         # target state + clothings[] + assembleAll()
│   │   └── assemblies.ts      # recent manifests for the history page
│   ├── views/
│   │   ├── Compose.vue        # / — the v2 modder UI
│   │   └── History.vue        # /history — read-only recent runs
│   ├── router/index.ts
│   ├── App.vue
│   └── main.ts
└── tests/e2e/
    ├── compose.spec.ts        # avatar pick, clothing inspect, mesh-drop UX, target strip, blendshape-drop UX
    └── landing.spec.ts        # history view + recent-runs panel
```

## Testing

```sh
npm run test:e2e        # 15 Playwright tests, auto-starts dev server + mocks the API
npx vue-tsc --noEmit    # type-check
```

Tests mock the API (no backend required). For end-to-end against the real backend, run `rigforge serve` then `npm run dev` and exercise the UI manually.
