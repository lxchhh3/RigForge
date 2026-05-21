# RigForge

Standalone Python tool for MHWilds clothing modding. Takes a Booth-purchased clothing FBX rigged for some other base avatar, retargets it onto a curated avatar, and emits a clean assembly FBX that REE-ModPilot's Phase D pipeline consumes.

The user-facing slice is a compose UI: pick an avatar, add clothing FBX paths, uncheck the meshes and blendshapes you don't want (target's bundled outfit + morphs on one side, unwanted clothing parts on the other), then assemble.

See [PLAN.md](PLAN.md) for the design doc + scope decisions, [ARCHITECTURE.md](ARCHITECTURE.md) for what's actually built, and [frontend/README.md](frontend/README.md) for the FE specifics.

## Quickstart

```sh
# 1. Python deps
pip install -e ".[api]"

# 2. Copy the Ollama config template and fill in your own API key
cp data/ollama.example.json data/ollama.json
# ↑ edit data/ollama.json — it's gitignored so your key never gets committed

# 3. Frontend deps
cd frontend && npm install && cd ..

# 4. Start the FastAPI bridge (terminal 1)
rigforge serve

# 5. Start the Vite dev server (terminal 2)
cd frontend && npm run dev

# 6. Open http://localhost:5173
```

CLI-only path (no FE) is also still supported:

```sh
rigforge assemble --clothing path/to/clothing.fbx --target maya --out out.fbx
```

## Required external toolchain

RigForge shells out to a separate conda env for binary↔ASCII FBX conversion (the FBX SDK install is fragile, so it stays isolated from the main env):

- `fbx_env` conda env with Autodesk FBX SDK 2020.3.9 Python bindings
- Three scripts: `fbx_bin2ascii.py`, `fbx_ascii2bin.py`, `fbx_compare.py`

The user's local paths are hardcoded in `tests/conftest.py` and `rigforge/ascii_fbx/convert.py`. Tests skip cleanly when the toolchain isn't present, so you can develop most of the code without it — you just can't run the conversion-dependent integration tests.

## Tests

```sh
pytest                  # 307 backend tests (~9 min)
cd frontend && npm run test:e2e   # 15 Playwright tests (~2 sec, mocked API)
```

## Repo layout

| Path | What |
|------|------|
| `rigforge/` | Python package — pipeline, ASCII FBX manipulation, FastAPI bridge |
| `frontend/` | Vite + Vue 3 compose UI |
| `data/` | Curated avatar registry, canonical schema, Ollama config |
| `tests/` | Pytest suite |
| `training/` | Smoke + stress scripts (manual; need real Ollama) |
| `PLAN.md` | Design doc — locked decisions, scope split |
| `ARCHITECTURE.md` | "As built" — module map, data flow, what's left |

## Status

v2 is complete. Bone reparenting, fingers/twist schema, materials/blendshapes dedup in Phase C cross-merge, compose UI, mesh-level + blendshape-channel user pre-filter, and the FastAPI bridge are all shipped.

Stress validation: zero-shot DeepSeek V4 Flash classified 877 bones across 5 real Maya-rigged clothings with 0 high-weight drops and no validator re-prompts. Details in `data/training/_stress/REPORT.md` (gitignored — generated locally).
