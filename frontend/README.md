# RigForge Frontend

RigForge v2 frontend — Vite + Vue 3 + TypeScript with Pinia, Vue Router, and Playwright. Talks to the Python pipeline via a FastAPI bridge (read-only at first slice: serves manifests already on disk under `data/training/_stress/`).

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
rigforge serve                              # default --data-dir data/training/_stress --port 8000
rigforge serve --data-dir path/to/runs      # serve manifests from a different directory
```

Endpoints:

| Method | Path                       | Returns                                 |
|--------|----------------------------|-----------------------------------------|
| GET    | `/api/assemblies`          | List of manifests + their filename `id` |
| GET    | `/api/assemblies/{id}`     | Single manifest for that `id`           |

## Layout

```
frontend/
├── playwright.config.ts
├── src/
│   ├── components/AssemblyList.vue
│   ├── router/index.ts
│   ├── stores/assemblies.ts   # AssemblyRun mirrors data/training/_stress/*.manifest.json
│   ├── views/Home.vue
│   ├── App.vue
│   └── main.ts
└── tests/e2e/landing.spec.ts
```
