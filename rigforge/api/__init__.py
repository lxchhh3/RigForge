"""FastAPI bridge between the RigForge Python pipeline and the Vite + Vue 3
frontend in `frontend/`.

The bridge is read-only at first slice: it serves manifests already on disk
under `data/training/_stress/`. A future slice will add an endpoint to kick
off a fresh assemble run.
"""
from .server import create_app

__all__ = ["create_app"]
