"""
server/app.py — compatibility shim for tools that expect the FastAPI
application to live at server/app.py.

OpenEnv tooling and some deployment runners look for the app object at
this path. This module re-exports the application from its canonical
location (api/app.py) so that both paths work without duplicating logic.

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 7860

The canonical Docker deployment uses:
    uvicorn api.app:app --host 0.0.0.0 --port 7860
"""

from api.app import app  # noqa: F401 — re-exported for compatibility
