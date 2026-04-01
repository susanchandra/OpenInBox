"""
server/app.py — server entrypoint for the OpenInbox environment.

This module:
  - exposes the FastAPI application at module level as `app`
  - defines a callable `main()` function that starts the server
  - supports direct execution and installed script entrypoints

Registered in pyproject.toml as:
    [project.scripts]
    openinbox-server = "server.app:main"

The FastAPI application logic lives in api/app.py.
This module is a thin runner that keeps concerns separated.
"""

import uvicorn

from api.app import app  # noqa: F401 — re-exported for compatibility

__all__ = ["app", "main"]


def main() -> None:
    """Start the OpenInbox FastAPI server on port 7860."""
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
    )


if __name__ == "__main__":
    main()
