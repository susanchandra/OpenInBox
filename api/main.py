"""
api/main.py — CLI entry point for running the OpenInbox server.

This module provides the `main()` function which is registered as a
console script in pyproject.toml under [project.scripts]:

    openinbox-server = "api.main:main"

Running `openinbox-server` after `pip install .` starts the FastAPI
server on 0.0.0.0:7860, the same port used in the Docker deployment.
"""

import uvicorn


def main() -> None:
    """Start the OpenInbox FastAPI server."""
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
    )


if __name__ == "__main__":
    main()
