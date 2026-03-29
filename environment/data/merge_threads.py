"""
Merges threads_easy_medium.json and threads_hard.json into a single threads.json file.
Run this once during setup or whenever the source files change.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent


def merge():
    easy_medium = json.loads((HERE / "threads_easy_medium.json").read_text(encoding="utf-8"))
    hard = json.loads((HERE / "threads_hard.json").read_text(encoding="utf-8"))

    merged = {**easy_medium, **hard}

    out_path = HERE / "threads.json"
    out_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(merged)} threads to {out_path}")


if __name__ == "__main__":
    merge()
