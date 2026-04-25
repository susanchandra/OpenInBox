"""
deploy_to_hf.py — uploads the project to the HF Space using stored HF credentials.

Run AFTER you have done:  huggingface-cli login

This uploads all project files, skipping cache/results/OS junk.
It does NOT require git. It uses the HF Hub API directly.
"""

from huggingface_hub import HfApi, whoami

REPO_ID = "susannnnn/OpenInBox"
REPO_TYPE = "space"

IGNORE_PATTERNS = [
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".pytest_cache",
    ".git",
    ".venv",
    "venv",
    "baseline/results",
    "inference_results.json",
    "*.json.bak",
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",
    # deployment script itself — not needed in the Space
    "deploy_to_hf.py",
]

def main():
    # Confirm the logged-in user before uploading
    try:
        user = whoami()
        print(f"Logged in as: {user['name']}")
    except Exception:
        print("ERROR: Not logged in. Run:  huggingface-cli login")
        raise SystemExit(1)

    print(f"Uploading to: https://huggingface.co/spaces/{REPO_ID}")
    print("This may take 1-2 minutes...")

    api = HfApi()
    api.upload_folder(
        folder_path=".",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        ignore_patterns=IGNORE_PATTERNS,
        commit_message="Final OpenInbox submission — 93/93 tests, 5 API endpoints",
    )

    print()
    print("Upload complete.")
    print(f"Space: https://huggingface.co/spaces/{REPO_ID}")
    print("Check build logs at the URL above — click the Logs tab.")
    print("Build takes 2-4 minutes. Space turns green when ready.")

if __name__ == "__main__":
    main()
