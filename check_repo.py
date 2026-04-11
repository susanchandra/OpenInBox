from huggingface_hub import HfApi

api = HfApi()
files = list(api.list_repo_files(repo_id="susannnnn/OpenInBox", repo_type="space"))

key = [
    "Dockerfile", "inference.py", "openenv.yaml", "README.md",
    "requirements.txt", "api/app.py", "environment/env.py",
    "environment/graders/task1.py", "environment/graders/task2.py",
    "environment/graders/task3.py",
]
bad = [".env", "inference_results.json"]

print("Key files in HF Space repo:")
for k in key:
    status = "OK" if k in files else "MISSING"
    print(f"  {status}  {k}")

print()
print("Files that must NOT be present:")
for b in bad:
    status = "FOUND - PROBLEM" if b in files else "not present - OK"
    print(f"  {status}  {b}")

print()
print(f"Total files in Space: {len(files)}")
