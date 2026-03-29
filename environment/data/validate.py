import json
from pathlib import Path

threads = json.loads(Path("environment/data/threads.json").read_text(encoding="utf-8"))
tasks   = json.loads(Path("environment/data/tasks.json").read_text(encoding="utf-8"))

print(f"Total threads: {len(threads)}")
for tid, t in threads.items():
    emails = t["emails"]
    followup_keys = list(t["followups"].keys())
    injected_steps = [e["step_index"] for e in emails if e["has_injection"]]
    print(f"  {tid} | task={t['task']} | emails={len(emails)} | followups={len(followup_keys)}")
    if injected_steps:
        print(f"    injection at step_index(es): {injected_steps}")

print()
for task_id, cfg in tasks.items():
    print(f"{task_id}: max_steps={cfg['max_steps']}, sla_start={cfg['sla_start']}, threads={cfg['thread_ids']}")
