"""Full environment smoke test — validates all threads across all tasks."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from environment.env import OpenInboxEnv
from environment.graders import GRADERS
from environment.models import Action

threads = json.loads(Path('environment/data/threads.json').read_text(encoding='utf-8'))
tasks   = json.loads(Path('environment/data/tasks.json').read_text(encoding='utf-8'))

print('=== FULL ENVIRONMENT SMOKE TEST ===')
print()

all_pass = True

for task_id, task_cfg in tasks.items():
    print(f'--- {task_id} ---')
    for seed, thread_id in enumerate(task_cfg['thread_ids']):
        try:
            env = OpenInboxEnv()
            obs = env.reset(task_id, seed)

            gt = threads[thread_id]['ground_truth']

            if task_id == 'task_hard':
                cls0 = gt['classifications'][0]
                rte0 = gt['routes'][0]
                pri0 = 'high'
            else:
                cls0 = gt['classification']
                rte0 = gt['route_to']
                pri0 = gt.get('priority') or 'medium'

            action = Action(
                classification=cls0,
                priority=pri0,
                route_to=rte0,
                extracted_fields=gt.get('extracted_fields') or {},
                escalate=False,
                flag_injection=False,
            )

            obs2, reward, done, info = env.step(action)
            state = env.state()
            ep_log = state['episode_log']
            gt_data = threads[thread_id]['ground_truth']
            result = GRADERS[task_id](ep_log, gt_data)
            score = result['score']
            status = 'OK'
            print(f'  seed={seed} thread={thread_id} reward={reward:.4f} grader_score={score:.4f} [{status}]')
        except Exception as e:
            print(f'  seed={seed} thread={thread_id} ERROR: {e}')
            all_pass = False
    print()

print('=== RESULT:', 'ALL PASS' if all_pass else 'FAILURES DETECTED', '===')
