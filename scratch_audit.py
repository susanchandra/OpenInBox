import sys; sys.path.insert(0, '.')

# Phase 0
with open('api/app.py') as f:
    api = f.read()
endpoints = ['/tasks', '/reset', '/step', '/grader']
p0 = all(ep in api for ep in endpoints)
print(f'PHASE 0  all endpoints present: {p0}')

# Phase 1A
from environment.env import OpenInboxEnv
from environment.models import Action

env = OpenInboxEnv()
obs = env.reset('task_easy', seed=0)
dh = obs.delegation_history
required_1a = {'team_A_recent', 'team_B_recent', 'team_A_uses', 'team_B_uses', 'escalate_uses'}
present_keys = set(dh.keys())
has_required = required_1a.issubset(present_keys)
has_profiles  = 'team_profiles' in obs.model_dump()
print(f'PHASE 1A  required keys present: {has_required}  team_profiles leaked: {has_profiles}')
print(f'          dh keys: {sorted(present_keys)}')

# Phase 1B
has_internal  = hasattr(env, 'internal_reliability')
has_old_attr  = hasattr(env, 'team_reliability')
env2 = OpenInboxEnv()
env2.reset('task_hard', seed=42)
for _ in range(7):
    a = Action(classification='billing', priority='medium', route_to='delegate_fast',
               extracted_fields={}, escalate=False, flag_injection=False, reply_draft=None)
    if env2.done: break
    env2.step(a)
r_after_7 = env2.internal_reliability.get('delegate_fast')
print(f'PHASE 1B  internal_reliability exists: {has_internal}  old team_reliability gone: {not has_old_attr}')
print(f'          delegate_fast after step-7 drift: {r_after_7} (should differ from 1.0)')

# Phase 1C
import json
with open('environment/data/threads.json') as f:
    threads = json.load(f)
enabled = sum(1 for t in threads.values() if t.get('auto_resolve', {}).get('enabled', False))
pct = enabled / len(threads) * 100
auto_in_obs = 'auto_resolve' in str(obs.model_dump())
print(f'PHASE 1C  auto_resolve ~30%: {pct:.1f}% ({enabled}/{len(threads)})  leaked_in_obs: {auto_in_obs}')

# Phase 1D
from environment.reward import compute, terminal_outcome_reward
a = Action(classification='billing', priority='medium', route_to='delegate_fast',
           extracted_fields={}, escalate=False, flag_injection=False, reply_draft=None)
r = compute(a, None, {}, False, False, 'task_easy')
rb = r.model_dump()
banned = {'classification_reward', 'extraction_reward', 'priority_reward'}
banned_gone = not any(k in rb for k in banned)
has_terminal = callable(terminal_outcome_reward)
sla_urgency_present = 'sla_urgency' in rb
sla_bonus_gone = 'sla_bonus' not in rb
computed_total = round(rb['budget_penalty'] + rb['sla_urgency'] + rb['cascade_penalty'] + rb['repeat_penalty'], 4)
total_correct  = abs(computed_total - rb['total']) < 1e-4
print(f'PHASE 1D  banned signals removed: {banned_gone}')
print(f'          sla_urgency field present: {sla_urgency_present}  sla_bonus gone: {sla_bonus_gone}')
print(f'          total = only 4 signals: {total_correct}')
print(f'          terminal_outcome_reward callable: {has_terminal}')
print(f'          reward keys: {sorted(rb.keys())}')
