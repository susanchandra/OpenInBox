import random
from environment.env import OpenInboxEnv
from environment.models import Action

env = OpenInboxEnv()

for i in range(10):
    obs = env.reset('task_hard', seed=i)
    while not env.done:
        action = Action(
            classification=random.choice(['billing', 'technical', 'hr', 'legal', 'spam', 'unknown']),
            priority=random.choice(['low', 'medium', 'high', 'critical']),
            route_to=random.choice(['handle_self', 'delegate_fast', 'delegate_thorough', 'wait', 'escalate']),
            flag_injection=random.choice([True, False])
        )
        obs, reward, done, info = env.step(action)
        print(f"Step {env.step_count} total={reward} breakdown: {info.get('reward_breakdown')}")
    if "terminal_reward" in env.episode_log[-1]:
        print(f"Terminal Reward at episode {i}: {env.episode_log[-1]['terminal_reward']}\n")
