"""
Baseline runner for OpenInbox.

Runs one full episode per task and scores it using the deterministic grader.
The primary baseline uses the OpenAI API. The --fallback flag switches to the
keyword-based agent, which needs no API key.

Usage:
    python baseline/run_baseline.py
    python baseline/run_baseline.py --task task_easy --seed 0
    python baseline/run_baseline.py --task all --seed 0
    python baseline/run_baseline.py --task all --fallback
"""

import argparse
import json
import sys
from pathlib import Path

# Make sure imports work when running this script directly from project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from environment.env import OpenInboxEnv
from environment.graders import GRADERS

_TASKS = ["task_easy", "task_medium", "task_hard"]
_THREADS_PATH = PROJECT_ROOT / "environment" / "data" / "threads.json"


def run_episode(task_id: str, seed: int, agent) -> dict:
    """
    Run one full episode for a given task and seed, then grade it.

    Returns a dict with score, breakdown, steps taken, and thread_id.
    """
    env = OpenInboxEnv()
    obs = env.reset(task_id, seed)

    while not env.done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        if done:
            break

    state = env.state()
    episode_log = state["episode_log"]
    thread_id = env.thread_id

    threads = json.loads(_THREADS_PATH.read_text(encoding="utf-8"))
    ground_truth = threads[thread_id]["ground_truth"]

    grader_result = GRADERS[task_id](episode_log, ground_truth)
    grader_result["steps_taken"] = state["step_count"]
    grader_result["thread_id"] = thread_id
    grader_result["task_id"] = task_id
    return grader_result


def main():
    parser = argparse.ArgumentParser(
        description="Run the OpenInbox baseline agent on one or all tasks."
    )
    parser.add_argument(
        "--task",
        default="all",
        help="Task to run: task_easy | task_medium | task_hard | all  (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for thread selection (default: 0)",
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Use rule-based fallback instead of OpenAI API",
    )
    args = parser.parse_args()

    # Load agent
    if args.fallback:
        from baseline.rule_agent import RuleAgent
        agent = RuleAgent()
        agent_label = "rule-based fallback"
    else:
        from baseline.openai_agent import OpenAIAgent
        # OpenAIAgent.__init__ exits cleanly if API key is missing
        agent = OpenAIAgent(model="gpt-4o-mini")
        agent_label = f"OpenAI ({agent.model}, temperature=0)"

    print(f"Agent: {agent_label}")
    print(f"Seed:  {args.seed}")
    print()

    # Determine which tasks to run
    if args.task == "all":
        tasks_to_run = _TASKS
    elif args.task in _TASKS:
        tasks_to_run = [args.task]
    else:
        print(f"Unknown task: {args.task!r}")
        print(f"Valid options: {_TASKS + ['all']}")
        sys.exit(1)

    results: dict[str, dict] = {}

    for task_id in tasks_to_run:
        print(f"Running {task_id} (seed={args.seed}) ...")
        try:
            result = run_episode(task_id, args.seed, agent)
            results[task_id] = result
            print(f"  score      : {result['score']:.4f}")
            print(f"  breakdown  : {result['breakdown']}")
            print(f"  thread     : {result['thread_id']}")
            print(f"  steps      : {result['steps_taken']}")
        except Exception as exc:
            print(f"  Error: {exc}")
            results[task_id] = {"score": None, "error": str(exc)}
        print()

    # Print summary
    divider = "-" * 42
    print(divider)
    print(f"{'Task':<22} {'Score':>8}")
    print(divider)
    scores = []
    for task_id in tasks_to_run:
        r = results.get(task_id, {})
        score = r.get("score")
        if score is not None:
            scores.append(score)
            print(f"  {task_id:<20} {score:>8.4f}")
        else:
            err = r.get("error", "unknown error")
            print(f"  {task_id:<20} {'ERROR':>8}  ({err})")

    if scores:
        avg = sum(scores) / len(scores)
        print(divider)
        print(f"  {'Average':<20} {avg:>8.4f}")
    print(divider)

    # Save results to baseline/results/
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    suffix = "fallback" if args.fallback else "openai"
    out_path = results_dir / f"scores_{suffix}_seed{args.seed}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
