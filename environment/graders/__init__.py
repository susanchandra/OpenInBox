"""
Grader package for OpenInbox.

Maps task IDs to their grader functions. Each grader takes
episode_log and ground_truth and returns a score dict.
"""

from environment.graders.task1 import grade as grade_task1
from environment.graders.task2 import grade as grade_task2
from environment.graders.task3 import grade as grade_task3

# Maps task_id -> grader function.
# Used by the /grader API endpoint to dispatch to the right grader.
GRADERS: dict[str, callable] = {
    "task_easy":   grade_task1,
    "task_medium": grade_task2,
    "task_hard":   grade_task3,
}


def grade(task_id: str, episode_log: list[dict], ground_truth: dict) -> dict:
    """
    Grade an episode for the given task.

    Returns {"score": float, "breakdown": dict}.
    Raises ValueError if task_id is not recognised.
    """
    if task_id not in GRADERS:
        raise ValueError(
            f"Unknown task_id: {task_id!r}. Valid options: {list(GRADERS)}"
        )
    return GRADERS[task_id](episode_log, ground_truth)
