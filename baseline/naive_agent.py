"""
Naive fixed-policy baseline agent for OpenInbox.

This agent always makes the same decision regardless of the email content:
  classification = "billing"
  priority       = "medium"
  route_to       = "billing_team"

It exists to establish a lower bound on task performance and to demonstrate
the gap between random/fixed policies and LLM-driven reasoning. A meaningful
environment should assign near-zero scores to this agent on all but the
simplest tasks.

Activate with:  python baseline/run_baseline.py --task all --naive
"""

from environment.models import Action, Observation


class NaiveAgent:
    """
    Fixed-policy agent. Returns the same action for every observation.
    No state, no API calls, fully deterministic.
    """

    FIXED_ACTION = Action(
        classification="billing",
        priority="medium",
        route_to="billing_team",
        extracted_fields={},
        escalate=False,
        flag_injection=False,
        reply_draft=None,
    )

    def act(self, obs: Observation) -> Action:  # noqa: ARG002
        return self.FIXED_ACTION
