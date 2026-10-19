"""Shared feasibility guard; no intent labels, optimization or silent repair."""

from copy import deepcopy

from ..metrics import SLICE_LIMITS, audit_state
from ..policies import OriginalAgent


def resource_violations(state: dict) -> list[str]:
    # Placeholder labels only satisfy membership checks. Never judge intent here.
    known = {str(u["user_id"]): name for name, limits in SLICE_LIMITS.items() for u in state[limits["key"]]["users"]}
    return audit_state(state, known, integer_bandwidth=True)["violations"]


class GuardedWirelessAgent(OriginalAgent):
    """Original KB workflow + whole-arrival atomic feasibility check.

    This separately named control must not replace the original reproduction.
    The same final-state check is applied to TKG-DTN plans before commitment.
    """

    def __init__(self, llm):
        super().__init__("wireless_kb", llm)

    def step(self, observation: dict) -> dict:
        before = self.snapshot()
        result = super().step(observation)
        violations = resource_violations(self.snapshot())
        if violations:
            self.module.update_network_state(deepcopy(before))
            return {"status": "rejected_by_common_guard", "slice_type": result.get("slice_type"),
                    "violations": violations, "original_result": result}
        return result | {"common_guard": "passed"}
