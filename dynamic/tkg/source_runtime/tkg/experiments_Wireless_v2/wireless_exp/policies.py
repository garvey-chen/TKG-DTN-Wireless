"""Adapters around copied upstream policies; no dependency on sibling projects."""

import importlib.util
from copy import deepcopy
from typing import Protocol

from . import ROOT
from .data import User
from .metrics import SLICE_LIMITS, empty_state


class Policy(Protocol):
    """Future TKG-DTN integration implements step(observation) and snapshot()."""

    def step(self, observation: dict) -> dict: ...
    def snapshot(self) -> dict: ...


def load_legacy(name: str):
    path = ROOT / "wireless_exp/legacy" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"wireless_legacy_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OriginalAgent:
    def __init__(self, method: str, llm):
        source = {"wireless_kb": "agent_kb", "wireless_nkb": "agent_nkb", "prompt": "prompt"}[method]
        self.module = load_legacy(source)
        self.llm = llm
        if method == "prompt":
            self.module.call_llm = lambda prompt: llm.complete([
                {"role": "system", "content": self.module.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ])
        else:
            self.module.llm = llm
        self.module.reset_network_state()

    def step(self, observation: dict) -> dict:
        """No label is passed to the original workflow."""
        before = self.snapshot()
        failures_before = self.llm.failures
        self.llm.user_id = observation["user_id"]
        try:
            result = self.module.process_user_request(**observation)
            # Some original routines swallow API errors and use a heuristic.
            # Do not present such output as a successful LLM experiment.
            if self.llm.failures > failures_before:
                raise RuntimeError("API transport failed; upstream fallback discarded.")
            if result.get("error"):
                raise RuntimeError("Upstream workflow error: " + result["error"])
        except Exception:
            self.module.update_network_state(before)
            raise
        return result

    def snapshot(self) -> dict:
        return deepcopy(self.module.get_current_network_state())


class OriginalRule:
    """Literal supplied proportional-fair solver + fallback, with oracle labels.

    Each step solves the prefix from scratch, as upstream process_*_users does.
    This deliberately retains upstream infeasible fallbacks; metrics audit them.
    """

    def __init__(self):
        self.module = load_legacy("rule")
        self.users = []
        self.state = empty_state()

    def step_oracle(self, user: User) -> dict:
        import numpy as np
        self.users.append(user)
        self.state = empty_state()
        outcomes = {}
        for slice_name, limits in SLICE_LIMITS.items():
            selected = [u for u in self.users if u.ground_truth == slice_name]
            if not selected:
                continue
            bandwidths, rates, success = self.module.solve_resource_allocation(
                limits["capacity"], len(selected), np.array([u.cqi for u in selected]),
                *limits["bandwidth"], *limits["rate"], 10.0,
            )
            outcomes[slice_name] = bool(success)
            if success:
                allocations = [
                    {"user_id": u.user_id, "cqi": u.cqi, "bandwidth": float(b), "rate": float(rate)}
                    for u, b, rate in zip(selected, bandwidths, rates)
                ]
                part = self.state[limits["key"]]
                part["users"] = allocations
                part["resource_usage"] = sum(a["bandwidth"] for a in allocations)
                part["utilization_rate"] = f'{part["resource_usage"] / limits["capacity"] * 100:.2f}%'
        self.state["total_users"] = sum(len(self.state[v["key"]]["users"]) for v in SLICE_LIMITS.values())
        return {"solver_reported_success": outcomes, "oracle_labels": True}

    def snapshot(self) -> dict:
        return deepcopy(self.state)
