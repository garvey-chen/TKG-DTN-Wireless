"""Exact integer resource calculations using current observations only.

No future users, labels, LLM calls, evictions or slice changes. Given a fixed
admission set, each rate is linear in bandwidth; satisfy minima then distribute
remaining integer MHz in descending marginal-rate order. This is exact under
the existing independent-user model, not a general radio/interference optimizer.
"""

import math
from copy import deepcopy

from ..metrics import SLICE_LIMITS, rate_mbps
from .executor import resource_violations
from .graph_store import normalized, allocation_map


def legal_bandwidths(slice_type, cqi):
    if slice_type not in SLICE_LIMITS:
        raise ValueError("Unknown slice")
    if isinstance(cqi, bool) or not isinstance(cqi, (int, float)) or not math.isfinite(cqi) or int(cqi) != cqi or not 1 <= cqi <= 15:
        raise ValueError("CQI must be an integer in 1..15")
    limits = SLICE_LIMITS[slice_type]
    return [b for b in range(limits["bandwidth"][0], limits["bandwidth"][1] + 1)
            if limits["rate"][0] - 0.011 <= rate_mbps(b, cqi) <= limits["rate"][1] + 0.011]


def bandwidth_options(state, user, slice_type):
    """Exhaustive admission feasibility with all current users retained."""
    options = legal_bandwidths(slice_type, user["cqi"])
    limits = SLICE_LIMITS[slice_type]
    existing = state[limits["key"]]["users"]
    if any(str(u["user_id"]) == str(user["user_id"]) for v in SLICE_LIMITS.values() for u in state[v["key"]]["users"]):
        raise ValueError("Current user is already allocated; use plan_reallocation")
    minima, infeasible = [], []
    for u in existing:
        choices = legal_bandwidths(slice_type, u["cqi"])
        if choices:
            minima.append(min(choices))
        else:
            infeasible.append(str(u["user_id"]))
    remaining = limits["capacity"] - sum(minima)
    feasible = [] if infeasible else [b for b in options if b <= remaining]
    return {"slice_type": slice_type, "cqi": user["cqi"], "capacity_mhz": limits["capacity"],
            "available_before_adjustment_mhz": limits["capacity"] - state[limits["key"]]["resource_usage"],
            "minimum_existing_bandwidth_mhz": sum(minima) if not infeasible else None,
            "minimum_new_user_bandwidth_mhz": min(options) if options else None,
            "existing_users_without_feasible_bandwidth": infeasible,
            "legal_bandwidths_mhz": options, "feasible_after_adjustment_mhz": feasible,
            "admission_feasible": bool(feasible),
            "rates_mbps": {str(b): rate_mbps(b, user["cqi"]) for b in options},
            "scope": "Selected slice, retain all existing users, no slice changes or future data"}


def optimize_fixed_users(state, slice_types=None, pinned_bandwidth=None):
    """Return a new state; never mutate the passed simulator/graph state.

    pinned_bandwidth maps user ID to an explicit bandwidth constraint, used when
    an agent requests a particular new-user allocation. No pin means full pool
    optimization. Existing user latency is retained and independently audited.
    """
    selected = list(SLICE_LIMITS) if slice_types is None else list(slice_types)
    if not selected or not set(selected).issubset(SLICE_LIMITS):
        raise ValueError("Unknown/empty slice selection")
    pinned = pinned_bandwidth or {}
    candidate = deepcopy(state)
    reasons = []
    for name in selected:
        limits = SLICE_LIMITS[name]
        users = candidate[limits["key"]]["users"]
        choices = {}
        for u in users:
            uid = str(u["user_id"])
            allowed = legal_bandwidths(name, u["cqi"])
            if uid in pinned:
                b = pinned[uid]
                if isinstance(b, bool) or not isinstance(b, int):
                    raise ValueError("Bandwidth must be an integer number of MHz")
                allowed = [b] if b in allowed else []
            if not allowed:
                reasons.append(f"{name}:{uid}:no_legal_bandwidth")
            choices[uid] = allowed
        if any(not c for c in choices.values()):
            continue
        required = sum(min(c) for c in choices.values())
        if required > limits["capacity"]:
            reasons.append(f"{name}:minimum_required_{required}_exceeds_capacity_{limits['capacity']}")
            continue
        remaining = limits["capacity"] - required
        for u in users:
            u["bandwidth"] = min(choices[str(u["user_id"])])
        for u in sorted(users, key=lambda u: (-rate_mbps(1, u["cqi"]), str(u["user_id"]))):
            extra = min(remaining, max(choices[str(u["user_id"])]) - u["bandwidth"])
            u["bandwidth"] += extra
            remaining -= extra
            u["rate"] = rate_mbps(u["bandwidth"], u["cqi"])
    if reasons:
        return {"feasible": False, "violations": reasons}
    candidate = normalized(candidate)
    violations = resource_violations(candidate)
    return {"feasible": not violations, "violations": violations, "candidate_state": candidate}


def describe_changes(before, after):
    old, new = allocation_map(before), allocation_map(after)
    changed = []
    for uid, allocation in new.items():
        previous = old.get(uid)
        if previous != allocation:
            changed.append({"user_id": uid, "slice_type": allocation["slice_type"],
                            "old_bandwidth": previous["bandwidth"] if previous else None,
                            "new_bandwidth": allocation["bandwidth"], "new_rate": allocation["rate"]})
    return {"changes": changed,
            "total_rate_before_mbps": sum(rate_mbps(a["bandwidth"], a["cqi"]) for a in old.values()),
            "total_rate_after_mbps": sum(rate_mbps(a["bandwidth"], a["cqi"]) for a in new.values())}
