"""WirelessAgent adapter for the unchanged TKG-DTN integer resource solver.

Planning is read-only. Both slices are optimized for the retained users plus the
new user, without a pinned model bandwidth, eviction, or access to intent labels.
"""

from copy import deepcopy

from .metrics import SLICE_LIMITS, rate_mbps
from .tkg.executor import resource_violations
from .tkg.graph_store import allocation_map, normalized
from .tkg.resources import bandwidth_options, optimize_fixed_users


def plan_admission(state, user_id, cqi, slice_type, latency):
    uid = str(user_id)
    certificate = bandwidth_options(state, {"user_id": uid, "cqi": cqi}, slice_type)
    plan = {"feasible": False, "user_id": uid, "slice_type": slice_type,
            "base_state": normalized(state), "certificate": certificate,
            "adjustments": [], "resource_engine": "exact_resources_v2"}
    if not certificate["admission_feasible"]:
        return plan
    candidate = deepcopy(state)
    bandwidth = certificate["minimum_new_user_bandwidth_mhz"]
    candidate[SLICE_LIMITS[slice_type]["key"]]["users"].append({
        "user_id": uid, "cqi": cqi, "bandwidth": bandwidth,
        "rate": rate_mbps(bandwidth, cqi), "latency": latency,
    })
    optimized = optimize_fixed_users(candidate)
    if not optimized["feasible"]:
        raise ValueError("Shared resource optimizer rejected the candidate: " + str(optimized["violations"]))
    candidate = optimized["candidate_state"]
    old, new = allocation_map(state), allocation_map(candidate)
    adjustments = [
        {"user_id": key, "slice_type": a["slice_type"],
         "old_bandwidth": a["bandwidth"], "new_bandwidth": new[key]["bandwidth"],
         "old_rate": a["rate"], "new_rate": new[key]["rate"],
         "bandwidth_reduction": a["bandwidth"] - new[key]["bandwidth"],
         "bandwidth_change": new[key]["bandwidth"] - a["bandwidth"]}
        for key, a in old.items() if a["bandwidth"] != new[key]["bandwidth"]
    ]
    allocation = next(u for u in candidate[SLICE_LIMITS[slice_type]["key"]]["users"] if str(u["user_id"]) == uid)
    return plan | {"feasible": True, "candidate_state": candidate,
                   "allocation": deepcopy(allocation), "adjustments": adjustments}


def commit_admission(state, plan):
    """Validate the preview against the current state before atomic publication."""
    if not plan["feasible"]:
        raise ValueError("Cannot commit an infeasible admission")
    if normalized(state) != plan["base_state"]:
        raise ValueError("Stale resource plan; calculate again from the current state")
    violations = resource_violations(plan["candidate_state"])
    if violations:
        raise ValueError("Invalid resource plan: " + str(violations))
    return deepcopy(plan["candidate_state"])
