"""Compact model observations; executable states and audit logs stay complete."""

from copy import deepcopy

from ..metrics import SLICE_LIMITS


def compact_schema(rows):
    return {row["source_type"]: [[r["relation"], r["target"]] for r in row["relations"]
                                 if r["relation"] is not None] for row in rows}


def compact_state(state):
    columns = ["user_id", "cqi", "bandwidth", "rate", "latency"]
    return {"allocation_columns": columns, "slices": {
        name: {"capacity_mhz": state[limits["key"]]["total_capacity"],
               "used_mhz": state[limits["key"]]["resource_usage"],
               "allocations": [[a[k] for k in columns] for a in state[limits["key"]]["users"]]}
        for name, limits in SLICE_LIMITS.items()}}


def model_view(name, result):
    value = deepcopy(result)
    if not isinstance(value, dict) or "error" in value:
        return value
    if name == "get_decision_context":
        value["user"] = {k: value["user"][k] for k in ("user_id", "request", "cqi", "location_json")}
        value["state"] = compact_state(value["state"])
    if name == "Network_topology_query":
        return compact_state(value)
    # Exact minima, feasible choices, violations, plan IDs, changes and rates of
    # the selected allocation remain visible. Per-option rate tables and fixed
    # descriptions are still available via full_evidence=true and the audit log.
    if name == "get_bandwidth_options":
        value.pop("rates_mbps", None)
        value.pop("scope", None)
    if name in ("plan_allocation", "plan_reallocation", "reject_request"):
        value.pop("objective", None)
        value.pop("resource_engine", None)
        if "feasibility_certificate" in value:
            value["feasibility_certificate"] = model_view("get_bandwidth_options", value["feasibility_certificate"])
    return value
