"""Recompute physical rates and audit upstream outputs without repairing them."""

import math
from copy import deepcopy

SLICE_LIMITS = {
    "eMBB": {"key": "embb_slice", "capacity": 90, "bandwidth": (6, 20), "rate": (100, 400), "latency": (10, 100)},
    "URLLC": {"key": "urllc_slice", "capacity": 30, "bandwidth": (1, 5), "rate": (1, 100), "latency": (1, 10)},
}


def rate_mbps(bandwidth: float, cqi: int) -> float:
    """Upstream engineering model, alpha=10; not a standardized PHY model."""
    return 10 * bandwidth * math.log10(1 + 10 ** (cqi / 10))


def empty_state() -> dict:
    return {
        item["key"]: {"users": [], "total_capacity": item["capacity"], "resource_usage": 0, "utilization_rate": "0%"}
        for item in SLICE_LIMITS.values()
    } | {"total_users": 0}


def audit_state(state: dict, ground_truth: dict, integer_bandwidth: bool) -> dict:
    """Labels are evaluator-only. Raw state is never modified by this function.

    Feasible throughput excludes invalid users and invalid resource pools.
    Raw recomputed throughput includes them, exposing upstream failure modes.
    """
    violations, seen, users_out = [], set(), []
    total_bw = raw_rate = reported_rate = feasible_rate = 0.0
    valid_count = correct_count = valid_correct_count = 0
    for slice_name, limits in SLICE_LIMITS.items():
        part = state[limits["key"]]
        pool = part["users"]
        bandwidth_sum = sum(float(u["bandwidth"]) for u in pool)
        pool_valid = bandwidth_sum <= limits["capacity"] + 1e-6 and bandwidth_sum >= -1e-6
        if not pool_valid:
            violations.append(f"{slice_name}:capacity")
        if abs(bandwidth_sum - float(part["resource_usage"])) > 1e-5:
            violations.append(f"{slice_name}:usage_bookkeeping")
        total_bw += bandwidth_sum
        for user in pool:
            uid = str(user["user_id"])
            bandwidth = float(user["bandwidth"])
            cqi = int(user["cqi"])
            reported = float(user.get("rate", 0))
            calculated = rate_mbps(bandwidth, cqi)
            failures = []
            if uid in seen:
                failures.append("duplicate_user")
            seen.add(uid)
            if uid not in ground_truth:
                failures.append("unknown_user")
            if not math.isfinite(bandwidth) or not limits["bandwidth"][0] - 1e-6 <= bandwidth <= limits["bandwidth"][1] + 1e-6:
                failures.append("bandwidth")
            if integer_bandwidth and abs(bandwidth - round(bandwidth)) > 1e-6:
                failures.append("noninteger_bandwidth")
            if not 1 <= cqi <= 15:
                failures.append("cqi")
            # Upstream rounds rates to two decimals. Tolerance covers that rounding.
            if not limits["rate"][0] - 0.011 <= calculated <= limits["rate"][1] + 0.011:
                failures.append("calculated_rate")
            if user.get("latency") is not None:
                latency = float(user["latency"])
                if not limits["latency"][0] <= latency <= limits["latency"][1]:
                    failures.append("assigned_latency")
            if not math.isfinite(reported) or abs(reported - calculated) > max(0.02, abs(calculated) * 0.001):
                violations.append(f"{uid}:reported_rate_mismatch")
            if not pool_valid:
                failures.append("pool_capacity")
            correct = ground_truth.get(uid) == slice_name
            valid = not failures
            correct_count += int(correct)
            valid_count += int(valid)
            valid_correct_count += int(valid and correct)
            if valid:
                feasible_rate += calculated
            raw_rate += calculated
            reported_rate += reported
            violations.extend(f"{uid}:{reason}" for reason in failures)
            users_out.append(deepcopy(user) | {
                "slice": slice_name, "calculated_rate_mbps": calculated,
                "correct_slice": correct, "resource_feasible": valid,
                "violations": failures,
            })
    if state.get("total_users", len(users_out)) != len(users_out):
        violations.append("total_users_bookkeeping")
    return {
        "admitted_users": len(users_out),
        "resource_feasible_users": valid_count,
        "correct_slice_users": correct_count,
        "valid_correct_users": valid_correct_count,
        "bandwidth_used_mhz": total_bw,
        "bandwidth_utilization_pct": total_bw / 120 * 100,
        "bandwidth_idle_pct": (120 - total_bw) / 120 * 100,
        "raw_calculated_throughput_mbps": raw_rate,
        "reported_throughput_mbps": reported_rate,
        "resource_feasible_throughput_mbps": feasible_rate,
        "violation_count": len(violations),
        "violations": violations, "allocations": users_out,
    }
