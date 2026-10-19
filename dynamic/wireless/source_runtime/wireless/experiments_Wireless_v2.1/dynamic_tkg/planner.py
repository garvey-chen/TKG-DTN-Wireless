"""Exact current-frame service assurance for a fixed admitted population.

This is a tool, not an autonomous policy: previewing never commits a decision.
The lexicographic objective is effective users, effective throughput, raw
throughput, then the smallest total absolute bandwidth change. All users remain
admitted on their original slice, even when a service minimum is unattainable.
Unlike the original all-or-nothing optimizer, a service breach does not prevent
repairing other users. Physical bounds, pool budgets and rate ceilings remain
hard constraints; service minima and the evaluator are unchanged.
"""

from copy import deepcopy
import math

from .service import (
    RATE_TOLERANCE_MBPS, RESOURCE_TOLERANCE, SLICE_LIMITS, _cqi,
    evaluate_service, rate_mbps,
)


SCORE_TOLERANCE = 1e-8
OBJECTIVE_PRIORITY = [
    "maximize_effective_users", "maximize_effective_throughput_mbps",
    "maximize_raw_throughput_mbps", "minimize_total_bandwidth_change_mhz",
]


def _normalize(allocations, current_cqi, contracts):
    for name, mapping in (("allocations", allocations), ("current_cqi", current_cqi), ("contracts", contracts)):
        if len({str(uid) for uid in mapping}) != len(mapping):
            raise ValueError(f"Duplicate normalized user ID in {name}")
    allocations = {str(uid): deepcopy(value) for uid, value in allocations.items()}
    if not allocations:
        raise ValueError("Require a nonempty fixed admitted cohort")
    cqis = {str(uid): _cqi(value) for uid, value in current_cqi.items()}
    agreements = {str(uid): deepcopy(value) for uid, value in contracts.items()}
    for uid, allocation in allocations.items():
        if uid not in cqis or uid not in agreements:
            raise ValueError(f"{uid}:missing_current_cqi_or_contract")
        if str(allocation.get("user_id", uid)) != uid:
            raise ValueError(f"{uid}:allocation_user_id_mismatch")
        name = allocation.get("slice_type")
        if name not in SLICE_LIMITS or agreements[uid].get("slice_type") != name:
            raise ValueError(f"{uid}:unknown_or_changed_contract_slice")
        minimum = float(agreements[uid]["min_rate_mbps"])
        maximum = float(agreements[uid]["max_rate_mbps"])
        if not math.isfinite(minimum) or not math.isfinite(maximum) or not 0 < minimum <= maximum:
            raise ValueError(f"{uid}:invalid_rate_contract")
        agreements[uid]["min_rate_mbps"] = minimum
        agreements[uid]["max_rate_mbps"] = maximum
        bandwidth = allocation.get("bandwidth")
        if isinstance(bandwidth, bool) or not isinstance(bandwidth, (int, float)):
            raise ValueError(f"{uid}:invalid_bandwidth")
        if not math.isfinite(bandwidth) or abs(bandwidth - round(bandwidth)) > RESOURCE_TOLERANCE:
            raise ValueError(f"{uid}:noninteger_bandwidth")
    return allocations, cqis, agreements


def validate_allocations(allocations, current_cqi, contracts, expected_allocations=None):
    """Validate hard constraints only; minimum-rate breaches remain observable.

    Supply ``expected_allocations`` at commit to prohibit admission, removal and
    migration. The caller owns the authoritative immutable service contracts.
    """
    reasons = []
    try:
        if len({str(uid) for uid in allocations}) != len(allocations):
            raise ValueError("Duplicate normalized allocation user ID")
        normalized, cqis, agreements = _normalize(allocations, current_cqi, contracts)
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        return {"resource_valid": False, "violations": [str(exc)], "slice_usage_mhz": {}}
    if expected_allocations is not None:
        expected = {str(uid): allocation for uid, allocation in expected_allocations.items()}
        if set(normalized) != set(expected):
            reasons.append("fixed_cohort_changed")
        for uid in set(normalized) & set(expected):
            if normalized[uid]["slice_type"] != expected[uid]["slice_type"]:
                reasons.append(f"{uid}:fixed_slice_changed")
    usage = {name: 0 for name in SLICE_LIMITS}
    for uid, allocation in normalized.items():
        name, bandwidth = allocation["slice_type"], allocation["bandwidth"]
        limits = SLICE_LIMITS[name]
        usage[name] += bandwidth
        if not limits["bandwidth"][0] <= bandwidth <= limits["bandwidth"][1]:
            reasons.append(f"{name}:{uid}:bandwidth_bounds")
        upper = min(limits["rate"][1], agreements[uid]["max_rate_mbps"])
        if rate_mbps(bandwidth, cqis[uid]) > upper + RATE_TOLERANCE_MBPS:
            reasons.append(f"{name}:{uid}:rate_above_maximum")
    for name, limits in SLICE_LIMITS.items():
        if not -RESOURCE_TOLERANCE <= usage[name] <= limits["capacity"] + RESOURCE_TOLERANCE:
            reasons.append(f"{name}:pool_capacity")
    return {"resource_valid": not reasons, "violations": reasons, "slice_usage_mhz": usage}


def describe_changes(before, after, current_cqi):
    """Return bandwidth deltas; all rates use the same current-frame channel."""
    return [{
        "user_id": uid, "slice_type": after[uid]["slice_type"],
        "old_bandwidth_mhz": before[uid]["bandwidth"],
        "new_bandwidth_mhz": after[uid]["bandwidth"],
        "old_rate_mbps": rate_mbps(before[uid]["bandwidth"], _cqi(current_cqi[uid])),
        "new_rate_mbps": rate_mbps(after[uid]["bandwidth"], _cqi(current_cqi[uid])),
    } for uid in sorted(after) if before[uid]["bandwidth"] != after[uid]["bandwidth"]]


def _better(first, second):
    """Compare count exactly; suppress roundoff-only rate and change ties."""
    if second is None:
        return True
    if first[0] != second[0]:
        return first[0] > second[0]
    for a, b in zip(first[1:], second[1:]):
        if abs(a - b) > SCORE_TOLERANCE:
            return a > b
    return False


def optimize_service(allocations, current_cqi, contracts):
    """Preview an exact integer plan using only the supplied current snapshot.

    Every state is a pool's consumed integer MHz. Keeping its best additive
    score gives an exhaustive dynamic program, without exponential cohort
    subset search. A score's effective-rate term excludes below-target users,
    exactly as ``evaluate_service`` does. Floating point ties use 1e-8 Mbps.
    """
    if len({str(uid) for uid in allocations}) != len(allocations):
        raise ValueError("Duplicate normalized allocation user ID")
    original, cqis, agreements = _normalize(allocations, current_cqi, contracts)
    candidate = deepcopy(original)
    per_slice, irrecoverable, reasons = {}, [], []
    for name, limits in SLICE_LIMITS.items():
        ids = sorted(uid for uid, value in original.items() if value["slice_type"] == name)
        options = {}
        for uid in ids:
            floor = max(limits["rate"][0], agreements[uid]["min_rate_mbps"])
            ceiling = min(limits["rate"][1], agreements[uid]["max_rate_mbps"])
            options[uid] = []
            for bandwidth in range(limits["bandwidth"][0], limits["bandwidth"][1] + 1):
                rate = rate_mbps(bandwidth, cqis[uid])
                if rate <= ceiling + RATE_TOLERANCE_MBPS:
                    valid = rate >= floor - RATE_TOLERANCE_MBPS
                    options[uid].append((bandwidth, valid, rate))
            if not any(valid for _, valid, _ in options[uid]):
                irrecoverable.append(uid)
            if not options[uid]:
                reasons.append(f"{name}:{uid}:no_hard_valid_bandwidth")
        # value = ((effective count, effective rate, raw rate, -change), bands)
        states = {0: ((0, 0.0, 0.0, 0.0), ())}
        for uid in ids:
            next_states = {}
            for used, (score, bands) in states.items():
                for bandwidth, valid, rate in options[uid]:
                    total = used + bandwidth
                    if total > limits["capacity"]:
                        continue
                    value = (score[0] + int(valid), score[1] + (rate if valid else 0.0),
                             score[2] + rate, score[3] - abs(bandwidth - original[uid]["bandwidth"]))
                    old = next_states.get(total)
                    if _better(value, old[0] if old else None):
                        next_states[total] = (value, bands + (bandwidth,))
            states = next_states
        best = None
        for used in sorted(states):
            value = states[used]
            if _better(value[0], best[0] if best else None):
                best = value
        if best is None:
            reasons.append(f"{name}:fixed_cohort_has_no_hard_valid_allocation")
            per_slice[name] = {"resource_valid": False, "user_ids": ids}
            continue
        score, bands = best
        for uid, bandwidth in zip(ids, bands):
            candidate[uid]["bandwidth"] = bandwidth
        per_slice[name] = {
            "resource_valid": True, "user_ids": ids, "capacity_mhz": limits["capacity"],
            "used_mhz": sum(bands), "max_effective_users": score[0],
            "effective_throughput_mbps": score[1], "raw_throughput_mbps": score[2],
            "total_bandwidth_change_mhz": -score[3],
        }
    if reasons:
        candidate = deepcopy(original)
    checked = validate_allocations(candidate, cqis, agreements, expected_allocations=original)
    reasons.extend(reason for reason in checked["violations"] if reason not in reasons)
    rows = [{"RX_ID": uid, "CQI": cqi} for uid, cqi in cqis.items()]
    evaluated = evaluate_service(rows, candidate, agreements)
    metrics = evaluated["metrics"]
    return {
        "allocations": candidate, "resource_valid": not reasons,
        "all_users_effective": not reasons and metrics["effective_users"] == len(original),
        "metrics": metrics, "violations": reasons,
        "irrecoverable_user_ids": sorted(irrecoverable),
        "degraded_user_ids": sorted(uid for uid, user in evaluated["users"].items() if user["degraded"]),
        "objective": {key: metrics[key] for key in
                      ("effective_users", "effective_throughput_mbps", "raw_throughput_mbps")} | {
            "total_bandwidth_change_mhz": sum(abs(candidate[uid]["bandwidth"] - original[uid]["bandwidth"])
                                              for uid in original)},
        "changes": describe_changes(original, candidate, cqis),
        "evidence": {
            "algorithm": "exact_per_slice_integer_dynamic_program", "priority": list(OBJECTIVE_PRIORITY),
            "score_tolerance_mbps": SCORE_TOLERANCE, "per_slice": per_slice,
            "scope": "Current CQI only; fixed admitted cohort and slices; no admission, removal, migration or contract changes.",
            "irrecoverable_definition": "No individual hard-valid bandwidth meets the unchanged minimum; pool competition is separate.",
            "latency_assessed": False,
        },
    }
