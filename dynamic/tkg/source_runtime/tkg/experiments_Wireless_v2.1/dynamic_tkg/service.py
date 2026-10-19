"""Pure resource/service calculations for the dynamic fixed-cohort experiment.

The model and limits come verbatim from v2 ``wireless_exp/metrics.py`` (vendored
as ``reference_metrics.py``). The integer optimizer follows v2
``wireless_exp/tkg/resources.py``: satisfy every retained user's minimum first,
then distribute MHz in descending marginal-rate order. It is throughput optimal
when retaining the complete cohort is feasible; it does not optimize admission
cardinality, evict users, migrate slices, or use future observations.

``Request_Label`` is read ONLY by ``initialize_cohort`` to bootstrap explicitly
known service contracts. This experiment is service assurance for existing
subscribers, not intent classification. Subsequent evaluation and planning use
these contracts and current CQI, never raw labels. Latency is not observed and
is never inferred from a slice name or an assigned placeholder value.
"""

from copy import deepcopy
import math

from .reference_metrics import SLICE_LIMITS, rate_mbps


RATE_TOLERANCE_MBPS = 0.011
RESOURCE_TOLERANCE = 1e-6
INITIALIZATION_MODES = ("feasible_admission", "fixed_population")
INITIALIZATION_DESCRIPTIONS = {
    "feasible_admission": "Known service contracts; source-order feasible admission using t0 CQI only, retaining only a fully effective selected cohort.",
    "fixed_population": "Known service contracts; retain the first requested users in source order, seed each at its hard bandwidth floor, then maximize effective count and proportional-fair log utility with zero completed history. Minimum-rate failures are recorded; hard bounds, rate ceilings and pool capacities are unchanged.",
}


def _cqi(value):
    """Accept CSV integer strings as well as integral numbers, without truncation."""
    if isinstance(value, bool):
        raise ValueError("CQI must be an integer in 1..15")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("CQI must be an integer in 1..15") from exc
    if not math.isfinite(number) or int(number) != number or not 1 <= number <= 15:
        raise ValueError("CQI must be an integer in 1..15")
    return int(number)


def _index_rows(rows):
    result = {}
    for row in rows:
        value = row.get("RX_ID", row.get("user_id"))
        if value is None or not str(value).strip():
            raise ValueError("Every observation must have an RX_ID/user_id")
        uid = str(value)
        if uid in result:
            raise ValueError(f"Duplicate user observation: {uid}")
        result[uid] = row
    return result


def _row_cqi(row):
    return _cqi(row.get("CQI", row.get("cqi")))


def legal_bandwidths(slice_type, cqi):
    """List exact legal integer MHz, using the original rate-rounding tolerance."""
    if slice_type not in SLICE_LIMITS:
        raise ValueError(f"Unknown slice: {slice_type}")
    cqi = _cqi(cqi)
    limits = SLICE_LIMITS[slice_type]
    return [
        bandwidth
        for bandwidth in range(limits["bandwidth"][0], limits["bandwidth"][1] + 1)
        if limits["rate"][0] - RATE_TOLERANCE_MBPS
        <= rate_mbps(bandwidth, cqi)
        <= limits["rate"][1] + RATE_TOLERANCE_MBPS
    ]


def optimize_allocations(allocations: dict, current_cqi: dict) -> dict:
    """Preview an exact fixed-cohort allocation, without modifying inputs.

    An infeasible result returns the original allocation unchanged, accompanied
    by reasons. The caller must not interpret that old allocation as feasible
    under the new CQI. Only the supplied current-frame CQI is used.
    """
    original = deepcopy(allocations)
    candidate = {}
    reasons = []
    cqis = {str(uid): value for uid, value in current_cqi.items()}
    choices = {}
    for key, allocation in allocations.items():
        uid = str(key)
        if uid in candidate:
            reasons.append(f"{uid}:duplicate_user")
            continue
        candidate[uid] = deepcopy(allocation)
        if str(allocation.get("user_id", uid)) != uid:
            reasons.append(f"{uid}:allocation_user_id_mismatch")
        candidate[uid]["user_id"] = uid
        name = allocation.get("slice_type")
        if name not in SLICE_LIMITS:
            reasons.append(f"{uid}:unknown_slice")
            continue
        try:
            cqis[uid] = _cqi(cqis.get(uid))
        except ValueError:
            reasons.append(f"{name}:{uid}:invalid_or_missing_cqi")
            continue
        choices[uid] = legal_bandwidths(name, cqis[uid])
        if not choices[uid]:
            reasons.append(f"{name}:{uid}:no_legal_bandwidth")
    if reasons:
        return {"feasible": False, "allocations": original, "violations": reasons}

    for name, limits in SLICE_LIMITS.items():
        users = [uid for uid, value in candidate.items() if value["slice_type"] == name]
        required = sum(min(choices[uid]) for uid in users)
        if required > limits["capacity"]:
            reasons.append(
                f"{name}:minimum_required_{required}_exceeds_capacity_{limits['capacity']}"
            )
            continue
        remaining = limits["capacity"] - required
        for uid in users:
            candidate[uid]["bandwidth"] = min(choices[uid])
        # rate_mbps is linear in bandwidth. Each user's legal choices form an
        # integer interval; taking the highest remaining marginal gain is exact.
        for uid in sorted(users, key=lambda item: (-rate_mbps(1, cqis[item]), item)):
            extra = min(remaining, max(choices[uid]) - candidate[uid]["bandwidth"])
            candidate[uid]["bandwidth"] += extra
            remaining -= extra
    return {
        "feasible": not reasons,
        "allocations": original if reasons else candidate,
        "violations": reasons,
    }


def initialize_cohort(initial_rows: list[dict], target_users: int = 10, *,
                      initialization_mode="feasible_admission") -> dict:
    """Initialize known contracts and the selected population at t=0.

    Each trial reallocates both pools while retaining every already selected
    user. Infeasible trials are skipped; no future CQI influences selection.
    Raise instead of quietly producing fewer users than the requested count.
    ``fixed_population`` instead retains the first requested subscribers even
    when their service minima cannot all be met. Its numerical initialization
    uses the existing count-first fair planner, with no observed traffic and
    no model invocation. Every hard resource constraint remains mandatory.
    """
    if isinstance(target_users, bool) or not isinstance(target_users, int) or target_users < 1:
        raise ValueError("target_users must be a positive integer")
    if initialization_mode not in INITIALIZATION_MODES:
        raise ValueError(f"Unknown initialization_mode: {initialization_mode}")
    indexed = _index_rows(initial_rows)
    contracts = {}
    cqis = {}
    for uid, row in indexed.items():
        name = row.get("Request_Label")
        if name not in SLICE_LIMITS:
            raise ValueError(f"Unknown/missing pre-established service label for user {uid}")
        # Labels terminate at this bootstrap boundary: no raw label is returned.
        limits = SLICE_LIMITS[name]
        contracts[uid] = {
            "slice_type": name,
            "min_rate_mbps": limits["rate"][0],
            "max_rate_mbps": limits["rate"][1],
            "latency_min_ms": limits["latency"][0],
            "latency_max_ms": limits["latency"][1],
        }
        cqis[uid] = _row_cqi(row)

    if initialization_mode == "fixed_population":
        if target_users > len(indexed):
            raise ValueError(f"Cannot initialize requested {target_users} users from only {len(indexed)} observations")
        selected = list(indexed)[:target_users]
        seed = {uid: {"user_id": uid, "slice_type": contracts[uid]["slice_type"],
                      "bandwidth": SLICE_LIMITS[contracts[uid]["slice_type"]]["bandwidth"][0]}
                for uid in selected}
        # Local import avoids the service -> fairness -> planner -> service cycle.
        from .fairness import optimize_fair_service
        history = {"duration_s": 0.0,
                   "users": {uid: {"traffic_mbit": 0.0} for uid in selected}}
        plan = optimize_fair_service(seed, cqis, contracts, history, horizon_s=10)
        if not plan["resource_valid"]:
            raise ValueError("Fixed population violates hard resource constraints: "
                             + "; ".join(plan["violations"]))
        allocations = plan["allocations"]
        services = evaluate_service(initial_rows, allocations, contracts)["users"]
        return {
            "allocations": allocations, "contracts": contracts,
            "selected_user_ids": selected,
            "admission_log": [
                {"user_id": uid, "admitted": uid in allocations,
                 "status": "admitted" if uid in allocations else "not_considered_after_target",
                 "violations": services[uid]["violations"]}
                for uid in indexed],
            "initialization_mode": initialization_mode,
            "initialization_evidence": {
                "algorithm": plan["evidence"]["algorithm"],
                "priority": plan["evidence"]["priority"],
                "objective": plan["objective"], "history_duration_s": 0.0,
                "observed_traffic_mbit": 0.0, "projection_horizon_s": 10,
                "resource_valid": True, "all_users_effective": plan["all_users_effective"],
                "degraded_user_ids": plan["degraded_user_ids"], "agent_executed": False,
            },
        }

    allocations, admission_log = {}, []
    for uid in indexed:
        name = contracts[uid]["slice_type"]
        if len(allocations) >= target_users:
            admission_log.append({"user_id": uid, "admitted": False, "status": "not_considered_after_target", "violations": []})
            continue
        tentative = deepcopy(allocations)
        tentative[uid] = {"user_id": uid, "slice_type": name, "bandwidth": 0}
        plan = optimize_allocations(tentative, cqis)
        admitted = plan["feasible"]
        admission_log.append({
            "user_id": uid, "admitted": admitted,
            "status": "admitted" if admitted else "skipped_infeasible_at_t0",
            "violations": plan["violations"],
        })
        if admitted:
            allocations = plan["allocations"]
    if len(allocations) != target_users:
        raise ValueError(
            f"Cannot initialize requested {target_users} users; source-order admission "
            f"selected only {len(allocations)} at t=0. Reduce target_users explicitly."
        )
    return {
        "allocations": allocations,
        "admission_log": admission_log,
        "contracts": contracts,
        "selected_user_ids": list(allocations),
    }


def _jain(values):
    """Jain index with the complete admitted population, including zero rates."""
    if not values:
        return 0.0
    squares = sum(value * value for value in values)
    return sum(values) ** 2 / (len(values) * squares) if squares else 0.0


def evaluate_service(rows: list[dict], allocations: dict, contracts: dict) -> dict:
    """Recompute all observed users using current CQI and an unchanged allocation.

    ``effective`` covers slice, bandwidth, rate and pool constraints only.
    Latency is explicitly unassessed. Unadmitted observed users are neither
    effective nor degraded. Fairness uses all admitted raw nonnegative rates;
    demand normalization is r/r_min, *unclipped*, with a fixed denominator.
    """
    indexed = _index_rows(rows)
    normalized = {str(uid): value for uid, value in allocations.items()}
    if len(normalized) != len(allocations):
        raise ValueError("Duplicate normalized allocation user ID")
    missing = set(normalized) - set(indexed)
    if missing:
        raise ValueError(f"Missing observations for admitted users: {sorted(missing)}")
    contracts = {str(uid): value for uid, value in contracts.items()}
    if set(indexed) - set(contracts):
        raise ValueError("Every observed user must have a pre-established service contract")

    slice_usage = {name: 0.0 for name in SLICE_LIMITS}
    for uid, allocation in normalized.items():
        try:
            bandwidth = float(allocation["bandwidth"])
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"Invalid bandwidth for user {uid}") from exc
        if not math.isfinite(bandwidth):
            raise ValueError(f"Nonfinite bandwidth for user {uid}")
        name = allocation.get("slice_type")
        if name in slice_usage:
            slice_usage[name] += bandwidth

    users = {}
    admitted_rates, normalized_rates = [], []
    for uid, row in indexed.items():
        cqi = _row_cqi(row)
        contract = contracts[uid]
        minimum, maximum = float(contract["min_rate_mbps"]), float(contract["max_rate_mbps"])
        if not math.isfinite(minimum) or not math.isfinite(maximum) or not 0 < minimum <= maximum:
            raise ValueError(f"Invalid service rate contract for user {uid}")
        admitted = uid in normalized
        allocation = normalized.get(uid, {})
        name = allocation.get("slice_type", contract["slice_type"])
        bandwidth = float(allocation["bandwidth"]) if admitted else 0.0
        rate = rate_mbps(bandwidth, cqi) if admitted else 0.0
        violations = []
        if admitted:
            if str(allocation.get("user_id", uid)) != uid:
                violations.append("allocation_user_id_mismatch")
            if name != contract["slice_type"]:
                violations.append("wrong_slice")
            if name not in SLICE_LIMITS:
                violations.append("unknown_slice")
            else:
                limits = SLICE_LIMITS[name]
                if not limits["bandwidth"][0] - RESOURCE_TOLERANCE <= bandwidth <= limits["bandwidth"][1] + RESOURCE_TOLERANCE:
                    violations.append("bandwidth_bounds")
                if not -RESOURCE_TOLERANCE <= slice_usage[name] <= limits["capacity"] + RESOURCE_TOLERANCE:
                    violations.append("pool_capacity")
                if not limits["rate"][0] - RATE_TOLERANCE_MBPS <= rate <= limits["rate"][1] + RATE_TOLERANCE_MBPS:
                    violations.append("slice_rate_bounds")
            if isinstance(allocation.get("bandwidth"), bool) or abs(bandwidth - round(bandwidth)) > RESOURCE_TOLERANCE:
                violations.append("noninteger_bandwidth")
            if rate < minimum - RATE_TOLERANCE_MBPS:
                violations.append("rate_below_minimum")
            if rate > maximum + RATE_TOLERANCE_MBPS:
                violations.append("rate_above_maximum")
            admitted_rates.append(max(0.0, rate))
            normalized_rates.append(max(0.0, rate) / minimum)
        effective = admitted and not violations
        users[uid] = {
            "user_id": uid, "admitted": admitted, "slice_type": name,
            "bandwidth_mhz": bandwidth, "rate_mbps": rate,
            "rate_min_mbps": minimum, "rate_max_mbps": maximum,
            "rate_margin_mbps": rate - minimum,
            "effective": effective, "degraded": admitted and not effective,
            "violations": violations, "latency_assessed": False,
            "service_status": "effective" if effective else ("degraded" if admitted else "unadmitted"),
        }
    effective_users = sum(int(user["effective"]) for user in users.values())
    return {
        "users": users,
        "metrics": {
            "observed_users": len(users), "admitted_users": len(normalized),
            "effective_users": effective_users,
            "degraded_users": len(normalized) - effective_users,
            "unadmitted_users": len(users) - len(normalized),
            "raw_throughput_mbps": sum(user["rate_mbps"] for user in users.values() if user["admitted"]),
            "effective_throughput_mbps": sum(user["rate_mbps"] for user in users.values() if user["effective"]),
            "jain_rate_all_admitted": _jain(admitted_rates),
            "jain_demand_normalized_all_admitted": _jain(normalized_rates),
            "mean_rate_satisfaction_clipped_all_admitted": (
                sum(min(1.0, value) for value in normalized_rates) / len(normalized_rates)
                if normalized_rates else 0.0
            ),
            "slice_usage_mhz": slice_usage, "latency_assessed": False,
        },
    }
