"""History-aware service-assurance previews for an unchanged admitted cohort.

Completed traffic intervals are the only history input. A candidate projects
the supplied CURRENT CQI over ``horizon_s``; this is a counterfactual allocation
score, never an observed traffic sample or a prediction of future movement.

The lexicographic objective is (effective users, sum log projected average
throughput, effective throughput, -absolute bandwidth change). Dividing each
user's average by its contract minimum only shifts the log utility by constants
(above the numerical floor): it does NOT introduce priority weights or directly
maximize Jain's index. Raw and demand-normalized Jain indices are both reported.
"""

from copy import deepcopy
import math

from .planner import _normalize, describe_changes, validate_allocations
from .service import (
    RATE_TOLERANCE_MBPS, SLICE_LIMITS, _jain, evaluate_service, rate_mbps,
)


SCORE_TOLERANCE = 1e-10
UTILITY_FLOOR = 1e-9
OBJECTIVE_PRIORITY = [
    "maximize_effective_users",
    "maximize_projected_log_average_throughput_utility",
    "maximize_effective_throughput_mbps",
    "minimize_total_bandwidth_change_mhz",
]
PROJECTION_NOTE = (
    "Completed historical intervals plus one counterfactual horizon at current CQI; "
    "projected traffic is not written as observed history. All fixed admitted users, "
    "including those below SLA, contribute delivered raw traffic to fairness. "
    "Log normalization by a fixed contract minimum is not a priority weight."
)


def _nonnegative(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and nonnegative")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and nonnegative") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _history(history, ids):
    """Normalize canonical volume/duration names and graph's explicit aliases."""
    if not isinstance(history, dict) or not isinstance(history.get("users"), dict):
        raise ValueError("History requires a users mapping of completed intervals")
    users = {str(uid): record for uid, record in history["users"].items()}
    if len(users) != len(history["users"]):
        raise ValueError("Duplicate normalized history user ID")
    common = (None if "duration_s" not in history else
              _nonnegative(history["duration_s"], "history.duration_s"))
    volumes = {}
    for uid in ids:
        if uid not in users or not isinstance(users[uid], dict):
            raise ValueError(f"{uid}:missing_completed_traffic_history")
        record = users[uid]
        if "traffic_mbit" not in record and "served_mbit" not in record:
            raise ValueError(f"{uid}:missing_traffic_mbit")
        volumes[uid] = _nonnegative(record.get("traffic_mbit", record.get("served_mbit")),
                                    f"{uid}.traffic_mbit")
        if "traffic_mbit" in record and "served_mbit" in record:
            alias = _nonnegative(record["served_mbit"], f"{uid}.served_mbit")
            if not math.isclose(volumes[uid], alias, rel_tol=1e-12, abs_tol=1e-9):
                raise ValueError(f"{uid}:conflicting_traffic_history_aliases")
        duration = record.get("observed_duration_s", record.get("duration_s", common))
        if duration is None:
            raise ValueError(f"{uid}:missing_observed_duration_s")
        duration = _nonnegative(duration, f"{uid}.observed_duration_s")
        if common is None:
            common = duration
        elif not math.isclose(common, duration, rel_tol=0, abs_tol=1e-9):
            raise ValueError("History duration must be identical for the fixed cohort")
        if duration == 0 and volumes[uid] != 0:
            raise ValueError(f"{uid}:positive_traffic_without_observed_duration")
    if common is None:
        raise ValueError("Cannot infer completed history duration")
    return common, volumes


def _fairness(volumes, duration, contracts):
    users = {}
    for uid, traffic in volumes.items():
        minimum = float(contracts[uid]["min_rate_mbps"])
        if not math.isfinite(minimum) or minimum <= 0:
            raise ValueError(f"{uid}:invalid_minimum_service_rate")
        mean = traffic / duration if duration > 0 else None
        users[uid] = {
            "user_id": uid, "slice_type": contracts[uid]["slice_type"],
            "traffic_mbit": traffic, "duration_s": duration,
            "mean_rate_mbps": mean,
            "normalized_mean_rate": mean / minimum if mean is not None else None,
        }

    def metrics(ids):
        return {
            "admitted_users": len(ids),
            "jain_rate_all_admitted": _jain([users[uid]["mean_rate_mbps"] for uid in ids])
            if duration > 0 and ids else None,
            "jain_demand_normalized_all_admitted":
            _jain([users[uid]["normalized_mean_rate"] for uid in ids])
            if duration > 0 and ids else None,
            "proportional_fair_log_utility":
            sum(math.log(max(users[uid]["normalized_mean_rate"], UTILITY_FLOOR)) for uid in ids)
            if duration > 0 and ids else None,
        }

    return {
        "duration_s": duration, "defined": duration > 0,
        **metrics(list(users)),
        "per_slice": {name: metrics([uid for uid in users if users[uid]["slice_type"] == name])
                      for name in SLICE_LIMITS},
        "users": users,
        "utility_floor": UTILITY_FLOOR,
        "normalization_note": "Fixed minimum-rate normalization is not weighted fairness or Jain maximization.",
    }


def summarize_history(history, contracts, allocations=None):
    """Actual completed-history fairness; t=0 indices/utility are undefined.

    Pass allocations to restrict history to the fixed admitted population when
    contracts or observation history also include unadmitted users.
    """
    agreements = {str(uid): value for uid, value in contracts.items()}
    ids = sorted(str(uid) for uid in (allocations if allocations is not None else history["users"]))
    if any(uid not in agreements for uid in ids):
        raise ValueError("Every historical user requires a service contract")
    duration, volumes = _history(history, ids)
    return _fairness(volumes, duration, agreements)


def compare_scores(left, right):
    """Lexicographic comparison, exact user count and tolerant floating terms.

    Accept either score dictionaries or four-element objective sequences.
    Return 1 for better, -1 for worse and 0 for numerically equal.
    """
    first = left["objective"] if isinstance(left, dict) else left
    second = right["objective"] if isinstance(right, dict) else right
    if len(first) != 4 or len(second) != 4:
        raise ValueError("Expected four lexicographic objective components")
    if not all(math.isfinite(float(value)) for value in [*first, *second]):
        raise ValueError("Objective components must be finite")
    if first[0] != second[0]:
        return 1 if first[0] > second[0] else -1
    for a, b in zip(first[1:], second[1:]):
        if abs(a - b) > SCORE_TOLERANCE:
            return 1 if a > b else -1
    return 0


def score_candidate(allocations, current_cqi, contracts, history, horizon_s=10.0,
                    reference_allocations=None):
    """Evaluate one candidate on identical causal history and current CQI."""
    candidate, cqis, agreements = _normalize(allocations, current_cqi, contracts)
    horizon = _nonnegative(horizon_s, "horizon_s")
    if horizon <= 0:
        raise ValueError("Projection horizon_s must be positive")
    duration, volumes = _history(history, sorted(candidate))
    reference = candidate if reference_allocations is None else {
        str(uid): value for uid, value in reference_allocations.items()}
    if set(reference) != set(candidate):
        raise ValueError("Candidate and reference must have the same admitted users")
    validated = validate_allocations(candidate, cqis, agreements, expected_allocations=reference)
    rows = [{"RX_ID": uid, "CQI": cqis[uid]} for uid in sorted(cqis)]
    evaluated = evaluate_service(rows, candidate, agreements)
    projected = _fairness({uid: volumes[uid] + evaluated["users"][uid]["rate_mbps"] * horizon
                          for uid in candidate}, duration + horizon, agreements)
    change = sum(abs(candidate[uid]["bandwidth"] - reference[uid]["bandwidth"]) for uid in candidate)
    metrics = evaluated["metrics"]
    return {
        "resource_valid": validated["resource_valid"], "violations": validated["violations"],
        "objective": [metrics["effective_users"], projected["proportional_fair_log_utility"],
                      metrics["effective_throughput_mbps"], -change],
        "metrics": metrics, "total_bandwidth_change_mhz": change,
        "historical_fairness": _fairness(volumes, duration, agreements),
        "projected_fairness": projected, "projected_per_user": projected["users"],
        "projection_horizon_s": horizon, "projection_note": PROJECTION_NOTE,
    }


def optimize_fair_service(allocations, current_cqi, contracts, history, horizon_s=10.0):
    """Exact per-slice integer DP, retaining every user and fixed slice.

    Hard bounds/pool budgets/rate ceilings always apply. Minimum-rate failures
    are observable and minimized first; a user's unrepairable channel does not
    block repairing other users. Only equal-count plans compete on historical
    log utility. States enumerate consumed MHz and retain the best additive
    score, with deterministic ties and a 1e-10 floating comparison tolerance.
    """
    original, cqis, agreements = _normalize(allocations, current_cqi, contracts)
    horizon = _nonnegative(horizon_s, "horizon_s")
    if horizon <= 0:
        raise ValueError("Projection horizon_s must be positive")
    duration, volumes = _history(history, sorted(original))
    candidate = deepcopy(original)
    reasons, irrecoverable, per_slice = [], [], {}
    for name, limits in SLICE_LIMITS.items():
        ids = sorted(uid for uid, allocation in original.items() if allocation["slice_type"] == name)
        options = {}
        for uid in ids:
            lower = max(limits["rate"][0], agreements[uid]["min_rate_mbps"])
            upper = min(limits["rate"][1], agreements[uid]["max_rate_mbps"])
            options[uid] = []
            for bandwidth in range(limits["bandwidth"][0], limits["bandwidth"][1] + 1):
                rate = rate_mbps(bandwidth, cqis[uid])
                if rate > upper + RATE_TOLERANCE_MBPS:
                    continue
                effective = rate >= lower - RATE_TOLERANCE_MBPS
                average = (volumes[uid] + rate * horizon) / (duration + horizon)
                utility = math.log(max(average / agreements[uid]["min_rate_mbps"], UTILITY_FLOOR))
                options[uid].append((bandwidth, int(effective), utility,
                                     rate if effective else 0.0,
                                     -abs(bandwidth - original[uid]["bandwidth"])))
            if not any(option[1] for option in options[uid]):
                irrecoverable.append(uid)
            if not options[uid]:
                reasons.append(f"{name}:{uid}:no_hard_valid_bandwidth")

        states = {0: ((0, 0.0, 0.0, 0.0), ())}
        for uid in ids:
            next_states = {}
            for used, (score, bands) in states.items():
                for bandwidth, *increment in options[uid]:
                    total = used + bandwidth
                    if total > limits["capacity"]:
                        continue
                    value = tuple(a + b for a, b in zip(score, increment))
                    old = next_states.get(total)
                    if old is None or compare_scores(value, old[0]) > 0:
                        next_states[total] = (value, bands + (bandwidth,))
            states = next_states
        best = None
        for used in sorted(states):
            entry = states[used]
            if best is None or compare_scores(entry[0], best[0]) > 0:
                best = entry
        if best is None:
            reasons.append(f"{name}:fixed_cohort_has_no_hard_valid_allocation")
            per_slice[name] = {"resource_valid": False, "user_ids": ids}
            continue
        objective, bands = best
        for uid, bandwidth in zip(ids, bands):
            candidate[uid]["bandwidth"] = bandwidth
        per_slice[name] = {
            "resource_valid": True, "user_ids": ids, "capacity_mhz": limits["capacity"],
            "used_mhz": sum(bands), "max_effective_users": objective[0],
            "proportional_fair_log_utility": objective[1],
            "effective_throughput_mbps": objective[2],
            "total_bandwidth_change_mhz": -objective[3],
        }
    if reasons:
        candidate = deepcopy(original)
    score = score_candidate(candidate, cqis, agreements, history, horizon,
                            reference_allocations=original)
    reasons.extend(reason for reason in score["violations"] if reason not in reasons)
    evaluated = evaluate_service([{"RX_ID": uid, "CQI": cqis[uid]} for uid in cqis],
                                 candidate, agreements)
    return {
        "allocations": candidate, "feasible": not reasons, "resource_valid": not reasons,
        "all_users_effective": not reasons and score["metrics"]["effective_users"] == len(original),
        "metrics": score["metrics"], "violations": reasons,
        "irrecoverable_user_ids": sorted(irrecoverable),
        "degraded_user_ids": sorted(uid for uid, user in evaluated["users"].items() if user["degraded"]),
        "objective": score["objective"], "score": score,
        "changes": describe_changes(original, candidate, cqis),
        "historical_fairness": score["historical_fairness"],
        "projected_fairness": score["projected_fairness"],
        "evidence": {
            "algorithm": "exact_per_slice_integer_dynamic_program",
            "priority": list(OBJECTIVE_PRIORITY), "score_tolerance": SCORE_TOLERANCE,
            "utility_floor": UTILITY_FLOOR, "per_slice": per_slice,
            "history_duration_s": duration, "projection_horizon_s": horizon,
            "projection_note": PROJECTION_NOTE, "latency_assessed": False,
            "irrecoverable_definition": "No individual hard-valid bandwidth meets the minimum; pool competition is separate.",
        },
    }
