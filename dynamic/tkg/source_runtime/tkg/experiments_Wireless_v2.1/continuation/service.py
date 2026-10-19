"""Exact static-state import and unchanged physical service evaluation.

Request_Label is an immutable evaluation contract, never a way to replace an
inherited assigned slice. No admission or numerical optimization occurs at t=0.
"""

from copy import deepcopy
import math

# Pure, unchanged evaluator/model; original modules are never edited.
from dynamic_tkg.service import (
    SLICE_LIMITS, RATE_TOLERANCE_MBPS, RESOURCE_TOLERANCE,
    _cqi, _index_rows, _row_cqi, _jain, rate_mbps, legal_bandwidths,
    optimize_allocations, evaluate_service,
)

INITIALIZATION_MODES = ("static_terminal",)
INITIALIZATION_DESCRIPTIONS = {
    "static_terminal": "Exact static final state for all 30 observed users; original allocations, incorrect slices and rejected users retained. No t0 allocation optimization or model invocation."
}


def initialize_cohort(initial_rows, target_users=30, *, initialization_mode="static_terminal", static_terminal=None):
    if initialization_mode != "static_terminal" or static_terminal is None:
        raise ValueError("An explicit verified static terminal is required; numerical initialization is forbidden")
    if isinstance(target_users, bool) or target_users != 30:
        raise ValueError("Static continuation observes exactly 30 users per scene")
    indexed = _index_rows(initial_rows)
    source_rows = static_terminal["rows"]
    if set(indexed) != set(source_rows):
        raise ValueError("Initial observation population differs from the full static population")
    fields = {"X": "original_x", "Y": "original_y", "Z": "original_z",
              "CQI": "original_cqi", "SNR_dB": "original_snr_db",
              "RX_Power_dBm": "original_rx_power_dbm", "LOS": "original_los"}
    contracts, allocations, admission_log, legacy_users = {}, {}, [], {}
    original_allocations = {str(a["user_id"]): a for a in static_terminal["terminal_step"]["allocations"]}
    for uid, row in indexed.items():
        source = source_rows[uid]
        for current_key, original_key in fields.items():
            if float(row[current_key]) != float(source[original_key]):
                raise ValueError(f"{uid}: t0 {current_key} must exactly inherit static terminal telemetry")
        if row["User_Request"] != source["request"] or row["Request_Label"] != source["ground_truth"]:
            raise ValueError(f"{uid}: t0 request or true service contract differs from static source")
        name = source["ground_truth"]
        limits = SLICE_LIMITS[name]
        contracts[uid] = {"slice_type": name, "min_rate_mbps": limits["rate"][0],
                          "max_rate_mbps": limits["rate"][1],
                          "latency_min_ms": limits["latency"][0], "latency_max_ms": limits["latency"][1]}
        admitted = source["legacy_admitted"] == "1"
        if admitted:
            original = original_allocations[uid]
            bandwidth = float(original["bandwidth"])
            if not bandwidth.is_integer() or original["slice"] not in SLICE_LIMITS:
                raise ValueError(f"{uid}: unsupported original allocation")
            allocations[uid] = {"user_id": uid, "slice_type": original["slice"], "bandwidth": int(bandwidth)}
        legacy_users[uid] = {
            "source": "exact_static_terminal_import", "admitted": admitted,
            "assigned_slice": source["assigned_slice"] if admitted else None,
            "required_slice": name,
            "bandwidth_mhz": float(source["bandwidth_mhz"]),
            "source_rate_mbps": float(source["source_rate_mbps"]),
            "legacy_latency_ms": float(source["source_latency_ms"]) if admitted else None,
            "latency_is_dynamic_measurement": False,
            "inherited_wrong_slice": admitted and source["assigned_slice"] != name,
            "legacy_status": source["legacy_status"],
        }
        admission_log.append({"user_id": uid, "admitted": admitted,
                              "status": "inherited_admitted" if admitted else "inherited_rejected",
                              "violations": []})
    evaluated = evaluate_service(initial_rows, allocations, contracts)
    for uid, service in evaluated["users"].items():
        if not math.isclose(service["rate_mbps"], legacy_users[uid]["source_rate_mbps"], rel_tol=1e-12, abs_tol=1e-10):
            raise ValueError(f"{uid}: imported rate differs from original static terminal")
    for row in admission_log:
        row["violations"] = evaluated["users"][row["user_id"]]["violations"]
    evidence = deepcopy(static_terminal["provenance"])
    evidence.update(raw_static_state=deepcopy(static_terminal["raw_static_state"]),
                    observed_user_ids=list(indexed), admitted_user_ids=list(allocations),
                    unadmitted_user_ids=[uid for uid in indexed if uid not in allocations],
                    wrong_slice_user_ids=[uid for uid, v in legacy_users.items() if v["inherited_wrong_slice"]],
                    history_duration_s=0.0, observed_traffic_mbit=0.0,
                    agent_executed=False)
    return {"allocations": allocations, "contracts": contracts,
            "selected_user_ids": list(indexed), "admission_log": admission_log,
            "initialization_mode": initialization_mode, "initialization_evidence": evidence,
            "legacy_users": legacy_users}
