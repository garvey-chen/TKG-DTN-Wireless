#!/usr/bin/env python3
"""Independently audit completion-driven runs without importing controller code.

Recompute the declared engineering rate model, service validity, event-split
traffic and fixed-admitted-cohort Jain fairness from recorded physical states.
No model or network call is made. Incomplete runs are never certified complete.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
METHODS = ("tkg_dtn", "wireless_agent", "frozen")
SCENES = ("center", "north", "south")
EXPECTED_ADMITTED = {"center": 27, "north": 30, "south": 29}
LIMITS = {"eMBB": {"capacity": 90, "bw": (6, 20), "rate": (100, 400)},
          "URLLC": {"capacity": 30, "bw": (1, 5), "rate": (1, 100)}}
EPS = 1e-7


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()] if path.exists() else []


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b, message="Numeric mismatch", tolerance=EPS):
    assert math.isfinite(float(a)) and math.isfinite(float(b)), message + ": nonfinite"
    assert math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=tolerance), f"{message}: {a} != {b}"


def truth(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() in ("true", "1"):
        return True
    if str(value).lower() in ("false", "0"):
        return False
    raise ValueError(f"Invalid boolean: {value!r}")


def rate(bandwidth, cqi):
    return 10 * float(bandwidth) * math.log10(1 + 10 ** (float(cqi) / 10))


def jain(values):
    values = list(values)
    denominator = len(values) * sum(value * value for value in values)
    return sum(values) ** 2 / denominator if denominator else 0.0


def allocation_signature(allocations):
    return {str(uid): (value["slice_type"], float(value["bandwidth"])) for uid, value in allocations.items()}


def physical_services(state):
    """Independent duplicate of the documented metric, never planner output."""
    allocations, contracts, users = state["allocations"], state["contracts"], state["users"]
    usage = {name: sum(float(a["bandwidth"]) for a in allocations.values() if a["slice_type"] == name)
             for name in LIMITS}
    result = {}
    for uid, user in users.items():
        cqi = float(user["cqi"])
        assert cqi.is_integer() and 1 <= cqi <= 15, f"Invalid CQI {uid}"
        a, contract = allocations.get(uid), contracts[uid]
        value = rate(a["bandwidth"], cqi) if a else 0.0
        effective = False
        if a:
            limits = LIMITS[a["slice_type"]]
            bw = float(a["bandwidth"])
            effective = (a["slice_type"] == contract["slice_type"]
                and limits["bw"][0] - 1e-6 <= bw <= limits["bw"][1] + 1e-6
                and abs(bw - round(bw)) <= 1e-6
                and -1e-6 <= usage[a["slice_type"]] <= limits["capacity"] + 1e-6
                and limits["rate"][0] - 0.011 <= value <= limits["rate"][1] + 0.011
                and float(contract["min_rate_mbps"]) - 0.011 <= value <= float(contract["max_rate_mbps"]) + 0.011)
        result[uid] = {"rate_mbps": value, "effective": effective, "admitted": a is not None}
    return result


def hard_valid(allocations, state, inherited):
    if set(allocations) != set(inherited):
        return False
    usage = defaultdict(float)
    for uid, a in allocations.items():
        if a["slice_type"] != inherited[uid]["slice_type"]:
            return False
        name, bw = a["slice_type"], float(a["bandwidth"])
        limits = LIMITS[name]
        if not math.isfinite(bw) or abs(bw - round(bw)) > 1e-6 or not limits["bw"][0] <= bw <= limits["bw"][1]:
            return False
        usage[name] += bw
        ceiling = min(limits["rate"][1], float(state["contracts"][uid]["max_rate_mbps"]))
        if rate(bw, state["users"][uid]["cqi"]) > ceiling + 0.011:
            return False
    return all(usage[name] <= limits["capacity"] + 1e-6 for name, limits in LIMITS.items())


def model_usage(directory):
    logs = read_jsonl(directory / "llm_calls.jsonl")
    successful = [row for row in logs if "usage" in row]
    calls = {row["call"] for row in logs}
    valid_calls = {row["call"] for row in logs if row.get("response") and row.get("finish_reason") != "length" and "error" not in row}
    return {"llm_calls": len({row["call"] for row in logs}),
            "api_attempts": len({(row["call"], row["attempt"]) for row in logs}),
            "llm_failures": len(calls - valid_calls),
            "prompt_tokens": sum(int(row["usage"].get("prompt_tokens", 0)) for row in successful),
            "completion_tokens": sum(int(row["usage"].get("completion_tokens", 0)) for row in successful)}


def audit_run(directory):
    directory = Path(directory)
    run = read_json(directory / "run.json")
    assert run["status"] == "complete", "Run is incomplete"
    # Publication subset hashes are verified by reproduce.py against RELEASE_MANIFEST.json.
    # Large graph databases and redundant policy traces are intentionally excluded.
    checked_hashes = 0
    root = HERE.parent
    run["input_dir"] = str(root / "data/mobility")
    run.pop("input_csv", None)
    handoff = dict(run["static_terminal_provenance"])
    handoff["directory"] = str(root / "dynamic/data" / (
        "wireless_static_terminal_handoff_20261015" if run["method"] == "wireless_agent"
        else "static_terminal_handoff_20261014"))
    method, scene = run["method"], run["scenario"]
    assert method in METHODS and scene in SCENES
    close(run["duration_s"], 300, "Duration")
    assert int(run["observed_users"]) == 30
    states = read_jsonl(directory / "states.jsonl")
    assert states, "Missing physical states"
    state_times = [float(row["time_s"]) for row in states]
    assert state_times == sorted(state_times), "Physical states are not chronological"
    close(state_times[0], 0, "Initial time")
    close(state_times[-1], 300, "Terminal time")
    initial = states[0]
    ids = set(map(str, range(1, 31)))
    admitted = set(initial["allocations"])
    assert len(admitted) == EXPECTED_ADMITTED[scene]
    assert set(map(str, run["initial_admitted_user_ids"])) == admitted
    original_step = read_json(Path(handoff["directory"]) / scene / "terminal_step.json")
    original_allocations = {str(row["user_id"]): (row["slice"], float(row["bandwidth"])) for row in original_step["allocations"]}
    assert allocation_signature(initial["allocations"]) == original_allocations, "Initial allocation differs from original static terminal"
    original_users = read_csv(Path(handoff["directory"]) / scene / "static_terminal_users.csv")
    for row in original_users:
        assert initial["contracts"][str(row["user_id"])]["slice_type"] == row["ground_truth"], "Original service contract changed"
    immutable_slices = {uid: a["slice_type"] for uid, a in initial["allocations"].items()}
    telemetry_path = Path(run.get("input_csv", Path(run["input_dir"]) / f"{scene}.csv"))
    telemetry = {(int(float(row["Time_s"])), str(row["RX_ID"])): row for row in read_csv(telemetry_path)}
    assert len(telemetry) == 9030, "Require 30 users x 301 telemetry samples"
    grouped = defaultdict(list)
    service_checks = 0
    for state in states:
        t, sample = float(state["time_s"]), float(state["telemetry_time_s"])
        assert 0 <= t <= 300
        close(sample, math.floor(t + 1e-10), "Latest causal telemetry second")
        assert set(state["users"]) == ids and set(state["contracts"]) == ids
        assert state["contracts"] == initial["contracts"], "Service contracts mutated"
        assert set(state["allocations"]) == admitted, "Fixed admitted cohort changed"
        assert {uid: a["slice_type"] for uid, a in state["allocations"].items()} == immutable_slices, "Assigned slices changed"
        usage = Counter()
        for uid, allocation in state["allocations"].items():
            bw = float(allocation["bandwidth"])
            name = allocation["slice_type"]
            assert bw.is_integer() and LIMITS[name]["bw"][0] <= bw <= LIMITS[name]["bw"][1]
            usage[name] += bw
        assert all(usage[name] <= limit["capacity"] + 1e-6 for name, limit in LIMITS.items())
        recomputed = physical_services(state)
        for uid, user in state["users"].items():
            row = telemetry[(int(sample), uid)]
            close(user["cqi"], row["CQI"], "CQI differs from causal input")
            for coordinate, key in zip(user["location"], ("X", "Y", "Z")):
                close(coordinate, row[key], "Position differs from causal input", 1.1e-6)
            assert user["request"] == row["User_Request"]
            close(user["service"]["rate_mbps"], recomputed[uid]["rate_mbps"], "Recorded modeled rate")
            assert bool(user["service"]["effective"]) == recomputed[uid]["effective"], "Recorded effectiveness"
            service_checks += 1
        grouped[t].append(state)
    times = sorted(grouped)
    assert set(map(float, range(301))).issubset(grouped), "Missing one-second observations"
    final_state_at = {t: rows[-1] for t, rows in grouped.items()}
    decisions = read_jsonl(directory / "decisions.jsonl")
    if method == "frozen":
        assert not decisions
        assert all(allocation_signature(state["allocations"]) == allocation_signature(initial["allocations"]) for state in states)
    else:
        assert decisions, "No real decisions"
        close(decisions[0]["source_time_s"], 1, "First decision must preserve t0 handoff")
    completions = {}
    for index, decision in enumerate(decisions):
        s, sample, c = (float(decision[key]) for key in ("source_time_s", "source_sample_time_s", "completed_time_s"))
        duration = float(decision["decision_wall_seconds"])
        assert 1 <= s < 300 and duration > 0 and int(decision["llm_calls"]) >= 0
        if decision["policy_status"] != "failed":
            assert int(decision["llm_calls"]) > 0, "Successful decisions require real model calls"
        close(sample, math.floor(s + 1e-10), "Decision's latest input second")
        close(c, s + duration, "Measured completion time")
        close(decision["scheduled_apply_time_s"], c, "No grid rounding of actuation")
        assert s in final_state_at, "Missing physical source state"
        assert allocation_signature(decision["allocation_before"]) == allocation_signature(final_state_at[s]["allocations"]), "Policy source allocation differs from actual state"
        assert int(decision["source_revision"]) == int(final_state_at[s]["revision"]), "Policy source revision mismatch"
        if index:
            close(s, decisions[index - 1]["completed_time_s"], "Next decision must immediately follow previous completion")
        if c >= 300:
            assert index == len(decisions) - 1 and decision["status"] == "terminal_drop"
            assert decision.get("applied_time_s") is None
            continue
        close(decision["checked_time_s"], c, "Candidate checked immediately on completion")
        assert c in grouped, "Missing physical completion event"
        assert len(grouped[c]) >= 2, "Completion requires both pre- and post-execution physical states"
        completions[c] = decision
        before, after = grouped[c][0], grouped[c][-1]
        assert allocation_signature(decision["allocation_after"]) == allocation_signature(after["allocations"])
        if run.get("execution_policy") == "hard_constraints_only_original_candidate_v1":
            assert decision.get("execution_policy") == run["execution_policy"], "Decision execution policy mismatch"
            if decision["policy_status"] in ("reallocated", "approved", "committed"):
                changed = allocation_signature(decision["allocations"]) != allocation_signature(before["allocations"])
                if changed:
                    valid = hard_valid(decision["allocations"], before, initial["allocations"])
                    expected_status = "reallocated" if valid else "rejected_stale_or_invalid"
                    assert decision["status"] == expected_status, "Hard-only executor rejected a valid candidate or applied an invalid one"
                    if valid:
                        close(decision["applied_time_s"], c, "Hard-valid changed candidate must execute")
        if decision.get("applied_time_s") is not None:
            close(decision["applied_time_s"], c, "Actual application must equal completion")
            assert hard_valid(decision["allocations"], before, initial["allocations"]), "Applied candidate invalid on latest CQI"
            assert allocation_signature(after["allocations"]) == allocation_signature(decision["allocations"])
        else:
            assert allocation_signature(before["allocations"]) == allocation_signature(after["allocations"]), "Unapplied candidate changed allocation"
        if method == "wireless_agent" and decision["policy_status"] != "failed":
            assert int(decision["llm_calls"]) >= 4
    if decisions:
        assert float(decisions[-1]["completed_time_s"]) >= 300, "Controller stopped before horizon"
    previous_allocations = allocation_signature(initial["allocations"])
    for state in states[1:]:
        current = allocation_signature(state["allocations"])
        if current != previous_allocations:
            t = float(state["time_s"])
            assert t in completions and completions[t].get("applied_time_s") is not None, "Allocation changed before a completed, applied decision"
            assert current == allocation_signature(completions[t]["allocations"])
        previous_allocations = current
    intervals = read_csv(directory / "traffic_intervals.csv")
    interval_index = {(float(row["start_time_s"]), float(row["end_time_s"]), str(row["user_id"])): row for row in intervals}
    assert len(interval_index) == len(intervals), "Duplicate traffic intervals"
    assert len(intervals) == 30 * (len(times) - 1), "Every physical interval must account for all 30 users"
    totals = {uid: {"traffic_mbit": 0.0, "effective_served_mbit": 0.0, "effective_time_s": 0.0, "observed_duration_s": 0.0} for uid in sorted(ids, key=int)}
    accumulated_at = {0.0: {uid: 0.0 for uid in ids}}
    for start, end in zip(times, times[1:]):
        assert 0 <= start < end <= 300
        services = physical_services(final_state_at[start])
        for uid in ids:
            row = interval_index[(start, end, uid)]
            dt, service = end - start, services[uid]
            traffic = dt * service["rate_mbps"]
            effective_traffic, effective_time = (traffic, dt) if service["effective"] else (0.0, 0.0)
            close(row["rate_mbps"], service["rate_mbps"], "Interval left-end physical rate")
            close(row["traffic_mbit"], traffic, "Traffic integration")
            close(row["effective_served_mbit"], effective_traffic, "Effective traffic integration")
            close(row["effective_time_s"], effective_time, "Effective-user time integration")
            assert truth(row["service_effective"]) == service["effective"]
            for key, value in (("traffic_mbit", traffic), ("effective_served_mbit", effective_traffic),
                               ("effective_time_s", effective_time), ("observed_duration_s", dt)):
                totals[uid][key] += value
        accumulated_at[end] = {uid: values["traffic_mbit"] for uid, values in totals.items()}
    for uid in ids:
        close(totals[uid]["observed_duration_s"], 300, "Full user observation coverage")
    frames = read_csv(directory / "frame_metrics.csv")
    assert [float(row["time_s"]) for row in frames] == list(map(float, range(301))), "Frame metrics must be exactly 0..300"
    for row in frames:
        t = float(row["time_s"])
        service = physical_services(final_state_at[t])
        close(row["effective_users"], sum(v["effective"] for v in service.values()), "Frame effective users")
        close(row["raw_throughput_mbps"], sum(v["rate_mbps"] for v in service.values()), "Frame raw throughput")
        close(row["effective_throughput_mbps"], sum(v["rate_mbps"] for v in service.values() if v["effective"]), "Frame effective throughput")
        if t > 0:
            fairness = jain(accumulated_at[t][uid] / float(initial["contracts"][uid]["min_rate_mbps"]) for uid in sorted(admitted, key=int))
            close(row["cumulative_jain_demand_normalized_admitted"], fairness, "Cumulative fixed-admitted-cohort Jain")
    usage = model_usage(directory)
    api_log = read_jsonl(directory / "llm_calls.jsonl")
    returned_models = sorted({str(row["model_returned"]) for row in api_log if row.get("model_returned")})
    requested_models = sorted({str(row["model_requested"]) for row in api_log if row.get("model_requested")})
    assert sum(int(row["llm_calls"]) for row in decisions) == usage["llm_calls"], "All LLM calls, including terminal drops, must be counted"
    for key, value in usage.items():
        assert sum(int(row[key]) for row in decisions) == value, f"Per-decision model accounting mismatch: {key}"
        if key in run.get("model_usage", {}):
            assert int(run["model_usage"][key]) == value, f"Model usage mismatch: {key}"
    if method == "frozen":
        assert usage["llm_calls"] == 0
    else:
        assert usage["llm_calls"] <= int(run["model_settings"]["max_calls_per_run"])
        for call in api_log:
            if "model_requested" in call:
                assert call["model_requested"] == run["model_settings"]["model"], "Undeclared model change"
    fixed_fairness = jain(totals[uid]["traffic_mbit"] / float(initial["contracts"][uid]["min_rate_mbps"]) for uid in sorted(admitted, key=int))
    durations = [float(row["decision_wall_seconds"]) for row in decisions]
    guard_durations = [float(row["executor_guard_wall_seconds"]) for row in decisions if "executor_guard_wall_seconds" in row]
    assert all(math.isfinite(value) and value >= 0 for value in guard_durations), "Invalid external guard duration"
    summary = {"method": method, "scenario": scene, "duration_s": 300, "observed_users": 30,
        "execution_policy": run.get("execution_policy", "archived_method_specific"),
        "admitted_users": len(admitted), "mean_effective_users": sum(v["effective_time_s"] for v in totals.values()) / 300,
        "effective_user_time_pct_all30": sum(v["effective_time_s"] for v in totals.values()) / 9000 * 100,
        "mean_effective_throughput_mbps": sum(v["effective_served_mbit"] for v in totals.values()) / 300,
        "mean_raw_throughput_mbps": sum(v["traffic_mbit"] for v in totals.values()) / 300,
        "final_cumulative_jain_demand_normalized_admitted": fixed_fairness,
        "final_cumulative_jain_rate_admitted": jain(totals[uid]["traffic_mbit"] for uid in sorted(admitted, key=int)),
        "decision_count": len(decisions), "nonempty_applied_time_events": sum(row.get("applied_time_s") is not None for row in decisions),
        "completed_before_horizon_decisions": sum(float(row["completed_time_s"]) < 300 for row in decisions),
        "reallocated_decisions": sum(row["status"] == "reallocated" for row in decisions),
        "held_decisions": sum(row["status"] == "held" for row in decisions),
        "rejected_decisions": sum("rejected" in row["status"] for row in decisions),
        "policy_failed_decisions": sum(row["status"] == "policy_failed" for row in decisions),
        "failed_workflows_including_terminal_drop": sum(row["policy_status"] == "failed" for row in decisions),
        "terminal_drop_decisions": sum(row["status"] == "terminal_drop" for row in decisions),
        "mean_decision_seconds": statistics.mean(durations) if durations else 0.0,
        "max_decision_seconds": max(durations, default=0.0),
        "guarded_decisions": len(guard_durations),
        "executor_guard_total_seconds": sum(guard_durations),
        "mean_executor_guard_seconds_per_guarded_decision": statistics.mean(guard_durations) if guard_durations else 0.0,
        "max_executor_guard_seconds": max(guard_durations, default=0.0),
        "mean_input_age_seconds": statistics.mean(float(row["source_time_s"]) - float(row["source_sample_time_s"]) for row in decisions) if decisions else 0.0,
        "total_tokens": usage["prompt_tokens"] + usage["completion_tokens"],
        "mean_tokens_per_decision": (usage["prompt_tokens"] + usage["completion_tokens"]) / len(decisions) if decisions else 0.0,
        "models_returned": ";".join(returned_models),
        "models_requested": ";".join(requested_models),
        **usage}
    verification = {"status": "passed", "method": method, "scenario": scene,
        "execution_policy": run.get("execution_policy", "archived_method_specific"),
        "physical_state_count": len(states), "physical_user_service_checks": service_checks,
        "event_intervals": len(times) - 1, "user_interval_rows": len(intervals),
        "decision_count": len(decisions), "total_user_seconds": 9000,
        "verified_artifact_source_and_handoff_hashes": checked_hashes,
        "continuous_timing_no_future_input_no_early_apply": True,
        "fixed_cohort_and_slice_assignments": True, "independent_traffic_and_fairness_recomputed": True,
        "input_csv_sha256": digest(telemetry_path),
        "artifact_sha256": {name: digest(directory / name) for name in
            ("run.json", "states.jsonl", "decisions.jsonl", "traffic_intervals.csv", "frame_metrics.csv", "llm_calls.jsonl") if (directory / name).exists()}}
    curve_metrics = []
    for t in times:
        state = final_state_at[t]
        service = physical_services(state)
        curve_metrics.append({"time_s": t, "telemetry_time_s": state["telemetry_time_s"],
            "event_kind": state["event_kind"],
            "effective_users": sum(row["effective"] for row in service.values()),
            "effective_throughput_mbps": sum(row["rate_mbps"] for row in service.values() if row["effective"]),
            "raw_throughput_mbps": sum(row["rate_mbps"] for row in service.values()),
            "cumulative_jain_demand_normalized_admitted": jain(
                accumulated_at[t][uid] / float(initial["contracts"][uid]["min_rate_mbps"])
                for uid in sorted(admitted, key=int)) if t else None})
    return {"run": run, "summary": summary, "audit": verification, "user_totals": totals,
            "initial_state": initial, "frames": frames, "decisions": decisions, "curve_metrics": curve_metrics}


def audit_package(directory, allow_partial=False):
    directory = Path(directory)
    results, failures, pending = [], [], []
    for method in METHODS:
        for scene in SCENES:
            path = directory / "runs" / f"{method}_{scene}"
            if not (path / "run.json").exists() or read_json(path / "run.json").get("status") != "complete":
                pending.append(str(path.relative_to(directory)))
                continue
            try:
                results.append(audit_run(path))
            except Exception as exc:
                failures.append({"run": str(path.relative_to(directory)), "type": type(exc).__name__, "message": str(exc)})
    for scene in SCENES:
        paired = [result for result in results if result["run"]["scenario"] == scene]
        if len(paired) > 1:
            first = paired[0]["initial_state"]
            for other in paired[1:]:
                second = other["initial_state"]
                same_users = all(all(first["users"][uid][field] == second["users"][uid][field]
                                    for field in ("cqi", "location", "request", "service")) for uid in first["users"])
                if first["allocations"] != second["allocations"] or first["contracts"] != second["contracts"] or not same_users:
                    failures.append({"run": scene, "type": "InitialPairingMismatch", "message": "Method initial physical states differ"})
    status = "failed" if failures or (pending and not allow_partial) else "partial" if pending else "passed"
    report = {"status": status, "completed_runs": len(results), "expected_runs": 9,
              "pending_runs": pending, "failures": failures, "runs": [result["audit"] for result in results],
              "summaries": [result["summary"] for result in results],
              "auditor_sha256": digest(Path(__file__)),
              "metric_semantics": "Independent saturated modeled-rate integration over [0,300); fixed admitted-cohort demand-normalized cumulative Jain; all 30 users in effective-user-time denominator; not packet measurements."}
    return report, results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=HERE)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    report, _ = audit_package(args.directory, args.allow_partial)
    (args.directory / "audit_event_runs.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "completed_runs", "expected_runs", "pending_runs", "failures")}, ensure_ascii=False, indent=2))
    return int(report["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
