#!/usr/bin/env python3
"""Read-only audit of all 60 completed trials, with repeat-level statistics.

Run: python3 audit_five.py. Outputs stay beside this script. No experiment,
model/API call, graph connection, or write to source experiments is performed.
The experimental unit for dispersion is one full repeated run, never an arrival.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.dont_write_bytecode = True
DEST = Path(__file__).resolve().parent
WORKSPACE = next(p for p in DEST.parents if (p / "main.tex").is_file())
EXP = WORKSPACE / "DTN-KG-transfer-package-20260909/DTN-KG-new-transfer-20260909/DTN-KG-new/experiments_Wireless_v2"
BATCH = EXP / "results/four_methods_5repeats_20261012_212809"
sys.path.insert(0, str(EXP))
from wireless_exp.metrics import audit_state
from wireless_exp.tkg.graph_store import normalized, packed, replay_events

SCENES = ("center", "north", "south")
METHODS = ("tkg_dtn", "wireless_kb", "rule_upstream", "prompt")
REPEATS = range(1, 6)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return str(path.relative_to(WORKSPACE))


def write_csv(name, rows):
    with (DEST / name).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def qualified_throughput(step):
    return sum(u["calculated_rate_mbps"] for u in step["allocations"]
               if u["correct_slice"] and u["resource_feasible"])


def summarize(rows, fields):
    result = {"n_repeats": len(rows)}
    for field in fields:
        values = [row[field] for row in rows]
        if all(v is None for v in values):
            result[field + "_mean"] = None
            result[field + "_sd"] = None
            continue
        result[field + "_mean"] = statistics.mean(values)
        result[field + "_sd"] = statistics.stdev(values)
    return result


def main():
    execution = read(BATCH / "execution_record.json")
    expected_code = execution["source_sha256"]
    audit = {
        "status": "passed",
        "audited_at_Asia_Shanghai": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "batch": relative(BATCH), "repeats_used_for_all_manuscript_metrics": list(REPEATS),
        "planned_repeats": 5, "trials_used": 60, "users_per_trial": 30,
        "audited_steps_all_methods": 1800, "paired_agent_arrival_positions": 450,
        "sources_sha256": {}, "trials": {}, "paired_checks": {}, "graph_checks": {},
        "original_r001_copy_provenance": {}, "execution_record_snapshot": execution,
        "recorded_code_sha256": expected_code,
        "definitions": {
            "valid_correct_users": "Users in the correct service slice AND satisfying per-user resource constraints and slice-pool capacity.",
            "resource_feasible_throughput_mbps": "Sum of recalculated rates of resource-feasible users; includes users assigned to the wrong service slice.",
            "correct_feasible_throughput_mbps": "Sum of recalculated rates of users BOTH in the correct service slice AND resource feasible.",
            "mean_decision_seconds": "Wall-clock time per incoming request including policy/model/tools, state snapshot and metric audit; excludes per-trial setup. This is not radio transmission latency.",
            "total_tokens": "API-reported input plus output tokens, including cached input tokens; not a monetary cost estimate.",
            "service_statistics": "Per scenario/method: arithmetic mean and SAMPLE standard deviation (ddof=1) of the five trial final values.",
            "cost_statistics": "First pool all 90 requests from three scenarios within each repeat and method; derive per-request means; then arithmetic mean and SAMPLE standard deviation across the five repeat means. The 450 requests are not treated as 450 independent experimental replicates.",
            "normalized_state": "Canonical resource state from the existing graph_store.normalized: ordered slice allocations and capacities with framework-specific metadata omitted. Equality refers to resource state, not raw log dictionary identity.",
            "tkg_model_tool_calls": "Records with origin=model in tool_calls.jsonl. Program-prefetched context reads are counted separately.",
            "wirelessagent_tool_calls": "Unavailable under an equivalent logged definition: no tool_calls.jsonl. Blank is unknown, not zero. Do not compare tool counts across frameworks.",
            "rule_upstream": "Original proportional-fair allocation and fallback, oracle service labels, continuous bandwidth. Integer-bandwidth audit disabled for this baseline as in the runner.",
        },
        "limitations": [
            "Five repeats reuse each scene's fixed 30-user trace and source arrival order; they measure execution repeatability, not generalization across five independently drawn networks.",
            "Both agents use exact_resources_v2; shared allocation tools do not force identical intent classification or tool trajectories.",
            "The locally adapted implementations differ in model context, staged reasoning, revision caching and HTTP handling; overhead differences do not isolate a causal effect of graph representation alone.",
            "Requested model alias is deepseek-chat; returned model in these logs is deepseek-flash.",
            "WirelessAgent has no equivalent tool-call instrumentation; cross-framework tool-call reductions cannot be calculated.",
            "CQI, location, request content and service composition vary jointly across scenes; differences are not a controlled CQI-only comparison.",
            "CQI is static per user. Mobility, dynamic fading, handover, interference and wireless hardware are not validated.",
            "Rate follows the inherited engineering CQI-to-rate approximation; the latency attribute is assigned, not measured.",
            "Wall-clock API timing depends on this execution environment and remote service conditions.",
        ],
    }
    summaries = [read(p) for p in BATCH.glob("*__*__r*/summary.json")]
    audit["complete_trials_at_audit"] = sum(s["status"] == "complete" and s["users_processed"] == 30 for s in summaries)
    assert len(summaries) == audit["complete_trials_at_audit"] == 60
    assert execution["status"] == "runs_complete" and not execution["failed_trials"]
    assert execution["total_trials_complete"] == 60
    audit["completion_snapshot"] = [
        {key: s[key] for key in ("scenario", "method", "repeat", "status", "users_processed")}
        for s in sorted(summaries, key=lambda s: (s["repeat"], s["scenario"], s["method"]))
    ]
    for filename, digest in expected_code.items():
        assert sha(EXP / filename) == digest, (filename, "source differs from recorded code")
    audit["current_source_matches_recorded_code"] = True
    original_batch = EXP / execution["first_round_source"]
    original_sources = {(s["scenario"], s["method"]): original_batch / s["directory"]
                        for s in read(original_batch / "trial_sources.json")}
    trial_rows, scene_rows, paired_rows, data = [], [], [], {}
    for repeat in REPEATS:
        for scene in SCENES:
            for method in METHODS:
                name = f"{scene}__{method}__r{repeat:03d}"
                path = BATCH / name
                summary, manifest = read(path / "summary.json"), read(path / "manifest.json")
                steps, selected = lines(path / "steps.jsonl"), read(path / "selected_users.json")
                assert summary["status"] == manifest["status"] == "complete"
                assert summary["repeat"] == manifest["repeat"] == repeat
                assert manifest["code_sha256"] == expected_code
                assert sha(EXP / f"data/scenarios/{scene}.csv") == manifest["scenario_sha256"]
                assert len(steps) == len(selected) == summary["users_processed"] == 30
                assert [s["user_id"] for s in steps] == manifest["user_ids"]
                assert [s["arrivals"] for s in steps] == list(range(1, 31))
                labels = {u["user_id"]: u["ground_truth"] for u in selected}
                for step in steps:
                    calc = audit_state(step["state"], labels, integer_bandwidth=method != "rule_upstream")
                    for metric in ("admitted_users", "valid_correct_users", "resource_feasible_users",
                                   "bandwidth_used_mhz", "raw_calculated_throughput_mbps",
                                   "resource_feasible_throughput_mbps", "violation_count"):
                        assert math.isclose(calc[metric], step[metric], abs_tol=1e-7), (name, step["arrivals"], metric)
                    assert calc["allocations"] == step["allocations"]
                    if method in ("tkg_dtn", "wireless_kb"):
                        assert not calc["violations"]
                last = steps[-1]
                for field, value in summary["final"].items():
                    assert value == last[field], (name, "summary final", field)
                assert summary["steps_with_audit_issues"] == sum(bool(s["violations"]) for s in steps)
                stats = summary.get("llm_statistics", {})
                calls = lines(path / "llm_calls.jsonl") if (path / "llm_calls.jsonl").exists() else []
                if calls:
                    assert len(calls) == stats["llm_calls"] == stats["api_attempts"]
                    assert sum(c["usage"]["prompt_tokens"] for c in calls) == stats["prompt_tokens"]
                    assert sum(c["usage"]["completion_tokens"] for c in calls) == stats["completion_tokens"]
                    assert all(c.get("response") is not None for c in calls)
                tools = lines(path / "tool_calls.jsonl") if (path / "tool_calls.jsonl").exists() else []
                row = {
                    "scenario": scene, "method": method, "repeat": repeat, "users_processed": 30,
                    "admitted_users": last["admitted_users"], "correct_feasible_users": last["valid_correct_users"],
                    "correct_feasible_pct": last["valid_correct_users"] / 30 * 100,
                    "resource_feasible_users": last["resource_feasible_users"],
                    "resource_feasible_throughput_mbps": last["resource_feasible_throughput_mbps"],
                    "correct_feasible_throughput_mbps": qualified_throughput(last),
                    "bandwidth_used_mhz": last["bandwidth_used_mhz"],
                    "final_bandwidth_utilization_pct": last["bandwidth_utilization_pct"],
                    "mean_bandwidth_utilization_pct": summary["mean_utilization_over_all_processed_arrivals_pct"],
                    "intent_accuracy_pct": summary["intent_accuracy_on_classified_arrivals_pct"],
                    "final_violation_count": last["violation_count"],
                    "steps_with_audit_issues": summary["steps_with_audit_issues"],
                    "llm_calls": stats.get("llm_calls", 0), "api_attempts": stats.get("api_attempts", 0),
                    "llm_failures": stats.get("llm_failures", 0),
                    "input_tokens": stats.get("prompt_tokens", 0), "output_tokens": stats.get("completion_tokens", 0),
                    "total_tokens": stats.get("prompt_tokens", 0) + stats.get("completion_tokens", 0),
                    "mean_decision_seconds": summary["mean_decision_seconds"],
                    "total_decision_seconds": sum(s["step_seconds"] for s in steps),
                    "trial_elapsed_seconds": manifest["elapsed_seconds"],
                    "logged_tool_calls": len(tools) if tools else "",
                    "model_origin_tool_calls": sum(t.get("origin") == "model" for t in tools) if tools else "",
                    "program_prefetch_tool_calls": sum(t.get("origin") == "program_prefetch" for t in tools) if tools else "",
                    "api_model_requested": ";".join(sorted({c["model_requested"] for c in calls})),
                    "api_model_returned": ";".join(sorted({c["model_returned"] for c in calls})),
                    "source_directory": relative(path),
                }
                assert math.isclose(row["mean_decision_seconds"], statistics.mean(s["step_seconds"] for s in steps), abs_tol=1e-10)
                assert math.isclose(row["mean_bandwidth_utilization_pct"], statistics.mean(s["bandwidth_utilization_pct"] for s in steps), abs_tol=1e-10)
                trial_rows.append(row)
                data[scene, method, repeat] = (manifest, steps, row, selected)
                audit["trials"][name] = {
                    **row, "tool_counts_by_name": dict(Counter(t["tool"] for t in tools)),
                    "tool_error_records": [
                        {"user_id": t["user_id"], "tool": t["tool"], "error": t.get("error") or t.get("result", {}).get("error")}
                        for t in tools if t.get("error") or (isinstance(t.get("result"), dict) and t["result"].get("error"))
                    ],
                    "expected_infeasible_admission_plans": sum(
                        t["tool"] == "plan_allocation" and isinstance(t.get("result"), dict) and t["result"].get("feasible") is False
                        for t in tools
                    ),
                    "rejected_user_ids": [s["user_id"] for s in steps if s["result"].get("status") == "rejected"],
                    "incorrectly_classified_user_ids": [s["user_id"] for s in steps if s["intent_correct_this_arrival"] is False],
                    "final_wrong_slice_users": [a["user_id"] for a in last["allocations"] if not a["correct_slice"]],
                }
                relevant_files = ["manifest.json", "summary.json", "steps.jsonl", "selected_users.json"]
                if calls:
                    relevant_files.append("llm_calls.jsonl")
                if tools:
                    relevant_files += ["tool_calls.jsonl", "graph/events.jsonl", "graph/mapping_stats.json"]
                for filename in relevant_files:
                    file = path / filename
                    audit["sources_sha256"][relative(file)] = sha(file)
                    if repeat == 1:
                        assert sha(file) == sha(original_sources[scene, method] / filename), (name, filename, "copy provenance")
                if repeat == 1:
                    audit["original_r001_copy_provenance"][name] = {
                        "original_directory": relative(original_sources[scene, method]),
                        "copied_directory": relative(path), "identical_file_contents": relevant_files,
                    }
                if method == "tkg_dtn":
                    events = lines(path / "graph/events.jsonl")
                    decision_matches, event_hash_matches = 0, 0
                    for index, event in enumerate(events):
                        reconstructed = replay_events(events[:index + 1])["state"]
                        digest = hashlib.sha256(packed(reconstructed).encode("utf-8")).hexdigest()
                        assert digest == event["state_sha256"], (name, index, "event hash")
                        assert event["projection_verified"] is True
                        event_hash_matches += 1
                        if event["event_type"] == "decision":
                            step = next(s for s in steps if s["user_id"] == event["payload"]["user_id"])
                            assert reconstructed == normalized(step["state"]), (name, index, "step state")
                            assert event["revision"] == step["result"]["mapped_revision"]
                            decision_matches += 1
                    assert decision_matches == 30
                    graph_stats = read(path / "graph/mapping_stats.json")
                    assert graph_stats["replay_verified"]
                    audit["graph_checks"][name] = {
                        "events": len(events), "event_hashes_reproduced": event_hash_matches,
                        "decision_states_reproduced": decision_matches, "projection_verified_logged_for_every_event": True,
                        "mapping_stats": graph_stats,
                        "verification_scope": "Replayed archived graph deltas and checked every archived state hash and post-decision state; no live graph queried.",
                    }
            tm, ts, tr, selected = data[scene, "tkg_dtn", repeat]
            wm, ws, wr, _ = data[scene, "wireless_kb", repeat]
            for field in ("scenario_sha256", "knowledge_sha256", "user_ids", "seed", "arrival_order", "llm", "code_sha256", "resource_tools_version"):
                assert tm[field] == wm[field], (scene, repeat, field)
            local_pairs = []
            for t, w in zip(ts, ws):
                paired = {"scenario": scene, "repeat": repeat, "arrival": t["arrivals"], "user_id": t["user_id"]}
                assert t["user_id"] == w["user_id"]
                fields = ("admitted_users", "valid_correct_users", "resource_feasible_throughput_mbps", "bandwidth_used_mhz")
                for field in fields:
                    paired["tkg_" + field] = t[field]
                    paired["wirelessagent_" + field] = w[field]
                    paired["difference_" + field] = t[field] - w[field]
                paired["tkg_correct_feasible_throughput_mbps"] = qualified_throughput(t)
                paired["wirelessagent_correct_feasible_throughput_mbps"] = qualified_throughput(w)
                paired["same_selected_slice"] = t["result"].get("slice_type") == w["result"].get("slice_type")
                paired["same_full_state"] = normalized(t["state"]) == normalized(w["state"])
                paired["same_service_metrics"] = all(abs(t[f] - w[f]) < 1e-7 for f in fields)
                paired_rows.append(paired)
                local_pairs.append(paired)
            audit["paired_checks"][f"{scene}__r{repeat:03d}"] = {
                "paired_inputs_config_code_and_resource_version_equal": True,
                "same_slice_decisions": sum(p["same_selected_slice"] for p in local_pairs),
                "same_full_postdecision_states": sum(p["same_full_state"] for p in local_pairs),
                "same_service_metric_steps": sum(p["same_service_metrics"] for p in local_pairs),
                "first_request_with_full_total_bandwidth": next((s["arrivals"] for s in ts if s["bandwidth_used_mhz"] >= 120 - 1e-9), None),
            }
            if repeat == 1:
                cqi = [u["cqi"] for u in selected]
                scene_rows.append({
                    "scenario": scene, "users": len(selected), "mean_cqi": statistics.mean(cqi),
                    "min_cqi": min(cqi), "max_cqi": max(cqi), "cqi_15_users": cqi.count(15),
                    "ground_truth_embb_users": sum(u["ground_truth"] == "eMBB" for u in selected),
                    "ground_truth_urllc_users": sum(u["ground_truth"] == "URLLC" for u in selected),
                    "cqi_histogram": json.dumps(dict(sorted(Counter(cqi).items()))),
                })
            for method in METHODS:
                assert data[scene, method, repeat][3] == data[scene, "tkg_dtn", 1][3], (scene, method, repeat, "trace changed")
    audit["paired_totals"] = {
        "paired_steps": len(paired_rows),
        "same_slice_decisions": sum(p["same_selected_slice"] for p in paired_rows),
        "same_normalized_resource_states": sum(p["same_full_state"] for p in paired_rows),
        "same_service_metric_steps": sum(p["same_service_metrics"] for p in paired_rows),
        "first_difference": next((p for p in paired_rows if not all(p[f] for f in ("same_selected_slice", "same_full_state", "same_service_metrics"))), None),
    }
    service_fields = ("admitted_users", "correct_feasible_users", "correct_feasible_pct", "resource_feasible_users",
                      "resource_feasible_throughput_mbps", "correct_feasible_throughput_mbps", "bandwidth_used_mhz",
                      "final_bandwidth_utilization_pct", "mean_bandwidth_utilization_pct", "intent_accuracy_pct",
                      "final_violation_count", "steps_with_audit_issues")
    summary_rows = []
    for scene in SCENES:
        for method in METHODS:
            rows = [r for r in trial_rows if r["scenario"] == scene and r["method"] == method]
            summary_rows.append({"scenario": scene, "method": method, **summarize(rows, service_fields)})
    repeat_costs = []
    for method in METHODS:
        for repeat in REPEATS:
            rows = [r for r in trial_rows if r["method"] == method and r["repeat"] == repeat]
            totals = {field: sum(r[field] for r in rows) for field in
                      ("users_processed", "llm_calls", "input_tokens", "output_tokens", "total_tokens", "total_decision_seconds")}
            row = {"method": method, "repeat": repeat, **totals,
                   "mean_decision_seconds": totals["total_decision_seconds"] / totals["users_processed"]}
            for field in ("llm_calls", "input_tokens", "output_tokens", "total_tokens"):
                row[field + "_per_request"] = totals[field] / totals["users_processed"]
            for field in ("logged_tool_calls", "model_origin_tool_calls", "program_prefetch_tool_calls"):
                row[field + "_per_request"] = sum(r[field] for r in rows) / 90 if method == "tkg_dtn" else ""
            repeat_costs.append(row)
    cost_fields = ("llm_calls_per_request", "input_tokens_per_request", "output_tokens_per_request", "total_tokens_per_request", "mean_decision_seconds")
    cost_rows = []
    for method in METHODS:
        rows = [r for r in repeat_costs if r["method"] == method]
        summary = {"method": method, "requests_per_repeat": 90, **summarize(rows, cost_fields)}
        for field in ("logged_tool_calls_per_request", "model_origin_tool_calls_per_request", "program_prefetch_tool_calls_per_request"):
            vals = [r[field] for r in rows]
            summary[field + "_mean"] = statistics.mean(vals) if method == "tkg_dtn" else ""
            summary[field + "_sd"] = statistics.stdev(vals) if method == "tkg_dtn" else ""
        cost_rows.append(summary)
    tkg = next(r for r in cost_rows if r["method"] == "tkg_dtn")
    wireless = next(r for r in cost_rows if r["method"] == "wireless_kb")
    audit["tkg_relative_reductions_pct"] = {
        field: 100 * (1 - tkg[field + "_mean"] / wireless[field + "_mean"])
        for field in cost_fields
    }
    audit["agent_aggregate_over_450_requests"] = {}
    for method in ("tkg_dtn", "wireless_kb"):
        rows = [r for r in trial_rows if r["method"] == method]
        audit["agent_aggregate_over_450_requests"][method] = {
            field: sum(r[field] for r in rows) for field in
            ("users_processed", "llm_calls", "api_attempts", "llm_failures", "input_tokens", "output_tokens", "total_tokens", "total_decision_seconds")
        }
    audit["tkg_logged_tool_totals"] = {
        "all": sum(r["logged_tool_calls"] for r in trial_rows if r["method"] == "tkg_dtn"),
        "model_origin": sum(r["model_origin_tool_calls"] for r in trial_rows if r["method"] == "tkg_dtn"),
        "program_prefetch": sum(r["program_prefetch_tool_calls"] for r in trial_rows if r["method"] == "tkg_dtn"),
    }
    audit["all_methods_total_llm_failures"] = sum(r["llm_failures"] for r in trial_rows)
    audit["tkg_tool_counts_by_name"] = dict(sum((Counter(r["tool_counts_by_name"]) for r in audit["trials"].values()), Counter()))
    audit["tool_error_count"] = sum(len(r["tool_error_records"]) for r in audit["trials"].values())
    audit["expected_infeasible_admission_plans"] = sum(r["expected_infeasible_admission_plans"] for r in audit["trials"].values())
    audit["infeasible_plan_interpretation"] = "Exact tools correctly identify capacity-infeasible requests and return certificates; these are expected planning outcomes, not execution exceptions or invalid committed allocations."
    audit["graph_totals"] = {field: sum(r[field] for r in audit["graph_checks"].values())
                             for field in ("events", "event_hashes_reproduced", "decision_states_reproduced")}
    audit["scene_statistics"] = scene_rows
    audit["summary_statistics"] = summary_rows
    audit["per_repeat_costs_over_90_requests"] = repeat_costs
    audit["cost_statistics"] = cost_rows
    audit["changes_from_r001"] = []
    for group in summary_rows:
        first = data[group["scenario"], group["method"], 1][2]
        changed = {field: {"r001": first[field], "five_repeat_mean": group[field + "_mean"], "five_repeat_sd": group[field + "_sd"]}
                   for field in service_fields if first[field] is not None and (
                       not math.isclose(first[field], group[field + "_mean"], abs_tol=1e-9) or group[field + "_sd"] > 1e-9)}
        audit["changes_from_r001"].append({"scenario": group["scenario"], "method": group["method"], "changed_service_metrics": changed})
    write_csv("trial_metrics.csv", trial_rows)
    write_csv("summary_statistics.csv", summary_rows)
    write_csv("paired_steps.csv", paired_rows)
    write_csv("scene_statistics.csv", scene_rows)
    write_csv("cost_statistics.csv", cost_rows)
    (DEST / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "complete_trials": 60, "audited_steps": 1800,
                      "paired": audit["paired_totals"], "graphs": audit["graph_totals"],
                      "costs": cost_rows, "reductions_pct": audit["tkg_relative_reductions_pct"],
                      "tools": audit["tkg_logged_tool_totals"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
