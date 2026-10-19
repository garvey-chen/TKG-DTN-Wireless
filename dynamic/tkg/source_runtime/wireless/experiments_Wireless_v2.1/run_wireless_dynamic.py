"""Original WirelessAgent StateGraph on the identical 30-user mobility experiment.

No temporal graph or traffic ledger is accessible to the policy. The external
evaluator integrates realized service after execution. In measured_delay mode,
the complete measured decision wall time determines the next eligible 10 s
actuation tick. At most one job per scene is in flight; busy ticks are dropped.
This is discrete-event latency simulation, not a real-time radio deployment.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import time

from run_dynamic_agent import V2, LoggedLLM, load_config, load_llm_settings, public_settings, metric_row
from run_dynamic_graph import ROOT, SCENES, METRIC_FIELDS, read_csv, sha256, write_csv, write_json
from run_static_terminal_dynamic import file_hashes, verify_hashes, verify_inherited_state
from run_fair_dynamic_agent import validate_timeline
from continuation.handoff import load_handoff
from continuation.service import initialize_cohort, evaluate_service, _jain
from continuation.graph import TemporalWirelessGraph
from continuation.planner import validate_allocations
from dynamic_tkg.experiment_report import usage_summary


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def due_tick(source_time, elapsed, interval=10.0):
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Decision time must be finite and nonnegative")
    return source_time + max(1, math.ceil(elapsed / interval)) * interval


def observe(scene, frame, timestamp, rows, terminal, previous=None):
    users = TemporalWirelessGraph._observations(rows, scene, frame, timestamp)
    if previous is None:
        initial = initialize_cohort(rows, static_terminal=terminal)
        state = {"scenario": scene, "initialization_mode": "static_terminal",
                 "allocations": initial["allocations"], "contracts": initial["contracts"],
                 "initialization_evidence": initial["initialization_evidence"]}
    else:
        if set(users) != set(previous["users"]):
            raise ValueError("Population changed")
        for row in rows:
            uid = str(row["RX_ID"])
            if (row["User_Request"] != previous["users"][uid]["request"]
                    or row["Request_Label"] != previous["contracts"][uid]["slice_type"]):
                raise ValueError("Immutable service request/contract changed")
        state = {key: deepcopy(previous[key]) for key in
                 ("scenario", "initialization_mode", "allocations", "contracts", "initialization_evidence")}
    state.update(frame_index=frame, time_s=timestamp, revision=(previous["revision"] + 1 if previous else 1),
                 phase="pre_control", users=users)
    return evaluate(state)


def evaluate(state):
    evaluated = evaluate_service(list(state["users"].values()), state["allocations"], state["contracts"])
    state["metrics"] = evaluated["metrics"]
    for uid, user in state["users"].items():
        user.update(admitted=evaluated["users"][uid]["admitted"], service=evaluated["users"][uid])
    return state


def apply_decision(state, decision):
    """Validate the actual current radio state; never optimize a stale proposal."""
    decision["applied_time_s"] = None
    decision["checked_time_s"] = state["time_s"]
    decision["staleness_s"] = state["time_s"] - decision["source_time_s"]
    if decision["policy_status"] not in ("approved", "committed", "held"):
        decision["status"] = "policy_failed"
        return state
    if decision["policy_status"] == "held":
        decision["status"] = "held"
        return state
    check = validate_allocations(decision["allocations"],
        {uid: u["cqi"] for uid, u in state["users"].items()}, state["contracts"], state["allocations"])
    if not check["resource_valid"]:
        decision.update(status="rejected_stale_or_invalid", validation_failure=check["violations"])
        return state
    if decision["allocations"] == state["allocations"]:
        decision.update(status="held", applied_time_s=state["time_s"])
        return state
    after = deepcopy(state)
    after.update(allocations=deepcopy(decision["allocations"]), revision=state["revision"] + 1,
                 phase="post_control")
    evaluate(after)
    # Objective claims from an old snapshot are not retroactively replaced by a
    # fresh optimizer. Minimum-rate breaches due to mobility remain observable.
    decision.update(status="reallocated", applied_time_s=state["time_s"],
                    applied_revision=after["revision"])
    return after


def integrate(previous, current, ledger, intervals):
    duration = current["time_s"] - previous["time_s"]
    if duration <= 0:
        raise ValueError("Nonpositive interval")
    for uid, user in previous["users"].items():
        service = user["service"]
        values = ledger.setdefault(uid, {"traffic_mbit": 0.0, "effective_served_mbit": 0.0,
            "effective_time_s": 0.0, "observed_duration_s": 0.0})
        volume = service["rate_mbps"] * duration
        values["traffic_mbit"] += volume
        values["effective_served_mbit"] += volume if service["effective"] else 0.0
        values["effective_time_s"] += duration if service["effective"] else 0.0
        values["observed_duration_s"] += duration
        intervals.append({"scenario": previous["scenario"], "user_id": uid,
            "start_time_s": previous["time_s"], "end_time_s": current["time_s"],
            "source_revision": previous["revision"], "modeled_rate_mbps": service["rate_mbps"],
            "service_effective": service["effective"], "traffic_mbit": volume,
            "effective_served_mbit": volume if service["effective"] else 0.0,
            "measured_packet_traffic": False})


def replay(*, timeline, input_dir, handoff, terminals, policy, mode, output, paired_initial,
           require_workflow=False):
    previous, pending, ledgers = {}, {}, {scene: {} for scene in SCENES}
    states, metrics, pre_metrics, decisions, intervals, scheduler = [], [], [], [], [], []

    def event(kind, scene, timestamp, **kwargs):
        value = dict(event=kind, scenario=scene, time_s=timestamp, **kwargs)
        scheduler.append(value)
        append_json(output / "scheduler_events.jsonl", value)

    def finish(decision):
        decisions.append(deepcopy(decision))
        append_json(output / "decisions.jsonl", decision)

    for timing in timeline:
        frame, timestamp = int(timing["Frame_Index"]), float(timing["Time_s"])
        for scene in SCENES:
            prior = previous.get(scene)
            state = observe(scene, frame, timestamp, read_csv(input_dir / timing[f"{scene}_csv"]),
                            terminals[scene], prior)
            verify_inherited_state(state, handoff, initial=frame == 0)
            if frame == 0:
                for key in ("allocations", "contracts", "metrics"):
                    if state[key] != paired_initial[scene][key]:
                        raise AssertionError(f"Unpaired static physical state: {scene}/{key}")
                for uid in state["users"]:
                    for key in ("cqi", "location", "request", "service"):
                        if state["users"][uid][key] != paired_initial[scene]["users"][uid][key]:
                            raise AssertionError(f"Unpaired initial user: {scene}/{uid}/{key}")
            else:
                integrate(prior, state, ledgers[scene], intervals)
            append_json(output / "frames_pre_control.jsonl", state)
            pre_metrics.append(metric_row(state, "wireless_kb_workflow", "pre_control"))
            if frame == len(timeline) - 1:
                if scene in pending:
                    job = pending.pop(scene)
                    job.update(status="terminal_drop", applied_time_s=None)
                    event("terminal_drop", scene, timestamp, decision_id=job["decision_id"])
                    finish(job)
            elif frame > 0:
                if scene in pending and pending[scene]["scheduled_apply_time_s"] <= timestamp:
                    job = pending.pop(scene)
                    state = apply_decision(state, job)
                    event(job["status"], scene, timestamp, decision_id=job["decision_id"])
                    finish(job)
                if scene in pending:
                    event("skipped_busy", scene, timestamp, decision_id=pending[scene]["decision_id"])
                else:
                    policy_started = time.perf_counter()
                    job = policy.step(deepcopy(state))
                    policy_elapsed = time.perf_counter() - policy_started
                    if require_workflow and (job.get("workflow_nodes") != ["initialize", "understand_intent",
                            "allocate_slice_type", "allocate_resources", "evaluate_network"]
                            or job.get("llm_calls", 0) < 4):
                        raise AssertionError("Original five-node workflow/actual model calls were not executed")
                    if require_workflow:
                        job["policy_reported_seconds"] = job["decision_wall_seconds"]
                        job["decision_wall_seconds"] = policy_elapsed
                    elapsed = float(job["decision_wall_seconds"])
                    if not math.isfinite(elapsed) or elapsed < 0:
                        raise ValueError("Invalid measured decision duration")
                    job.update(policy_status=job["status"], source_time_s=timestamp,
                        submitted_time_s=timestamp, completed_time_s=timestamp + elapsed,
                        source_revision=state["revision"], missed_deadline_10s=elapsed > 10,
                        scheduled_apply_time_s=(timestamp if mode == "synchronous" else due_tick(timestamp, elapsed)))
                    if mode == "synchronous":
                        state = apply_decision(state, job)
                        finish(job)
                    else:
                        pending[scene] = job
                    event("decision_submitted", scene, timestamp, decision_id=job["decision_id"],
                          elapsed_seconds=elapsed, scheduled_apply_time_s=job["scheduled_apply_time_s"])
            state["phase"] = "post_control"
            verify_inherited_state(state, handoff, initial=frame == 0)
            append_json(output / "frames_post_control.jsonl", state)
            previous[scene] = state
            states.append(state)
            metrics.append(metric_row(state, "wireless_kb_workflow", "post_control"))
            write_json(output / "progress.json", {"completed_scene_frames": len(states),
                "scenario": scene, "frame_index": frame, "time_s": timestamp,
                "finalized_decisions": len(decisions), "pending_decisions": len(pending)})
            print(f"{mode} {scene} t={timestamp:g}s effective={state['metrics']['effective_users']} "
                  f"decisions={len(decisions)} pending={len(pending)}", flush=True)
    if pending:
        raise AssertionError("Unfinalized decisions")
    for scene, ledger in ledgers.items():
        if len(ledger) != 30 or any(v["observed_duration_s"] != 300 for v in ledger.values()):
            raise AssertionError("Traffic must cover 30 users and exactly [0,300)")
        initial = paired_initial[scene]
        normalized = [v["traffic_mbit"] / 300 / initial["contracts"][uid]["min_rate_mbps"]
                      for uid, v in ledger.items()]
        write_json(output / f"final_traffic_history_{scene}.json", {
            "scenario": scene, "duration_s": 300, "users": ledger,
            "evaluation_only_not_policy_input": True,
            "jain_all_30_mean_rate": _jain([v["traffic_mbit"] / 300 for v in ledger.values()]),
            "jain_all_30_demand_normalized_mean_rate": _jain(normalized)})
    write_csv(output / "frame_metrics.csv", ["method", "phase", *METRIC_FIELDS], metrics)
    write_csv(output / "frame_metrics_pre_control.csv", ["method", "phase", *METRIC_FIELDS], pre_metrics)
    write_csv(output / "traffic_intervals.csv", list(intervals[0]), intervals)
    return {"completed_scene_frames": len(states), "decision_count": len(decisions),
        "skipped_busy_ticks": sum(e["event"] == "skipped_busy" for e in scheduler),
        "terminal_dropped_decisions": sum(d["status"] == "terminal_drop" for d in decisions),
        "rejected_stale_decisions": sum(d["status"] == "rejected_stale_or_invalid" for d in decisions),
        "traffic_validation": "30 users per scene; 30 forward intervals; t300 contributes zero duration"}


def main(argv=None):
    from wireless_dynamic.workflow_policy import ENGINE_MODES, ENGINE_OBJECTIVES, Policy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-mode", choices=("synchronous", "measured_delay"), required=True)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/mobility_5min_static_terminal_20261014")
    parser.add_argument("--static-terminal-dir", type=Path, default=ROOT / "data/wireless_static_terminal_handoff_20261015")
    parser.add_argument("--baseline-dir", type=Path, default=ROOT / "results/dynamic_frozen_30users_static_terminal_20261014_r001")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--credential-config", type=Path, default=ROOT.parent / "config/config.yaml")
    parser.add_argument("--max-calls", type=int, default=800)
    parser.add_argument("--resource-engine-mode", choices=ENGINE_MODES,
                        default="matched_snapshot_fair")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    input_dir, handoff, output = args.input_dir.resolve(), args.static_terminal_dir.resolve(), args.output.resolve()
    inputs, static_hashes = file_hashes(input_dir), file_hashes(handoff)
    terminals, _ = load_handoff(handoff)
    timeline = read_csv(input_dir / "timeline.csv")
    validate_timeline(timeline, input_dir, inputs)
    if [float(t["Time_s"]) for t in timeline] != list(range(0, 301, 10)):
        raise ValueError("Require the published 0..300 s experiment")
    baseline_dir = args.baseline_dir.resolve()
    baseline = json.loads((baseline_dir / "run.json").read_text(encoding="utf-8"))
    if (baseline["status"] != "complete" or baseline["controller"] != "frozen"
            or baseline["input_sha256"] != inputs):
        raise ValueError("Require completed identical-input frozen baseline")
    verify_hashes(baseline_dir, baseline["output_sha256"])
    initial = {}
    for line in (baseline_dir / "frames_post_control.jsonl").read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if value["frame_index"] == 0:
            initial[value["scenario"]] = value
    settings = load_llm_settings(load_config(V2 / "configs/default.json"), args.credential_config)
    settings["max_calls_per_run"] = args.max_calls
    if args.max_calls < 1:
        parser.error("Positive --max-calls required")
    if args.preflight_only:
        print(json.dumps({"preflight": "passed", "timing_mode": args.timing_mode,
            "resource_engine_mode": args.resource_engine_mode,
            "objective": ENGINE_OBJECTIVES[args.resource_engine_mode],
            "model_settings": public_settings(settings), "api_executed": False}, ensure_ascii=False))
        return 0
    if output.exists():
        parser.error(f"Refusing to overwrite existing output: {output}")
    sources = sorted(set([Path(__file__), ROOT / "prepare_wireless_static_handoff.py",
        ROOT / "run_dynamic_agent.py", ROOT / "run_dynamic_graph.py", ROOT / "run_fair_dynamic_agent.py",
        ROOT / "run_static_terminal_dynamic.py", *sorted((ROOT / "wireless_dynamic").glob("*.py")),
        ROOT / "wireless_dynamic/workflow_source_audit.json", ROOT / "wireless_dynamic/workflow_adaptation.diff",
        *sorted((ROOT / "continuation").glob("*.py")), *sorted((ROOT / "dynamic_tkg").glob("*.py")),
        V2 / "wireless_exp/llm.py", V2 / "wireless_exp/settings.py", V2 / "wireless_exp/policies.py",
        V2 / "wireless_exp/legacy/agent_kb.py", V2 / "data/knowledge/Intent_Understand.txt",
        V2 / "wireless_exp/shared_resources.py", V2 / "wireless_exp/tkg/resources.py",
        V2 / "wireless_exp/tkg/executor.py", V2 / "wireless_exp/tkg/graph_store.py",
        V2 / "wireless_exp/metrics.py",
        V2 / "configs/default.json"]))
    output.mkdir(parents=True)
    record = {"status": "running", "controller": "wireless_kb_workflow", "agent_executed": True,
        "comparison_eligibility": "original_stategraph_dynamic_adaptation",
        "resource_engine_mode": args.resource_engine_mode,
        "workflow_nodes": ["initialize", "understand_intent", "allocate_slice_type", "allocate_resources", "evaluate_network"],
        "workflow_semantics": "Full original five-node StateGraph and original multi-stage prompts; original conditional extra calls retained; resource tool commits only to an isolated candidate snapshot before final LLM summary. No added JSON approval gate.",
        "maintenance_focus": "Lowest current modeled rate / assigned-slice minimum-rate ratio among admitted users; stable numeric user-ID tie; no traffic history or change history.",
        "timing_mode": args.timing_mode, "repeats": 1, "observed_users_per_scene": 30,
        "initialization_mode": "static_terminal", "static_source_method": "wireless_kb",
        "initial_admitted_users": {scene: len(initial[scene]["allocations"]) for scene in SCENES},
        "input_dir": str(input_dir), "input_sha256": inputs,
        "static_terminal_dir": str(handoff), "static_terminal_sha256": static_hashes,
        "baseline_dir": str(baseline_dir), "baseline_manifest_sha256": sha256(baseline_dir / "run.json"),
        "model_settings": public_settings(settings), "backend": "snapshot_dict; external_offline_evaluator",
        "policy_information": ("Current telemetry, fixed assigned slices, original requests/KB/public resource rules; shared fixed service contracts available to the matched current-snapshot resource tool. No traffic history, temporal graph, past frames or future telemetry." if args.resource_engine_mode != "original_static" else
            "Current telemetry, fixed assigned slices, original requests/KB/public resource rules. Original resource tool uses assigned-slice constraints, not evaluation truth contracts; no traffic history, temporal graph, past frames or future telemetry."),
        "history_access": False, "dynamic_traffic_history_starts_at_zero": True,
        "objective": ENGINE_OBJECTIVES[args.resource_engine_mode],
        "fairness_scope": ("Instantaneous sum-log utility over fixed admitted users; no historical service "
                           "input; cumulative Jain is evaluation only." if args.resource_engine_mode
                           == "matched_snapshot_fair" else "No explicit fairness objective."),
        "timing_semantics": ("Synchronous t10..290; measured API cost does not advance radio time." if args.timing_mode == "synchronous" else
            "Discrete-event measured-delay model: source t plus complete measured policy duration; apply at first eligible 10s tick; single in-flight job per scene; busy ticks skipped; revalidate on arrival without reoptimizing; t300 drops pending work and launches none."),
        "traffic_semantics": "Saturated modeled Mbps integrated over [0,300); last observation adds no duration; not packet measurement.",
        "evaluation_population": "All 30 observed users including rejects at zero; inherited wrong slices never count as effective.",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {str(p.relative_to(ROOT.parent)): sha256(p) for p in sources}}
    for source in sources:
        target = output / "source_snapshot" / source.relative_to(ROOT.parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    write_json(output / "run.json", record)
    llm = LoggedLLM(settings, output / "llm_calls.jsonl")
    started = time.perf_counter()
    try:
        details = replay(timeline=timeline, input_dir=input_dir, handoff=handoff, terminals=terminals,
            policy=Policy(llm, output, resource_engine_mode=args.resource_engine_mode),
            mode=args.timing_mode, output=output, paired_initial=initial, require_workflow=True)
        usage = usage_summary(output)
        for key in ("llm_calls", "api_attempts", "prompt_tokens", "completion_tokens"):
            if usage["total"][key] != llm.statistics()[key]:
                raise AssertionError(f"Model accounting mismatch: {key}")
        write_json(output / "model_usage.json", usage)
        verify_hashes(input_dir, inputs)
        verify_hashes(handoff, static_hashes)
        verify_hashes(ROOT.parent, record["source_sha256"])
        record.update(status="complete", **details, llm_statistics=llm.statistics(), model_usage=usage["total"])
    except BaseException as exc:
        record.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error_type=type(exc).__name__,
                      error=str(exc).replace(settings["api_key"], "[REDACTED]"))
        write_json(output / "model_usage_partial.json", usage_summary(output))
        raise
    finally:
        llm.session.close()
        record.update(elapsed_wall_seconds=time.perf_counter() - started,
                      ended_at_utc=datetime.now(timezone.utc).isoformat())
        record["output_sha256"] = {name: digest for name, digest in file_hashes(output).items() if name != "run.json"}
        write_json(output / "run.json", record)
    print(f"Completed {args.timing_mode}: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
