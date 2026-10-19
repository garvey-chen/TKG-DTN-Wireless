"""Paired dynamic replay of all 30 users, importing the exact static terminal.

This runner preserves prior experiments. Frozen and controlled runs share the
same imported state and anchored channel observations; no admission is retried.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import time

from run_dynamic_agent import (
    ROOT, SCENES, V2, METRIC_FIELDS, LoggedLLM, load_config, load_llm_settings,
    public_settings, metric_row,
)
from run_dynamic_graph import read_csv, sha256, write_csv, write_json
from run_fair_dynamic_agent import validate_timeline, verify_traffic_ledger
from dynamic_tkg.experiment_report import usage_summary
from continuation.fairness_graph import FairnessWirelessGraph
from continuation.fair_agent import FairDynamicTKGPolicy
from continuation.fairness import optimize_fair_service
from dynamic_tkg.fairness_export import export_graph


def file_hashes(directory):
    return {str(p.relative_to(directory)): sha256(p)
            for p in sorted(directory.rglob("*")) if p.is_file()}


def verify_hashes(directory, hashes):
    for name, digest in hashes.items():
        if sha256(directory / name) != digest:
            raise AssertionError(f"Evidence changed: {directory.name}/{name}")


def paired_baseline(directory, inputs, handoff):
    record = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if (record["status"] != "complete" or record["controller"] != "frozen"
            or record["initialization_mode"] != "static_terminal"
            or record["observed_users_per_scene"] != 30):
        raise ValueError("Require a completed frozen static-terminal baseline")
    if record["input_sha256"] != inputs or record["static_terminal_sha256"] != handoff:
        raise ValueError("Baseline channels/trajectories/static handoff differ")
    verify_hashes(directory, record["output_sha256"])
    initial = {}
    for line in (directory / "frames_post_control.jsonl").read_text(encoding="utf-8").splitlines():
        state = json.loads(line)
        if state["frame_index"] == 0:
            initial[state["scenario"]] = state
    if set(initial) != set(SCENES):
        raise ValueError("Incomplete paired initial states")
    return record, initial


def verify_inherited_state(state, handoff, *, initial=False):
    """Independent invariant checks using exported static CSV, not optimizer."""
    scene = state["scenario"]
    source = {r["user_id"]: r for r in read_csv(handoff / scene / "static_terminal_users.csv")}
    if len(source) != 30 or set(state["users"]) != set(source):
        raise AssertionError("The full 30-user observation population was not retained")
    admitted = {uid for uid, row in source.items() if int(row["legacy_admitted"])}
    if set(state["allocations"]) != admitted:
        raise AssertionError("Static admission membership changed")
    for uid, row in source.items():
        user, contract = state["users"][uid], state["contracts"][uid]
        service = user["service"]
        if contract["slice_type"] != row["ground_truth"] or user["request"] != row["request"]:
            raise AssertionError("Inherited true contract/request changed")
        if uid not in admitted:
            if service["admitted"] or service["rate_mbps"] != 0 or service["effective"]:
                raise AssertionError("Rejected static user became admitted")
        else:
            allocation = state["allocations"][uid]
            if allocation["slice_type"] != row["assigned_slice"]:
                raise AssertionError("Inherited assigned slice changed")
            if row["assigned_slice"] != row["ground_truth"] and service["effective"]:
                raise AssertionError("Inherited wrong slice counted as effective")
            if initial and float(allocation["bandwidth"]) != float(row["bandwidth_mhz"]):
                raise AssertionError("Initial static bandwidth changed")
        if initial:
            values = [(user["cqi"], row["original_cqi"]),
                      (user["snr_db"], row["original_snr_db"]),
                      (user["rx_power_dbm"], row["original_rx_power_dbm"]),
                      (user["los"], row["original_los"])]
            values += list(zip(user["location"], [row["original_x"], row["original_y"], row["original_z"]]))
            if any(float(a) != float(b) for a, b in values):
                raise AssertionError(f"Initial radio state differs: {scene}/{uid}")
            if not math.isclose(service["rate_mbps"], float(row["source_rate_mbps"]), abs_tol=1e-10):
                raise AssertionError("Initial static modeled rate changed")


def numeric_step(graph, scene, output):
    started = time.perf_counter()
    state, history = graph.current_state(scene), graph.planning_history(scene)
    plan = optimize_fair_service(state["allocations"],
        {uid: user["cqi"] for uid, user in state["users"].items()}, state["contracts"],
        history, horizon_s=graph.horizon_s)
    if not plan["resource_valid"]:
        raise ValueError(f"No hard-legal resource plan: {scene}: {plan['violations']}")
    decision = {"controller": "numeric_validation", "agent_executed": False,
        "scenario": scene, "frame_index": state["frame_index"], "time_s": state["time_s"],
        "based_on_revision": state["revision"], "objective_proposed": plan["objective"],
        "history_through_time_s": history["through_time_s"],
        "history_through_revision": history["through_revision"], "changes": plan["changes"]}
    if plan["allocations"] != state["allocations"]:
        after = graph.commit_allocations(scene, plan["allocations"],
            expected_revision=state["revision"], decision=decision)
        decision.update(status="reallocated", mapped_revision=after["revision"])
    else:
        decision.update(status="held", mapped_revision=state["revision"])
    decision["decision_wall_seconds"] = time.perf_counter() - started
    with (output / "numeric_decisions.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(decision, ensure_ascii=False, allow_nan=False) + "\n")
    return decision


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=("frozen", "agent", "numeric_validation"), required=True)
    parser.add_argument("--agent-mode", choices=("prefetched", "adaptive"), default="prefetched",
        help="Model evidence interface; prefetched preserves the existing policy")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/mobility_5min_static_terminal_20261014")
    parser.add_argument("--static-terminal-dir", type=Path, default=ROOT / "data/static_terminal_handoff_20261014")
    parser.add_argument("--baseline-dir", type=Path,
        default=ROOT / "results/dynamic_frozen_30users_static_terminal_20261014_r001")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--credential-config", type=Path, default=ROOT.parent / "config/config.yaml")
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--max-calls", type=int, default=400)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if min(args.max_iterations, args.max_calls) < 1:
        parser.error("Iteration and call limits must be positive")
    input_dir, handoff, output = args.input_dir.resolve(), args.static_terminal_dir.resolve(), args.output.resolve()
    inputs, static_hashes = file_hashes(input_dir), file_hashes(handoff)
    timeline = read_csv(input_dir / "timeline.csv")
    validate_timeline(timeline, input_dir, inputs)
    generation = json.loads((input_dir / "generation.json").read_text(encoding="utf-8"))
    baseline, initial = (None, {}) if args.controller == "frozen" else paired_baseline(
        args.baseline_dir.resolve(), inputs, static_hashes)
    settings = None
    if args.controller == "agent":
        settings = load_llm_settings(load_config(V2 / "configs/default.json"), args.credential_config)
        settings["max_calls_per_run"] = args.max_calls
    if args.preflight_only:
        print(json.dumps({"preflight": "passed", "controller": args.controller,
            "agent_mode": args.agent_mode if args.controller == "agent" else None,
            "agent_executed": False, "observed_users_per_scene": 30,
            "settings": public_settings(settings) if settings else None}, ensure_ascii=False))
        return 0
    if output.exists():
        parser.error(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    sources = sorted(set([Path(__file__), ROOT / "run_dynamic_graph.py", ROOT / "run_dynamic_agent.py",
        ROOT / "run_fair_dynamic_agent.py", ROOT / "continuation/source_origin.json",
        *sorted((ROOT / "continuation").glob("*.py")),
        *sorted((ROOT / "dynamic_tkg").glob("*.py")), V2 / "wireless_exp/llm.py",
        V2 / "wireless_exp/settings.py", *sorted((V2 / "wireless_exp/tkg").glob("*.py"))]))
    record = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "controller": args.controller, "agent_executed": args.controller == "agent", "repeats": 1,
        "agent_mode": args.agent_mode if args.controller == "agent" else None,
        "max_iterations_per_decision": args.max_iterations,
        "decision_interface": ("Summary entry; model-selected evidence queries and exact previews; explicit guarded commit/hold"
            if args.controller == "agent" and args.agent_mode == "adaptive" else "Existing prefetched interface or non-model control"),
        "observed_users_per_scene": 30, "initial_users_per_scene": 30,
        "initial_admitted_users": {"center": 27, "north": 30, "south": 29},
        "initialization_mode": "static_terminal", "backend": "sqlite_local_property_graph",
        "neo4j_import_executed": False, "planning_horizon_s": 10, "history_window_s": None,
        "initialization": "Exact import of static TKG-DTN r001 after 30 arrivals; no numerical reinitialization, re-admission, slice correction, or t0 action.",
        "channel_semantics": "Static SNR/RX baseline plus per-user changes from the existing movement model; explicit calibration, not a recovered historical propagation configuration.",
        "traffic_semantics": "Modeled saturated/backlogged previous post-control Mbps times interval seconds; no packet measurement; dynamic cumulative history starts at zero.",
        "control_timing": "Synchronous at t=10..290; t0 imported and t300 closes history without control. API wall time recorded separately from radio time.",
        "objective": ["maximize_current_effective_users", "maximize_projected_historical_sum_log_utility",
            "maximize_effective_throughput", "minimize_bandwidth_change"],
        "fixed_membership": "Retain all 30 observations; rejected users stay rejected. Existing assigned slices and true contracts fixed, including inherited classification errors.",
        "evaluation_population": "Primary 30 observed users including rejected users at zero rate; secondary admitted cohort 27/30/29. Wrong-slice traffic enters raw throughput and Jain, never effective service.",
        "input_dir": str(input_dir), "input_sha256": inputs, "static_terminal_dir": str(handoff),
        "static_terminal_sha256": static_hashes, "model_settings": public_settings(settings) if settings else None,
        "source_sha256": {str(p.relative_to(ROOT.parent)): sha256(p) for p in sources}}
    if baseline:
        if baseline["source_sha256"] != record["source_sha256"]:
            raise AssertionError("Paired baseline and controller require identical source code")
        record.update(baseline_dir=str(args.baseline_dir.resolve()),
            baseline_manifest_sha256=sha256(args.baseline_dir / "run.json"),
            baseline_output_sha256=baseline["output_sha256"])
    for source in sources:
        target = output / "source_snapshot" / source.relative_to(ROOT.parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    write_json(output / "run.json", record)
    graph = llm = None
    previous, states, metrics, pre_metrics, decisions = {}, [], [], [], []
    started = time.perf_counter()
    try:
        graph = FairnessWirelessGraph(output / "graph.sqlite", static_terminal_dir=handoff)
        llm = LoggedLLM(settings, output / "llm_calls.jsonl") if settings else None
        policy_type = FairDynamicTKGPolicy
        if args.controller == "agent" and args.agent_mode == "adaptive":
            from continuation.adaptive_agent import AdaptiveFairDynamicTKGPolicy
            policy_type = AdaptiveFairDynamicTKGPolicy
        policy = policy_type(graph, llm, output, max_iterations=args.max_iterations) if llm else None
        with (output / "frames_pre_control.jsonl").open("w", encoding="utf-8") as pre_file, \
             (output / "frames_post_control.jsonl").open("w", encoding="utf-8") as post_file, \
             (output / "traffic_history.jsonl").open("w", encoding="utf-8") as history_file:
            for timing in timeline:
                index, timestamp = int(timing["Frame_Index"]), float(timing["Time_s"])
                for scene in SCENES:
                    prior = previous.get(scene)
                    meta = generation.get("scenarios", {}).get(scene, {})
                    state = graph.sync_frame(scene, index, timestamp,
                        read_csv(input_dir / timing[f"{scene}_csv"]),
                        expected_revision=prior["revision"] if prior else 0, target_users=30,
                        initialization_mode="static_terminal",
                        scene_info={k: meta[k] for k in ("tx_position_m", "geographic_origin_lat_lon", "map_sha256") if k in meta})
                    verify_inherited_state(state, handoff, initial=index == 0)
                    if index == 0 and initial:
                        for key in ("allocations", "contracts", "users", "metrics"):
                            if state[key] != initial[scene][key]:
                                raise AssertionError(f"Paired initial state differs: {scene}/{key}")
                    pre_file.write(json.dumps(state, ensure_ascii=False, allow_nan=False) + "\n")
                    pre_metrics.append(metric_row(state, args.controller, "pre_control"))
                    decision = {"status": "frozen" if args.controller == "frozen" else
                        ("initialization" if index == 0 else "terminal_observation")}
                    if args.controller != "frozen" and 0 < index < 30:
                        trigger = {"type": "periodic_history_fairness", "interval_s": 10,
                            "admitted_cqi_changed": [uid for uid in state["allocations"]
                                if state["users"][uid]["cqi"] != prior["users"][uid]["cqi"]]}
                        decision = policy.step(scene, trigger) if policy else numeric_step(graph, scene, output)
                        decisions.append(decision)
                    after = graph.current_state(scene)
                    verify_inherited_state(after, handoff, initial=index == 0)
                    if args.controller == "frozen" and prior and after["allocations"] != prior["allocations"]:
                        raise AssertionError("Frozen baseline allocation changed")
                    post_file.write(json.dumps(after, ensure_ascii=False, allow_nan=False) + "\n")
                    history = graph.traffic_history(scene)
                    history_file.write(json.dumps({k: v for k, v in history.items() if k != "intervals"},
                        ensure_ascii=False, allow_nan=False) + "\n")
                    for handle in (pre_file, post_file, history_file):
                        handle.flush()
                    previous[scene] = after
                    states.append(after)
                    metrics.append(metric_row(after, args.controller, "post_control"))
                    write_json(output / "progress.json", {"completed_scene_frames": len(states),
                        "scenario": scene, "frame_index": index, "time_s": timestamp,
                        "decisions": len(decisions), "llm_statistics": llm.statistics() if llm else None})
                    print(f"{scene} t={timestamp:g}s observed=30 admitted={after['metrics']['admitted_users']} "
                        f"effective={state['metrics']['effective_users']}->{after['metrics']['effective_users']} decision={decision['status']}", flush=True)
        record["traffic_validation"] = verify_traffic_ledger(graph, states)
        events = [event for scene in SCENES for event in graph.history(scene)]
        write_csv(output / "frame_metrics.csv", ["method", "phase", *METRIC_FIELDS], metrics)
        write_csv(output / "frame_metrics_pre_control.csv", ["method", "phase", *METRIC_FIELDS], pre_metrics)
        write_csv(output / "service_events.csv", ["event_id", "scenario", "frame_index", "time_s", "revision",
            "user_id", "event_type", "episode_id", "before", "after", "reason"], events)
        intervals = []
        for scene in SCENES:
            history = graph.traffic_history(scene)
            intervals.extend(dict(scenario=scene, **row) for row in history["intervals"])
            write_json(output / f"final_traffic_history_{scene}.json", history)
        write_csv(output / "traffic_intervals.csv", list(intervals[0]), intervals)
        if graph.db.execute("PRAGMA integrity_check").fetchone()[0] != "ok" or graph.db.execute("PRAGMA foreign_key_check").fetchall():
            raise AssertionError("Graph integrity check failed")
        record["graph"] = export_graph(output / "graph.sqlite", output)
        record["graph"]["artifacts"] = {name: name for name in record["graph"]["artifacts"]}
        usage = usage_summary(output)
        if llm:
            for key in ("llm_calls", "api_attempts", "prompt_tokens", "completion_tokens"):
                if usage["total"][key] != llm.statistics()[key]:
                    raise AssertionError(f"Usage counters differ: {key}")
        else:
            usage["accounting"]["source"] = "No model executed; zero actual API calls and tokens."
        write_json(output / "model_usage.json", usage)
        verify_hashes(input_dir, inputs)
        verify_hashes(handoff, static_hashes)
        verify_hashes(ROOT.parent, record["source_sha256"])
        if baseline:
            paired_baseline(args.baseline_dir.resolve(), inputs, static_hashes)
        record.update(status="complete", completed_scene_frames=len(states), decision_count=len(decisions),
            model_usage=usage["total"], llm_statistics=llm.statistics() if llm else None,
            elapsed_wall_seconds=time.perf_counter() - started,
            ended_at_utc=datetime.now(timezone.utc).isoformat())
    except KeyboardInterrupt:
        record.update(status="interrupted", completed_scene_frames=len(states),
            decision_count=len(decisions), error_type="KeyboardInterrupt",
            error="Run interrupted; partial records are not a completed experiment.",
            elapsed_wall_seconds=time.perf_counter() - started,
            llm_statistics=llm.statistics() if llm else None,
            ended_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(output / "model_usage_partial.json", usage_summary(output))
        raise
    except Exception as exc:
        error = str(exc).replace(settings["api_key"], "[REDACTED]") if settings else str(exc)
        record.update(status="failed", completed_scene_frames=len(states), error_type=type(exc).__name__,
            error=error, elapsed_wall_seconds=time.perf_counter() - started)
        write_json(output / "model_usage_partial.json", usage_summary(output))
        raise
    finally:
        if graph:
            graph.close()
        if llm:
            llm.session.close()
        record["output_sha256"] = {name: digest for name, digest in file_hashes(output).items() if name != "run.json"}
        write_json(output / "run.json", record)
    print(f"Completed {args.controller}: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
