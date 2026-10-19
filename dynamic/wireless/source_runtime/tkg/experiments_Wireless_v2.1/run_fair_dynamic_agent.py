"""Cumulative traffic fairness extension; real agent or explicitly numeric validation."""

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
    public_settings, baseline_preflight, metric_row,
)
from run_dynamic_graph import read_csv, sha256, write_csv, write_json, summarize
from dynamic_tkg.experiment_report import usage_summary
from dynamic_tkg.fairness import optimize_fair_service
from dynamic_tkg.fairness_graph import FairnessWirelessGraph
from dynamic_tkg.fair_agent import FairDynamicTKGPolicy
from dynamic_tkg.fairness_export import export_graph
from dynamic_tkg.service import INITIALIZATION_MODES, INITIALIZATION_DESCRIPTIONS


def validate_timeline(timeline, input_dir, hashes):
    """Bind every consumed frame to the paired baseline and the 10 s protocol."""
    if len(timeline) != 31:
        raise ValueError("This protocol requires 31 observations from 0 through 300 seconds")
    for index, row in enumerate(timeline):
        if int(row["Frame_Index"]) != index or float(row["Time_s"]) != index * 10:
            raise ValueError("Timeline must be consecutive 10 s frames from t=0 to t=300")
        for scene in SCENES:
            path = (input_dir / row[f"{scene}_csv"]).resolve()
            if not path.is_relative_to(input_dir):
                raise ValueError("Snapshot path is outside the paired input directory")
            name = str(path.relative_to(input_dir))
            if name not in hashes or sha256(path) != hashes[name]:
                raise ValueError(f"Unverified paired input frame: {name}")


def numeric_step(graph, scene, output):
    """Exercise the mathematical tool, never pretend to be an LLM decision."""
    started = time.perf_counter()
    state = graph.current_state(scene)
    history = graph.planning_history(scene)
    plan = optimize_fair_service(
        state["allocations"], {uid: u["cqi"] for uid, u in state["users"].items()},
        state["contracts"], history, horizon_s=graph.horizon_s)
    if not plan["resource_valid"]:
        raise ValueError(f"No resource-valid allocation for {scene}: {plan['violations']}")
    decision = {
        "controller": "numeric_validation", "agent_executed": False,
        "scenario": scene, "frame_index": state["frame_index"], "time_s": state["time_s"],
        "based_on_revision": state["revision"],
        "history_through_time_s": history["through_time_s"],
        "history_through_revision": history["through_revision"],
        "objective_proposed": plan["objective"], "changes": plan["changes"],
    }
    if plan["allocations"] != state["allocations"]:
        after = graph.commit_allocations(scene, plan["allocations"],
                                         expected_revision=state["revision"], decision=decision)
        decision.update(status="reallocated", mapped_revision=after["revision"])
    else:
        decision.update(status="held", mapped_revision=state["revision"],
                        reason="Existing allocation is optimal under the ordered objectives")
    decision["decision_wall_seconds"] = time.perf_counter() - started
    with (output / "numeric_decisions.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(decision, ensure_ascii=False, allow_nan=False) + "\n")
    return decision


def verify_traffic_ledger(graph, states):
    """Independently integrate saved final rates, never reuse planner history."""
    total_intervals = 0
    for scene in SCENES:
        selected = sorted((s for s in states if s["scenario"] == scene), key=lambda s: s["time_s"])
        history = graph.traffic_history(scene)
        if history["duration_s"] != 300 or history["completed_intervals"] != 30 * len(selected[0]["allocations"]):
            raise AssertionError("Missing/extra traffic intervals or experiment duration")
        for uid in selected[0]["allocations"]:
            traffic = effective_time = deficit = 0.0
            for before, after in zip(selected, selected[1:]):
                duration = after["time_s"] - before["time_s"]
                service = before["users"][uid]["service"]
                traffic += service["rate_mbps"] * duration
                effective_time += duration * bool(service["effective"])
                deficit += max(0, service["rate_min_mbps"] - service["rate_mbps"]) * duration
            actual = history["users"][uid]
            for key, expected in (("traffic_mbit", traffic), ("effective_time_s", effective_time),
                                  ("deficit_mbit", deficit)):
                if not math.isclose(actual[key], expected, rel_tol=1e-10, abs_tol=1e-8):
                    raise AssertionError(f"Traffic ledger mismatch: {scene}/{uid}/{key}")
        total_intervals += history["completed_intervals"]
    return {"passed": True, "independent_post_control_rate_integral": True,
            "completed_user_intervals": total_intervals, "duration_s_per_scene": 300}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=("agent", "numeric_validation"), default="agent")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/mobility_5min")
    parser.add_argument("--baseline-dir", type=Path, default=ROOT / "results/dynamic_graph_5min_20users")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--initial-users", type=int, default=20)
    parser.add_argument("--initialization-mode", choices=INITIALIZATION_MODES,
                        default="feasible_admission",
                        help="Must match the frozen baseline; fixed_population retains all requested users even when some initial service minima are unmet")
    parser.add_argument("--history-window-s", type=float, default=None,
                        help="Omit for cumulative history from t=0 (default); optional trailing window >=10 s")
    parser.add_argument("--credential-config", type=Path, default=ROOT.parent / "config/config.yaml")
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--max-calls", type=int, default=400)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if min(args.initial_users, args.max_iterations, args.max_calls) < 1:
        parser.error("Users, iteration limit and call limit must be positive")
    if args.history_window_s is not None and (not math.isfinite(args.history_window_s) or args.history_window_s < 10):
        parser.error("History window must be finite and at least 10 seconds")
    input_dir, baseline_dir = args.input_dir.resolve(), args.baseline_dir.resolve()
    baseline, initial_states = baseline_preflight(baseline_dir, input_dir, args.initial_users,
                                                 initialization_mode=args.initialization_mode)
    timeline = read_csv(input_dir / "timeline.csv")
    validate_timeline(timeline, input_dir, baseline["input_sha256"])
    generation = json.loads((input_dir / "generation.json").read_text(encoding="utf-8"))
    settings = None
    if args.controller == "agent":
        config = load_config(V2 / "configs/default.json")
        settings = load_llm_settings(config, args.credential_config if args.credential_config.is_file() else None)
        settings["max_calls_per_run"] = args.max_calls
    if args.preflight_only:
        print(json.dumps({"preflight": "passed", "controller": args.controller, "agent_executed": False,
                          "initialization_mode": args.initialization_mode,
                          "history_window_s": args.history_window_s, "planned_active_decisions": 87,
                          "settings": public_settings(settings) if settings else None}, ensure_ascii=False))
        return 0
    from dynamic_tkg.fairness_report import write_fairness_report
    output = (args.output or ROOT / "results" / (
        f"dynamic_fair_{args.controller}_" + datetime.now().strftime("%Y%m%d_%H%M%S"))).resolve()
    if output.exists():
        parser.error(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    record = {
        "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "controller": args.controller, "agent_executed": args.controller == "agent",
        "numeric_validation_is_agent_experiment": False, "repeats": 1,
        "initial_users_per_scene": args.initial_users, "history_window_s": args.history_window_s,
        "initialization_mode": args.initialization_mode,
        "planning_horizon_s": 10, "backend": "sqlite_local_property_graph",
        "neo4j_import_executed": False,
        "objective": ["maximize_current_effective_users", "maximize_projected_historical_sum_log_utility",
                      "maximize_effective_throughput", "minimize_bandwidth_change"],
        "fairness_evaluation": "Jain of each user's completed historical mean throughput; also demand normalized and per slice. This is not the exact optimized utility.",
        "trigger": "Every 10 s at t=10..290, even if CQI is unchanged; t=0 preserves paired initialization; t=300 closes history only.",
        "initialization": "Same fixed cohort, service contracts, slices, budgets and t0 allocations as the frozen baseline.",
        "initialization_details": INITIALIZATION_DESCRIPTIONS[args.initialization_mode],
        "initialization_summary": baseline.get("initialization_summary"),
        "traffic_semantics": "Simulated saturated/backlogged traffic: previous final post-control rate(Mbps) times completed interval(s) = Mbit; never measured packet traffic.",
        "control_timing": "Synchronous same-frame control. Post-control rate held over the next interval; API wall time reported separately, not injected as radio delay.",
        "forecast_semantics": "Current CQI held for the next 10 s planning projection only; forecasts never enter observed history.",
        "input_dir": str(input_dir), "input_sha256": baseline["input_sha256"],
        "baseline_dir": str(baseline_dir), "baseline_manifest_sha256": sha256(baseline_dir / "manifest.json"),
        "baseline_output_sha256": baseline["output_sha256"],
        "model_settings": public_settings(settings) if settings else None,
        "max_iterations_per_decision": args.max_iterations,
    }
    sources = [Path(__file__), ROOT / "run_dynamic_agent.py", ROOT / "run_dynamic_graph.py",
               *sorted((ROOT / "dynamic_tkg").glob("*.py")), V2 / "wireless_exp/llm.py",
               V2 / "wireless_exp/settings.py", *sorted((V2 / "wireless_exp/tkg").glob("*.py"))]
    record["source_sha256"] = {str(p.relative_to(ROOT.parent)): sha256(p) for p in sources}
    for source in sources:
        target = output / "source_snapshot" / source.relative_to(ROOT.parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    write_json(output / "run.json", record)
    graph = FairnessWirelessGraph(output / "graph.sqlite", history_window_s=args.history_window_s)
    llm = LoggedLLM(settings, output / "llm_calls.jsonl") if settings else None
    policy = FairDynamicTKGPolicy(graph, llm, output, max_iterations=args.max_iterations) if llm else None
    previous, states, metrics, pre_metrics, decisions = {}, [], [], [], []
    started = time.perf_counter()
    try:
        with (output / "frames_pre_control.jsonl").open("w", encoding="utf-8") as pre_file, \
             (output / "frames_post_control.jsonl").open("w", encoding="utf-8") as post_file, \
             (output / "traffic_history.jsonl").open("w", encoding="utf-8") as traffic_file:
            for timing in timeline:
                index, timestamp = int(timing["Frame_Index"]), float(timing["Time_s"])
                for scene in SCENES:
                    prior = previous.get(scene)
                    meta = generation.get("scenarios", {}).get(scene, {})
                    state = graph.sync_frame(
                        scene, index, timestamp, read_csv(input_dir / timing[f"{scene}_csv"]),
                        expected_revision=prior["revision"] if prior else 0, target_users=args.initial_users,
                        initialization_mode=args.initialization_mode,
                        scene_info={k: meta[k] for k in ("tx_position_m", "geographic_origin_lat_lon", "map_sha256") if k in meta})
                    if index == 0:
                        for key in ("allocations", "contracts", "users", "metrics"):
                            if state[key] != initial_states[scene][key]:
                                raise AssertionError(f"Paired initial state differs: {scene}/{key}")
                    pre_file.write(json.dumps(state, ensure_ascii=False, allow_nan=False) + "\n")
                    pre_metrics.append(metric_row(state, args.controller, "pre_control"))
                    decision = {"status": "initialization" if index == 0 else "terminal_observation"}
                    if 0 < index < len(timeline) - 1:
                        changed = [uid for uid in state["allocations"]
                                   if state["users"][uid]["cqi"] != prior["users"][uid]["cqi"]]
                        trigger = {"type": "periodic_history_fairness", "interval_s": 10,
                                   "admitted_cqi_changed": changed}
                        decision = (policy.step(scene, trigger) if policy
                                    else numeric_step(graph, scene, output))
                        decisions.append(decision)
                    after = graph.current_state(scene)
                    post_file.write(json.dumps(after, ensure_ascii=False, allow_nan=False) + "\n")
                    history = graph.traffic_history(scene)  # evaluation always cumulative
                    traffic_file.write(json.dumps({k: v for k, v in history.items() if k != "intervals"},
                                                  ensure_ascii=False, allow_nan=False) + "\n")
                    for handle in (pre_file, post_file, traffic_file):
                        handle.flush()
                    previous[scene] = after
                    states.append(after)
                    metrics.append(metric_row(after, args.controller, "post_control"))
                    write_json(output / "progress.json", {"completed_scene_frames": len(states),
                        "scenario": scene, "frame_index": index, "time_s": timestamp,
                        "controller": args.controller, "decisions": len(decisions),
                        "llm_statistics": llm.statistics() if llm else None})
                    print(f"{scene} t={timestamp:g}s effective={state['metrics']['effective_users']}->{after['metrics']['effective_users']} "
                          f"decision={decision['status']}", flush=True)
        record["traffic_validation"] = verify_traffic_ledger(graph, states)
        events = [e for scene in SCENES for e in graph.history(scene)]
        write_json(output / "summary.json", summarize(states, events))
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
            raise AssertionError("Fairness graph failed integrity checks")
        record["graph"] = export_graph(output / "graph.sqlite", output)
        record["graph"]["artifacts"] = {name: name for name in record["graph"]["artifacts"]}
        usage = usage_summary(output)
        if llm:
            for key in ("llm_calls", "api_attempts", "prompt_tokens", "completion_tokens"):
                if usage["total"][key] != llm.statistics()[key]:
                    raise AssertionError(f"Usage logs disagree with gateway counters: {key}")
        else:
            usage["accounting"]["source"] = "Numeric tool validation only. No model/API executed; zero actual API/token consumption."
        write_json(output / "model_usage.json", usage)
        record["comparison"] = write_fairness_report(output, baseline_dir, args.controller, usage=usage,
                                                     plots=not args.no_plots)
        baseline_preflight(baseline_dir, input_dir, args.initial_users,
                           initialization_mode=args.initialization_mode)
        for name, digest in record["source_sha256"].items():
            if sha256(ROOT.parent / name) != digest:
                raise AssertionError(f"Source changed during run: {name}")
        record.update(status="complete", completed_scene_frames=len(states), decision_count=len(decisions),
                      model_usage=usage["total"], llm_statistics=llm.statistics() if llm else None,
                      elapsed_wall_seconds=time.perf_counter() - started,
                      ended_at_utc=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        error = str(exc)
        if settings:
            error = error.replace(settings["api_key"], "[REDACTED]")
        record.update(status="failed", completed_scene_frames=len(states), error_type=type(exc).__name__,
                      error=error, elapsed_wall_seconds=time.perf_counter() - started)
        write_json(output / "model_usage_partial.json", usage_summary(output))
        raise
    finally:
        graph.close()
        if llm:
            llm.session.close()
        record["output_sha256"] = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*"))
                                   if p.is_file() and p.name != "run.json"}
        write_json(output / "run.json", record)
    print(f"Completed {args.controller}: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

