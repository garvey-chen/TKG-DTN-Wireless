"""One real TKG-DTN run against the saved, identical frozen-allocation baseline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

from run_dynamic_graph import ROOT, SCENES, METRIC_FIELDS, read_csv, sha256, write_csv, write_json, summarize
from dynamic_tkg.experiment_report import write_report, usage_summary
from dynamic_tkg.service import INITIALIZATION_MODES

V2 = ROOT.parent / "experiments_Wireless_v2"
sys.path.insert(0, str(V2))
from wireless_exp.llm import LoggedLLM
from wireless_exp.settings import load_config, load_llm_settings, public_settings


def metric_row(state, method, phase):
    metrics = state["metrics"]
    return {"method": method, "phase": phase, "scenario": state["scenario"],
            "frame_index": state["frame_index"], "time_s": state["time_s"], "revision": state["revision"],
            **{k: metrics[k] for k in METRIC_FIELDS if k in metrics},
            "embb_used_mhz": metrics["slice_usage_mhz"]["eMBB"],
            "urllc_used_mhz": metrics["slice_usage_mhz"]["URLLC"]}


def baseline_preflight(directory, input_dir, initial_users, *, initialization_mode="feasible_admission"):
    if initialization_mode not in INITIALIZATION_MODES:
        raise ValueError(f"Unknown initialization_mode: {initialization_mode}")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["initial_users_per_scene"] != initial_users or manifest["agent_executed"]:
        raise ValueError("Baseline must be the requested frozen-allocation cohort")
    if manifest.get("initialization_mode", "feasible_admission") != initialization_mode:
        raise ValueError("Paired baseline initialization_mode differs from requested mode")
    for name, digest in manifest["input_sha256"].items():
        if sha256(input_dir / name) != digest:
            raise ValueError(f"Different paired input: {name}")
    for name, digest in manifest["output_sha256"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Baseline artifact changed: {name}")
    initial = {}
    with (directory / "frames.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            state = json.loads(line)
            if state["frame_index"] == 0:
                if state.get("initialization_mode", "feasible_admission") != initialization_mode:
                    raise ValueError("Baseline initial state has a different initialization_mode")
                if len(state["allocations"]) != initial_users:
                    raise ValueError("Baseline initial population differs from requested count")
                initial[state["scenario"]] = state
    if set(initial) != set(SCENES):
        raise ValueError("Require all three baseline scenarios")
    return manifest, initial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/mobility_5min")
    parser.add_argument("--baseline-dir", type=Path, default=ROOT / "results/dynamic_graph_5min_20users")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--initial-users", type=int, default=20)
    parser.add_argument("--credential-config", type=Path, default=ROOT.parent / "config/config.yaml")
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--max-calls", type=int, default=400)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--preflight-only", action="store_true", help="Check inputs/configuration/imports without API calls")
    args = parser.parse_args()
    if min(args.initial_users, args.max_iterations, args.max_calls) < 1:
        parser.error("Users, iteration limit and call limit must be positive")
    from dynamic_tkg.control import ControlledWirelessGraph
    from dynamic_tkg.agent import DynamicTKGPolicy
    from dynamic_tkg.export import export_graph

    input_dir, baseline_dir = args.input_dir.resolve(), args.baseline_dir.resolve()
    manifest, initial_states = baseline_preflight(baseline_dir, input_dir, args.initial_users)
    config = load_config(V2 / "configs/default.json")
    settings = load_llm_settings(config, args.credential_config if args.credential_config.is_file() else None)
    settings["max_calls_per_run"] = args.max_calls
    if args.preflight_only:
        print(json.dumps({"preflight": "passed", "agent_executed": False, "settings": public_settings(settings)}, ensure_ascii=False))
        return 0
    output = (args.output or ROOT / "results" / ("dynamic_tkg_agent_20users_" + datetime.now().strftime("%Y%m%d_%H%M%S"))).resolve()
    if output.exists():
        parser.error(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    record = {
        "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "agent_executed": True, "repeats": 1, "initial_users_per_scene": args.initial_users,
        "backend": "sqlite_local_property_graph", "neo4j_import_executed": False,
        "model_settings": public_settings(settings), "max_iterations_per_decision": args.max_iterations,
        "trigger": "After frame 0, invoke the model if any admitted user's CQI differs from the previous frame. Synchronize every frame regardless.",
        "initialization": "Identical source-order feasible t0 cohort and allocation as frozen baseline; no model action at t0.",
        "objective": "Maximize effective users, then effective throughput, with fixed cohort/slices and unchanged resource limits; fairness reported.",
        "control_timing": "Synchronous control after observation at the same simulation timestamp; no simulated API delay. Post-control state held until next sample.",
        "input_dir": str(input_dir), "input_sha256": manifest["input_sha256"],
        "baseline_dir": str(baseline_dir), "baseline_manifest_sha256": sha256(baseline_dir / "manifest.json"),
        "baseline_output_sha256": manifest["output_sha256"],
    }
    sources = [Path(__file__), ROOT / "run_dynamic_graph.py", *sorted((ROOT / "dynamic_tkg").glob("*.py")),
               V2 / "wireless_exp/llm.py", V2 / "wireless_exp/settings.py",
               *sorted((V2 / "wireless_exp/tkg").glob("*.py"))]
    record["source_sha256"] = {str(p.relative_to(ROOT.parent)): sha256(p) for p in sources}
    for source in sources:
        target = output / "source_snapshot" / source.relative_to(ROOT.parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    write_json(output / "run.json", record)
    baseline_rows = [{"method": "no_agent", "phase": "frozen", **r} for r in read_csv(baseline_dir / "frame_metrics.csv")]
    timeline = read_csv(input_dir / "timeline.csv")
    generation = json.loads((input_dir / "generation.json").read_text(encoding="utf-8"))
    llm = LoggedLLM(settings, output / "llm_calls.jsonl")
    graph = ControlledWirelessGraph(output / "graph.sqlite")
    policy = DynamicTKGPolicy(graph, llm, output, max_iterations=args.max_iterations)
    final_states, all_metrics, pre_metrics, previous = [], list(baseline_rows), [], {}
    started = time.perf_counter()
    try:
        with (output / "frames_pre_control.jsonl").open("w", encoding="utf-8") as pre_file, \
             (output / "frames_post_control.jsonl").open("w", encoding="utf-8") as post_file:
            for timing in timeline:
                index, time_s = int(timing["Frame_Index"]), float(timing["Time_s"])
                for scene in SCENES:
                    prior = previous.get(scene)
                    meta = generation.get("scenarios", {}).get(scene, {})
                    state = graph.sync_frame(scene, index, time_s, read_csv(input_dir / timing[f"{scene}_csv"]),
                                             expected_revision=prior["revision"] if prior else 0,
                                             target_users=args.initial_users,
                                             scene_info={k: meta[k] for k in ("tx_position_m", "geographic_origin_lat_lon", "map_sha256") if k in meta})
                    if index == 0:
                        for key in ("allocations", "contracts", "users", "metrics"):
                            if state[key] != initial_states[scene][key]:
                                raise AssertionError(f"Paired initial state differs: {scene}/{key}")
                    pre_file.write(json.dumps(state, ensure_ascii=False) + "\n")
                    pre_file.flush()
                    pre_metrics.append(metric_row(state, "tkg_dtn", "pre_control"))
                    changed = [uid for uid in state["allocations"]
                               if prior and state["users"][uid]["cqi"] != prior["users"][uid]["cqi"]]
                    decision = {"status": "not_triggered"}
                    if changed:
                        decision = policy.step(scene, {"type": "admitted_cqi_changed", "user_ids": changed})
                    after = graph.current_state(scene)
                    post_file.write(json.dumps(after, ensure_ascii=False) + "\n")
                    post_file.flush()
                    previous[scene] = after
                    final_states.append(after)
                    all_metrics.append(metric_row(after, "tkg_dtn", "post_control"))
                    write_json(output / "progress.json", {"completed_scene_frames": len(final_states),
                               "scenario": scene, "frame_index": index, "time_s": time_s,
                               "llm_statistics": llm.statistics()})
                    print(f"{scene} t={time_s:g}s valid={state['metrics']['effective_users']}->{after['metrics']['effective_users']} "
                          f"decision={decision['status']} calls={llm.calls}", flush=True)
        events = [event for scene in SCENES for event in graph.history(scene)]
        write_json(output / "summary.json", summarize(final_states, events))
        write_csv(output / "frame_metrics.csv", ["method", "phase", *METRIC_FIELDS], all_metrics)
        write_csv(output / "frame_metrics_pre_control.csv", ["method", "phase", *METRIC_FIELDS], pre_metrics)
        write_csv(output / "service_events.csv", ["event_id", "scenario", "frame_index", "time_s", "revision",
                  "user_id", "event_type", "episode_id", "before", "after", "reason"], events)
        episodes = [episode for scene in SCENES for episode in graph.current_state(scene)["episodes"]]
        write_csv(output / "episodes.csv", ["episode_id", "scenario", "user_id", "start_time_s", "end_time_s",
                  "last_observed_time_s", "observed_duration_s", "is_open", "reason", "onset_event_id", "recovery_event_id"], episodes)
        for scene in SCENES:
            write_json(output / f"latest_context_{scene}.json", graph.read_context(scene))
        integrity = graph.db.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok" or graph.db.execute("PRAGMA foreign_key_check").fetchall():
            raise AssertionError("Controlled graph integrity check failed")
        record["graph"] = export_graph(output / "graph.sqlite", output)
        record["graph"]["artifacts"] = {name: name for name in record["graph"]["artifacts"]}
        _, usage = write_report(all_metrics, output, plots=not args.no_plots)
        for key in ("llm_calls", "api_attempts", "prompt_tokens", "completion_tokens"):
            if usage["total"][key] != llm.statistics()[key]:
                raise AssertionError(f"Logged usage disagrees with live gateway counters: {key}")
        # Recheck input/baseline immutability after all live calls and exports.
        baseline_preflight(baseline_dir, input_dir, args.initial_users)
        record.update(status="complete", completed_scene_frames=len(final_states),
                      model_usage=usage["total"], llm_statistics=llm.statistics(),
                      elapsed_wall_seconds=time.perf_counter() - started,
                      ended_at_utc=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        record.update(status="failed", completed_scene_frames=len(final_states),
                      error_type=type(exc).__name__,
                      error=str(exc).replace(settings["api_key"], "[REDACTED]"),
                      llm_statistics=llm.statistics(), elapsed_wall_seconds=time.perf_counter() - started)
        write_json(output / "model_usage_partial.json", usage_summary(output))
        raise
    finally:
        graph.close()
        llm.session.close()
        record["output_sha256"] = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*"))
                                    if p.is_file() and p.name != "run.json"}
        write_json(output / "run.json", record)
    print(f"Completed real paired experiment: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
